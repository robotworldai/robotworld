"""Bounded recovery of failed turns, never replay of simulator actions."""
import json
import time


def failure_kind(error):
    text = json.dumps(error, ensure_ascii=True).lower()
    for code, markers in (
        ('protocol_error', ('protocol_error', 'assistant_prefill_unsupported', 'conflicting cb opaque reasoning')),
        ('request_too_large', ('request_too_large', 'request entity too large', '"httpstatuscode": 413')),
        ('content_policy_violation', ('content_policy', 'content policy', 'content_filter')),
        ('context_length_exceeded', ('context_length_exceeded', 'context window')),
        ('authentication_error', ('unauthorized', 'invalid api key', '"httpstatuscode": 401', '"httpstatuscode": 403')),
    ):
        if any(marker in text for marker in markers):
            return code
    # A structured HTTP error takes precedence over generic disconnect wording.
    def statuses(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('httpStatusCode', 'status') and type(item) is int:
                    yield item
                else:
                    yield from statuses(item)
        elif isinstance(value, list):
            for item in value:
                yield from statuses(item)
    codes = list(statuses(error))
    if any(400 <= code < 500 and code not in (408, 429) for code in codes):
        return 'invalid_request'
    if any(code in (408, 429) or 500 <= code < 600 for code in codes):
        return 'transient_api'
    if any(marker in text for marker in ('rate limit', 'ratelimitexceeded', 'timed out',
                                        'timeout', 'error sending request', 'connection reset',
                                        'stream disconnected', 'responsestreamdisconnected')):
        return 'transient_api'
    return 'unknown'


class TurnRecovery:
    def __init__(self, deadline, events, max_recoveries=3, max_image_recoveries=3):
        self.deadline = deadline
        self.events = events
        self.max_recoveries = max_recoveries
        self.attempts = 0
        self.max_image_recoveries = max_image_recoveries
        self.image_attempts = 0

    def wait(self, error, *, thread, turn, control_step):
        kind = failure_kind(error)
        # Narrow allowance for the observed image-processing service error;
        # keep its classification and do not retry arbitrary filtered outputs.
        image_error = (kind == 'content_policy_violation' and
                       'image processing blocked due to content policy violation'
                       in json.dumps(error, ensure_ascii=True).lower())
        allowed = kind == 'transient_api' or (
            image_error and self.image_attempts < self.max_image_recoveries)
        if not allowed or self.attempts >= self.max_recoveries:
            self.events.write('turn_recovery_stopped', dict(
                reason=kind, recoveries=self.attempts, image_recoveries=self.image_attempts,
                control_step=control_step))
            return False
        delay = min(10 * 2 ** (self.image_attempts if image_error else self.attempts), 60)
        remaining = self.deadline - time.monotonic()
        if remaining <= delay:
            raise TimeoutError('Agent wall timeout during API recovery')
        self.attempts += 1
        if image_error:
            self.image_attempts += 1
        self.events.write('turn_recovery_wait', dict(
            reason=kind, attempt=self.attempts, delay_seconds=delay,
            image_recoveries=self.image_attempts,
            thread_id=thread, failed_turn_id=turn, control_step=control_step))
        time.sleep(delay)
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Agent wall timeout during API recovery')
        return True
