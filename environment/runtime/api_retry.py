"""Bounded HTTP-open retries only; never replay a stream or robot command."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import errno
import socket
import time
import urllib.error


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
NETWORK_ERRNOS = {errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
                  errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE}


class RetryCancelled(RuntimeError):
    pass


def retryable_transport(error):
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    return (isinstance(reason, (TimeoutError, ConnectionError)) or
            isinstance(reason, socket.gaierror) and reason.errno == socket.EAI_AGAIN or
            isinstance(reason, OSError) and reason.errno in NETWORK_ERRNOS)


def retry_delay(response, attempt):
    delay = min(2 ** attempt, 30)
    value = response.headers.get("Retry-After") if response is not None else None
    if value:
        try:
            seconds = float(value) if value.isdigit() else (
                parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            if seconds > 30:
                return None  # Respect long server backoffs instead of retrying too early.
            delay = max(delay, max(0, seconds))
        except (ValueError, TypeError, OverflowError):
            pass
    return delay


def open_with_retry(opener, request, *, max_retries=3, timeout=90,
                    attempts, persist, cancelled=lambda: False, sleep=time.sleep):
    """Same bytes/headers on each attempt; returned response is never retried here."""
    for index in range(max_retries + 1):
        if cancelled():
            raise RetryCancelled("Controller cancelled request before upstream attempt")
        row = dict(attempt=index + 1, started_at=time.time(), status=None,
                   request_body_unchanged=True, response_forwarded=False)
        attempts.append(row)
        persist()
        response = error = None
        try:
            try:
                response = opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as caught:
                response = caught
            row["status"] = response.status
            retryable = response.status in RETRYABLE_STATUS
        except Exception as caught:
            error = caught
            row["error_type"] = type(caught).__name__
            retryable = retryable_transport(caught)
        row["finished_at"] = time.time()
        delay = retry_delay(response, index + 1) if retryable else None
        retry = retryable and index < max_retries and delay is not None and not cancelled()
        row.update(retryable=retryable, will_retry=retry)
        if retry:
            row["backoff_seconds"] = delay
        persist()
        if not retry:
            if error is not None:
                raise error
            return response
        if response is not None:
            response.close()
        # Short sleeps let controller cancellation prevent an unnecessary billable retry.
        remaining = delay
        while remaining > 0:
            if cancelled():
                raise RetryCancelled("Controller cancelled upstream backoff")
            interval = min(remaining, .25)
            sleep(interval)
            remaining -= interval
