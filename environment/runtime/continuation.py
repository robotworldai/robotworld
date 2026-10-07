"""Reuse the last delivered packet on a text-only continuation, not a new sample."""
import copy
from functools import wraps

FEEDBACK = ('The episode is still active. Text answers and offline analysis do not '
            'advance simulation. Use the available robot tools to execute your plan '
            'when the observations are sufficient. Obtain a new viewpoint through '
            'legal robot actions. Necessary analysis is allowed. The following packet '
            'is the last observation, not a newly executed action.')


def remember_packet(step):
    def decorate(function):
        @wraps(function)
        def capture(self, *args, **kwargs):
            packet = function(self, *args, **kwargs)
            self._continuation_packet = (step(self), copy.deepcopy(packet))
            return packet
        return capture
    return decorate


def continue_packet(owner, step, capture):
    saved = getattr(owner, '_continuation_packet', None)
    packet = copy.deepcopy(saved[1]) if saved and saved[0] == step else capture()
    return [{'type':'inputText', 'text':FEEDBACK}] + packet
