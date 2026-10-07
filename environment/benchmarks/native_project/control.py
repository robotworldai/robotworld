"""Native action validation without imposing fictitious normalized limits."""
import math


def validate_action(value, metadata):
    if not isinstance(value, list) or len(value) != metadata['dim']:
        raise ValueError(f"Expected {metadata['dim']} native action scalars")
    result = []
    for i, x in enumerate(value):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError('Actions must be finite numbers')
        lo, hi = metadata['lower'][i], metadata['upper'][i]
        if lo is not None and x < lo or hi is not None and x > hi:
            raise ValueError(f'Action {i} outside native bounds [{lo}, {hi}]')
        result.append(float(x))
    return result


def specs(metadata, coding=True, action_guide=None):
    def tool(name, description, properties):
        return {'name': name, 'description': description, 'inputSchema': {
            'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}}
    note = {'type': 'string'}
    result = [tool('observe', 'Read current native policy observation without advancing physics.', {'note': note}),
              tool('apply_action', 'Execute the simultaneous native action vector for the requested control steps. Follow the zero-based ACTION CONTRACT in system instructions for each channel, physical gain, zero value and sign; these are not generic strength percentages. ' + str(metadata),
                   {'note': note, 'action': {'type': 'array', 'items': {'type': 'number'},
                     'minItems': metadata['dim'], 'maxItems': metadata['dim']},
                    'steps': {'type': 'integer', 'minimum': 1, 'maximum': 50}})]
    if coding:
        result.append(tool('coding_control', 'Run your feedback program at every native control step. '
                           'Define control(obs,memory), return {"action":[native values]} or {"done":true}. '
                           'Use a single control function; helper/nested function definitions are unsupported. '
                           'Only provided observations and JSON memory are available; no simulator access.',
                           {'note': note, 'code': {'type': 'string', 'maxLength': 16000},
                            'max_steps': {'type': 'integer', 'minimum': 1, 'maximum': 250}}))
    if action_guide:
        from environment.benchmarks.action_contracts import describe_tools
        result = describe_tools(result, action_guide)
    return result
