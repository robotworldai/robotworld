"""Explicit diagnostic profiles; independent of the frozen native protocol file."""
PROFILES = {
    "legacy": dict(native_steps=500, motion_requests=40, tool_calls=96,
                   wall_seconds=600, allow_give_up=True, max_requests=110,
                   request_bytes=32 * 1024**2, launcher_seconds=2100,
                   api_retries=0, model_idle_seconds=300),
    "steps10000": dict(native_steps=10000, motion_requests=10000, tool_calls=20000,
                       wall_seconds=None, allow_give_up=False, max_requests=21000,
                       request_bytes=256 * 1024**2, launcher_seconds=88200,
                       api_retries=3, model_idle_seconds=480),
}
PROFILES["steps5000"] = dict(PROFILES["steps10000"], native_steps=5000)
DEFAULT_BATCH_PROFILE = "steps5000"


def get_budget(name):
    return dict(PROFILES[name])


def configure_protocol(protocol, name):
    profile = get_budget(name)
    protocol.MAX_STEPS = profile["native_steps"]
    protocol.MAX_COMMANDS = profile["motion_requests"]
    protocol.MAX_TOOLS = profile["tool_calls"]
    return protocol
