"""Simulation environments for Aerial-Balance-Bench."""

__all__ = ["AerialBalanceEnv", "AerialBalanceEnvCfg"]


def __getattr__(name: str):
    if name in __all__:
        from .aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg

        return {"AerialBalanceEnv": AerialBalanceEnv, "AerialBalanceEnvCfg": AerialBalanceEnvCfg}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
