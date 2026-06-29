"""Franka RL manipulation package for Isaac Lab."""

from .envs.franka_cabinet_place_env import (
    FrankaCabinetPlaceEnv,
    FrankaCabinetPlaceEnvCfg,
)

__all__ = ["FrankaCabinetPlaceEnv", "FrankaCabinetPlaceEnvCfg"]
