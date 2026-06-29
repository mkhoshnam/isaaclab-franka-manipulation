# Watch an rsl_rl-trained cabinet policy.
#
#   ./isaaclab.sh -p play_rsl_rl.py --num_envs 16 \
#       --checkpoint logs/rsl_rl/franka_cabinet_open/<run>/model_1499.pt

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play an rsl_rl Franka cabinet policy.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--enable_place", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from franka_cabinet_place_env import FrankaCabinetPlaceEnv, FrankaCabinetPlaceEnvCfg  # noqa: E402
from rsl_rl_ppo_cfg import FrankaCabinetPPORunnerCfg  # noqa: E402


def main():
    env_cfg = FrankaCabinetPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.enable_place = args_cli.enable_place

    env = FrankaCabinetPlaceEnv(env_cfg, render_mode="human")
    device = env.device
    env = RslRlVecEnvWrapper(env)

    agent_cfg = FrankaCabinetPPORunnerCfg()
    agent_cfg.device = device
    cfg_dict = agent_cfg.to_dict()
    cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})

    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    out = env.get_observations()
    obs = out[0] if isinstance(out, tuple) else out
    for _ in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(obs)
        step_out = env.step(actions)
        obs = step_out[0]

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
