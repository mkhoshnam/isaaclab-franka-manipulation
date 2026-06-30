# Train the Franka cabinet open/place task with rsl_rl (self-contained, no gym
# registration needed).
#
#   Phase 1 (open):  ./isaaclab.sh -p train_rsl_rl.py --num_envs 4096 --headless --max_iterations 1500
#   Phase 2 (place): ./isaaclab.sh -p train_rsl_rl.py --num_envs 4096 --headless --max_iterations 3000 \
#                        --enable_place --init_checkpoint logs/rsl_rl/franka_cabinet_open/<run>/model_1499.pt
#
# Isaac Sim must launch (AppLauncher) BEFORE importing isaaclab / rsl_rl modules.

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka cabinet with rsl_rl.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--enable_place", action="store_true",
                    help="enable the pick-and-place stage (default: open only)")
parser.add_argument("--init_checkpoint", type=str, default="",
                    help="warm-start from an rsl_rl checkpoint (e.g. the open policy)")
parser.add_argument("--experiment_name", type=str, default="")
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
    # --- environment (your file; unchanged) ---
    env_cfg = FrankaCabinetPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.enable_place = args_cli.enable_place

    print("=" * 70)
    print(f"[TASK] enable_place = {env_cfg.enable_place}  ->  "
          f"{'PLACE reward ACTIVE (open + pick + place)' if env_cfg.enable_place else 'OPEN-ONLY (cube/place reward OFF)'}")
    print("=" * 70)

    
    # Force Isaac Lab environment/device to match CLI device
    env_cfg.sim.device = args_cli.device

    env = FrankaCabinetPlaceEnv(env_cfg, render_mode=None)
    device = env.device
    env = RslRlVecEnvWrapper(env)   # must be the last wrapper

    # --- agent / runner config ---
    agent_cfg = FrankaCabinetPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    if args_cli.experiment_name:
        agent_cfg.experiment_name = args_cli.experiment_name
    elif args_cli.enable_place:
        agent_cfg.experiment_name = "franka_cabinet_place"

    log_dir = os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    cfg_dict = agent_cfg.to_dict()
    # rsl-rl >= 4.0 hangs in OnPolicyRunner init without obs_groups (IsaacLab #5363)
    cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})

    runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir, device=device)
    if args_cli.init_checkpoint:
        runner.load(args_cli.init_checkpoint)
        print(f"warm-started from {args_cli.init_checkpoint}")

    # init_at_random_ep_len staggers episode boundaries across envs -> no more
    # synchronized resets / value-spike thrashing.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"done. checkpoints in {log_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
