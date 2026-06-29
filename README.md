# Franka RL Manipulation for Isaac Lab

This repository contains a Franka manipulation reinforcement learning setup for Isaac Lab using rsl_rl. The main environment is a long-horizon cabinet task that stages the problem into:

1. Opening a cabinet drawer.
2. Reaching and grasping a cube.
3. Placing it inside the drawer.

The environment is implemented in [franka_cabinet_place_env.py](franka_cabinet_place_env.py) and is intended for research and experimentation with long-horizon, multi-stage manipulation in Isaac Sim.

## Included files

- `franka_cabinet_place_env.py` — Isaac Lab DirectRLEnv for the cabinet open/place task
- `train_rsl_rl.py` — training entry point for rsl_rl
- `play_rsl_rl.py` — policy playback entry point
- `rsl_rl_ppo_cfg.py` — PPO hyperparameters for the environment
- `install_denver_isaac.sh` — installation helper for the Isaac Lab environment
- `isaaclab_exact_requirements.txt` — pinned dependencies
- `open_rsl_rl_good.pt` — example checkpoint file

## Quick start

1. Create and activate the Isaac Lab Python environment.
2. Run training:

```bash
./isaaclab.sh -p train_rsl_rl.py --num_envs 4096 --headless --max_iterations 1500
```

3. To enable the place stage after training the open phase, use:

```bash
./isaaclab.sh -p train_rsl_rl.py --num_envs 4096 --headless --max_iterations 3000 \
  --enable_place --init_checkpoint logs/rsl_rl/franka_cabinet_open/<run>/model_1499.pt
```

4. To play a trained checkpoint:

```bash
./isaaclab.sh -p play_rsl_rl.py --checkpoint logs/rsl_rl/franka_cabinet_open/<run>/model_1499.pt
```

## Notes

- This setup is intended for Isaac Lab + rsl_rl.
- The task is deliberately staged to make the long-horizon behavior easier to learn than a single monolithic reward.
- The environment can be extended to additional cabinet or manipulation tasks in the same framework.
