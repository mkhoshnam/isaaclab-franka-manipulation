# rsl_rl training configuration for the Franka cabinet open/place task.
#
# This is the ONLY place the training hyperparameters live. The environment
# (reward, observations, curriculum, place stage) stays entirely in
# franka_cabinet_place_env.py and is untouched by rsl_rl.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FrankaCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24            # rollout horizon per env per update
    max_iterations = 1500
    save_interval = 100
    experiment_name = "franka_cabinet_open"
    empirical_normalization = True    # rsl_rl normalizes observations for us

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,           # lowered: was letting the noise std blow up to ~4
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,           # lowered for value-function stability (was diverging)
        schedule="adaptive",          # KL-adaptive LR, done correctly by rsl_rl
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
