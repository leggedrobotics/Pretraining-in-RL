from isaaclab.utils import configclass
from p4rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoResidualActorCriticCfg


@configclass
class AnymalDRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough"
    empirical_normalization = False
    policy = RslRlPpoResidualActorCriticCfg(
        num_residual_blocks=10,
        residual_block_hidden_layers=2,
        residual_block_hidden_dim=128,
        init_noise_std=1.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalDFlatPPORunnerCfg(AnymalDRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat"
        self.policy=RslRlPpoResidualActorCriticCfg(
                num_residual_blocks=10,
                residual_block_hidden_layers=2,
                residual_block_hidden_dim=128,
                init_noise_std=1.0,
                activation="elu",
            )
