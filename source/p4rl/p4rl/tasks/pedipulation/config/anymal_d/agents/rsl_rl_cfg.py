# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from p4rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoCommandedDeepActorCriticCfg
from rsl_rl.addons.kinematics.modules import KinematicSubmoduleConfig




@configclass
class AnymalDFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 250
    experiment_name = "anymal_d_pedipulation_flat"
    empirical_normalization = False
    policy = RslRlPpoCommandedDeepActorCriticCfg(
        num_residual_blocks=8,
        residual_block_hidden_layers=2,
        residual_block_hidden_dim=128,
        command_dim=3,
        final_mlp_dims=[128, 128, 128],
        init_noise_std=1.0,
        activation="elu",
    ) 
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
        # optimizer="SGD",
    )


@configclass
class AnymalDPreKineFlatPPORunnerCfg(AnymalDFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.pretrained_module_config = KinematicSubmoduleConfig(input_dim = 12,
                                                            num_bodies = 17,
                                                            num_output_features_per_body= 6,
                                                            hidden_dim = 256,
                                                            input_slice= slice(12, 24), # must be consistent with the ObservationCfg! 
                                                            weight_path= "./logs/pretrain/kinematic_submodule.pt")


@configclass
class AnymalDRoughPPORunnerCfg(AnymalDFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_d_pedipulation_rough"
        self.max_iterations = 10000