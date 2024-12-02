# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def base_yaw_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_up_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_rotate(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)

def ball_position_(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Returns the x,y ball position in the simulation world frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the ball entity.
        
    Returns:
        torch.Tensor: Ball position in world frame with shape (num_envs, 2).
    """
    # extract the used quantities (to enable type-hinting)
    ball = env.scene[asset_cfg.name]
    # get ball position in world frame
    return ball.data.root_pos_w[:, :2]


def cube_position_(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Returns the x,y cube position in the simulation world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the ball entity.

    Returns:
        torch.Tensor: Cube position in world frame with shape (num_envs, 2).
    """
    # extract the used quantities (to enable type-hinting)
    cube = env.scene[asset_cfg.name]
    # get cube position in world frame
    return cube.data.root_pos_w[:, :2]