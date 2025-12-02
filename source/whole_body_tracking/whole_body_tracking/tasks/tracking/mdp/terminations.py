"""
终止条件模块

该模块定义了全身追踪任务的终止条件函数。
当机器人的追踪误差超过阈值时，episode会提前终止，避免无效的学习。

主要功能：
1. 锚点位置/方向偏差检测
2. 身体部位位置偏差检测
3. 提前终止失败的追踪尝试

终止条件设计原则：
- 使用阈值判断（threshold）：误差超过阈值 → 终止
- 返回布尔张量：True = 终止，False = 继续
- 帮助加速训练：快速跳过失败案例
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    检查锚点位置偏差是否超过阈值（3D欧式距离）
    
    当机器人根部锚点与目标锚点的距离过大时，认为追踪失败。
    这是最基本的终止条件，防止机器人偏离轨迹太远。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称（通常为 "motion"）
        threshold: 距离阈值（米）
    
    返回:
        torch.Tensor: [num_envs,] - 布尔张量
            - True: 位置偏差超过阈值，需要终止
            - False: 位置偏差在允许范围内，继续训练
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算3D欧式距离并与阈值比较
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    检查锚点高度偏差是否超过阈值（仅Z轴）
    
    只检查垂直方向的偏差，对水平偏差更宽容。
    适用于需要严格控制高度但允许水平漂移的场景。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        threshold: 高度差阈值（米）
    
    返回:
        torch.Tensor: [num_envs,] - 布尔张量
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 只取Z轴（最后一维），计算高度差的绝对值
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """
    检查锚点方向偏差是否超过阈值（使用重力投影法）
    
    通过比较"重力在机器人坐标系下的投影"来判断方向偏差。
    这种方法比直接比较四元数更直观。
    
    参数:
        env: 强化学习环境实例
        asset_cfg: 资产配置（机器人）
        command_name: 运动指令的名称
        threshold: 方向偏差阈值（范围 [0, 2]）
    
    返回:
        torch.Tensor: [num_envs,] - 布尔张量
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 将重力向量转换到目标锚点坐标系
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    
    # 将重力向量转换到参考锚点坐标系
    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    
    # 比较Z分量差值
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    检查身体部位位置偏差是否超过阈值（3D距离）
    
    当任意指定的身体部位与参考位置的距离过大时，终止episode。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        threshold: 距离阈值（米）
        body_names: 要检查的身体部位列表（None = 所有部位）
    
    返回:
        torch.Tensor: [num_envs,] - 布尔张量
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    
    # 计算每个身体部位的位置误差
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    
    # 检查是否有任意身体部位超过阈值
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    检查身体部位高度偏差是否超过阈值（仅Z轴）
    
    只检查身体部位的垂直偏差，对水平偏差更宽容。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        threshold: 高度差阈值（米）
        body_names: 要检查的身体部位列表
    
    返回:
        torch.Tensor: [num_envs,] - 布尔张量
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    
    # 计算每个身体部位的高度误差（只取Z轴）
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    
    # 检查是否有任意身体部位的高度超过阈值
    return torch.any(error > threshold, dim=-1)
