"""
观测函数模块

该模块定义了全身追踪任务的所有观测函数。
这些函数从运动指令(MotionCommand)中提取参考状态，用于策略网络的输入。

主要功能：
1. 提取参考运动的锚点状态（位置、方向、速度）
2. 提取参考身体各部位的姿态
3. 计算当前状态与目标状态的差异

坐标系说明：
- _w: 世界坐标系 (World Frame)
- _b: 机器人基座坐标系 (Body/Robot Frame)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取参考运动的锚点方向（世界坐标系）
    
    锚点(anchor)通常是机器人的根部或质心，作为追踪的参考点。
    使用6D旋转表示法（旋转矩阵的前两列），比四元数更适合神经网络学习。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称（通常为 "motion"）
    
    返回:
        torch.Tensor: [num_envs, 6] - 每个环境的参考锚点方向
            - 6维来自旋转矩阵的前两列：mat[:, :, 0] 和 mat[:, :, 1]
            - 第三列可由前两列叉乘得到（保证正交性）
    
    流程:
        1. 从指令管理器获取 MotionCommand 实例
        2. 获取参考锚点四元数 robot_anchor_quat_w [N, 4]
        3. 转换为旋转矩阵 [N, 3, 3]
        4. 提取前两列 [N, 3, 2] 并展平为 [N, 6]
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)  # [N, 3, 3]
    return mat[..., :2].reshape(mat.shape[0], -1)  # [N, 6]


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取参考运动的锚点线速度（世界坐标系）
    
    提供机器人根部应该具有的线速度目标，用于速度追踪。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
    
    返回:
        torch.Tensor: [num_envs, 3] - 每个环境的参考锚点线速度 (vx, vy, vz)
    
    流程:
        1. 从 MotionCommand 获取 robot_anchor_vel_w [N, 6]
            - robot_anchor_vel_w[:, :3]  -> 线速度
            - robot_anchor_vel_w[:, 3:6] -> 角速度
        2. 提取前3维（线速度分量）
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)  # [N, 3]


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取参考运动的锚点角速度（世界坐标系）
    
    提供机器人根部应该具有的角速度目标，用于旋转追踪。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
    
    返回:
        torch.Tensor: [num_envs, 3] - 每个环境的参考锚点角速度 (ωx, ωy, ωz)
    
    流程:
        1. 从 MotionCommand 获取 robot_anchor_vel_w [N, 6]
        2. 提取后3维（角速度分量）
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)  # [N, 3]


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取参考身体各部位的位置（机器人坐标系）
    
    将世界坐标系下的各身体部位位置转换到机器人锚点的局部坐标系。
    这样可以提供姿态不变的观测（无论机器人朝向如何）。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
    
    返回:
        torch.Tensor: [num_envs, num_bodies*3] - 所有身体部位的相对位置
            - 例如：num_bodies=10，则返回 [N, 30]
            - 每3个值代表一个身体部位的 (x, y, z) 坐标
    
    流程:
        1. 获取身体部位数量 num_bodies
        2. 使用 subtract_frame_transforms 计算坐标变换：
           pos_b = T_anchor^(-1) * pos_world
        3. 将 [N, num_bodies, 3] 展平为 [N, num_bodies*3]
    
    数学原理:
        subtract_frame_transforms(anchor_pos, anchor_quat, body_pos, body_quat)
        返回从 anchor 坐标系看到的 body 的相对位置和姿态
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)  # 身体部位数量
    
    # 计算从锚点坐标系观察的身体位置
    # repeat: 将锚点状态扩展到每个身体部位 [N, 1, 3] -> [N, num_bodies, 3]
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),   # [N, num_bodies, 3]
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),  # [N, num_bodies, 4]
        command.robot_body_pos_w,    # [N, num_bodies, 3]
        command.robot_body_quat_w,   # [N, num_bodies, 4]
    )

    return pos_b.view(env.num_envs, -1)  # [N, num_bodies*3]


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取参考身体各部位的方向（机器人坐标系）
    
    将世界坐标系下的各身体部位方向转换到机器人锚点的局部坐标系，
    同样使用6D旋转表示法。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
    
    返回:
        torch.Tensor: [num_envs, num_bodies*6] - 所有身体部位的相对方向
            - 例如：num_bodies=10，则返回 [N, 60]
            - 每6个值代表一个身体部位的6D旋转表示
    
    流程:
        1. 获取身体部位数量 num_bodies
        2. 使用 subtract_frame_transforms 计算姿态差
        3. 将四元数转换为旋转矩阵
        4. 提取前两列并展平
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    
    # 计算从锚点坐标系观察的身体方向
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),   # [N, num_bodies, 3]
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),  # [N, num_bodies, 4]
        command.robot_body_pos_w,    # [N, num_bodies, 3]
        command.robot_body_quat_w,   # [N, num_bodies, 4]
    )
    
    # 转换为6D旋转表示
    mat = matrix_from_quat(ori_b)  # [N, num_bodies, 3, 3]
    return mat[..., :2].reshape(mat.shape[0], -1)  # [N, num_bodies*6]


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    计算目标锚点位置与当前锚点的位置差（机器人坐标系）
    
    这是一个关键的追踪误差信号，告诉策略"需要往哪个方向移动"。
    在机器人局部坐标系下表示，提供方向不变性。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
    
    返回:
        torch.Tensor: [num_envs, 3] - 每个环境的目标位置差 (dx, dy, dz)
            - 正值表示目标在该方向的正方向
            - 策略应该尝试减小这个差值
    
    流程:
        1. 从 MotionCommand 获取：
           - robot_anchor_pos_w: 当前参考锚点位置（机器人应该在的位置）
           - anchor_pos_w: 运动数据中的目标锚点位置
        2. 计算相对位置差（在 robot_anchor 坐标系下）
    
    应用场景:
        - 如果 motion_anchor_pos_b = [0.1, 0, 0]，表示需要向前移动0.1米
        - 策略会学习产生向前的动作来减小这个误差
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # 计算目标位置相对于当前参考锚点的偏移
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,   # 当前参考锚点（坐标系原点）
        command.robot_anchor_quat_w,  # 当前参考方向（坐标系方向）
        command.anchor_pos_w,         # 目标锚点位置（世界系）
        command.anchor_quat_w,        # 目标锚点方向（世界系）
    )

    return pos.view(env.num_envs, -1)  # [N, 3]


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    计算目标锚点方向与当前锚点的方向差（机器人坐标系）
    
    这是方向追踪的误差信号，告诉策略"需要往哪个方向旋转"。
    使用6D旋转表示法，避免四元数的不连续性问题。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
    
    返回:
        torch.Tensor: [num_envs, 6] - 每个环境的目标方向差（6D表示）
            - 策略应该尝试减小这个方向差
    
    流程:
        1. 从 MotionCommand 获取当前参考方向和目标方向
        2. 计算相对旋转（在 robot_anchor 坐标系下）
        3. 转换为6D旋转表示
    
    应用场景:
        - 如果目标方向与当前方向有偏差，这个观测会提供旋转误差
        - 策略会学习产生旋转动作来对齐方向
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # 计算目标方向相对于当前参考方向的旋转差
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,   # 当前参考锚点位置
        command.robot_anchor_quat_w,  # 当前参考方向（坐标系原点）
        command.anchor_pos_w,         # 目标锚点位置
        command.anchor_quat_w,        # 目标锚点方向（目标旋转）
    )
    
    # 转换为6D旋转表示
    mat = matrix_from_quat(ori)  # [N, 3, 3]
    return mat[..., :2].reshape(mat.shape[0], -1)  # [N, 6]
