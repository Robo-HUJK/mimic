"""
奖励函数模块

该模块定义了全身追踪任务的所有奖励函数。
奖励函数用于引导策略学习，使机器人能够准确追踪参考运动。

主要功能：
1. 锚点追踪奖励（位置和方向）
2. 身体部位追踪奖励（位置、方向、速度）
3. 接触时间奖励（足部着地控制）

奖励计算方式：
- 使用指数衰减形式：exp(-error/std²)
- error 越小，奖励越接近 1
- error 越大，奖励快速衰减到 0
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """
    辅助函数：获取指定身体部位的索引
    
    参数:
        command: 运动指令实例
        body_names: 要获取的身体部位名称列表
                   如果为 None，则返回所有身体部位的索引
    
    返回:
        list[int]: 身体部位在配置中的索引列表
    
    示例:
        body_names = ["left_foot", "right_foot"]
        返回 [5, 10]  # 假设这两个部位在配置中的索引
    """
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """
    计算锚点位置追踪的奖励（世界坐标系）
    
    奖励机器人根部锚点靠近目标锚点的位置。
    这是最基础的追踪奖励，确保机器人整体移动轨迹正确。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称（通常为 "motion"）
        std: 标准差参数，控制奖励衰减速度
             - std 越小，对误差越敏感（要求更精确）
             - std 越大，对误差越宽容
    
    返回:
        torch.Tensor: [num_envs,] - 每个环境的位置追踪奖励 [0, 1]
            - 1.0: 完美追踪（位置完全一致）
            - 0.0: 追踪失败（位置偏差很大）
    
    计算公式:
        error = ||anchor_pos - robot_anchor_pos||²
        reward = exp(-error / std²)
    
    物理意义:
        - anchor_pos_w: 目标锚点位置（未来应该到的位置）
        - robot_anchor_pos_w: 当前参考锚点位置（现在应该在的位置）
        - 差值越小，说明追踪越准确
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算位置误差的平方和
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    # 使用指数衰减将误差转换为奖励
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """
    计算锚点方向追踪的奖励（世界坐标系）
    
    奖励机器人根部锚点的方向与目标方向对齐。
    确保机器人不仅在正确的位置，还面向正确的方向。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        std: 标准差参数，控制奖励衰减速度
    
    返回:
        torch.Tensor: [num_envs,] - 每个环境的方向追踪奖励 [0, 1]
    
    计算公式:
        error = quat_error_magnitude(anchor_quat, robot_anchor_quat)²
        reward = exp(-error / std²)
    
    说明:
        quat_error_magnitude 计算两个四元数之间的角度差
        返回值范围 [0, π]，平方后用于奖励计算
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算四元数方向误差（角度差的平方）
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    计算身体部位相对位置追踪的奖励
    
    奖励各个身体部位相对于锚点的位置与参考运动一致。
    这确保了机器人的"姿态形状"正确（例如：手臂、腿的摆放）。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        std: 标准差参数
        body_names: 要考察的身体部位列表
                   None 表示考察所有身体部位
                   例如：["left_hand", "right_hand"] 只关注双手
    
    返回:
        torch.Tensor: [num_envs,] - 身体部位位置追踪奖励 [0, 1]
    
    计算公式:
        error = mean(||body_pos_relative - robot_body_pos||²)
        reward = exp(-error / std²)
    
    关键点:
        - body_pos_relative_w: 参考身体位置（相对于目标锚点）
        - robot_body_pos_w: 当前身体位置（相对于参考锚点）
        - 使用相对位置可以分离整体移动和局部姿态
        - mean(-1) 对所有选定的身体部位取平均
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算选定身体部位的位置误差
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    # 对所有身体部位取平均误差，然后计算奖励
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    计算身体部位相对方向追踪的奖励
    
    奖励各个身体部位的方向与参考运动一致。
    例如：确保手掌朝向正确、脚掌角度正确等。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        std: 标准差参数
        body_names: 要考察的身体部位列表（None = 所有部位）
    
    返回:
        torch.Tensor: [num_envs,] - 身体部位方向追踪奖励 [0, 1]
    
    计算公式:
        error = mean(quat_error_magnitude(body_quat_relative, robot_body_quat)²)
        reward = exp(-error / std²)
    
    应用场景:
        - 模仿手势动作时，手掌方向很重要
        - 模仿舞蹈动作时，身体各部位的扭转角度需要精确
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算选定身体部位的方向误差
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    # 对所有身体部位取平均误差，然后计算奖励
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    计算身体部位线速度追踪的奖励（世界坐标系）
    
    奖励各个身体部位的运动速度与参考运动一致。
    确保动作的"动态特性"正确（不只是静态姿态，还有运动快慢）。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        std: 标准差参数
        body_names: 要考察的身体部位列表
    
    返回:
        torch.Tensor: [num_envs,] - 身体部位速度追踪奖励 [0, 1]
    
    计算公式:
        error = mean(||body_lin_vel - robot_body_lin_vel||²)
        reward = exp(-error / std²)
    
    物理意义:
        - 不仅要求身体部位到达正确位置
        - 还要求以正确的速度到达
        - 例如：挥手动作要快，缓慢抬腿要慢
    
    应用场景:
        - 快速动作（出拳、踢腿）需要速度匹配
        - 平滑动作（太极、瑜伽）需要速度控制
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算身体部位线速度的误差
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    计算身体部位角速度追踪的奖励（世界坐标系）
    
    奖励各个身体部位的旋转速度与参考运动一致。
    控制身体部位的"旋转快慢"，例如扭腰的速度、转头的速度。
    
    参数:
        env: 强化学习环境实例
        command_name: 运动指令的名称
        std: 标准差参数
        body_names: 要考察的身体部位列表
    
    返回:
        torch.Tensor: [num_envs,] - 身体部位角速度追踪奖励 [0, 1]
    
    计算公式:
        error = mean(||body_ang_vel - robot_body_ang_vel||²)
        reward = exp(-error / std²)
    
    物理意义:
        - body_ang_vel_w: 参考运动的角速度 (ωx, ωy, ωz)
        - robot_body_ang_vel_w: 当前实际的角速度
        - 差值越小，旋转速度越匹配
    
    应用场景:
        - 快速转身动作需要高角速度
        - 缓慢扭转动作需要低角速度
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算身体部位角速度的误差
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    计算足部接触时间奖励
    
    奖励足部在离地后短时间内重新接触地面。
    这用于鼓励快速的步态（例如跑步），避免足部长时间悬空。
    
    参数:
        env: 强化学习环境实例
        sensor_cfg: 接触传感器配置
                   - name: 传感器名称（通常为 "contact_sensor"）
                   - body_ids: 要检测的身体部位ID（通常是双脚）
        threshold: 时间阈值（秒）
                  例如：0.1 表示足部在离地后 0.1 秒内重新着地才给奖励
    
    返回:
        torch.Tensor: [num_envs,] - 接触时间奖励（整数，计数）
            - 返回满足条件的足部数量
            - 例如：2.0 表示两只脚都满足快速着地条件
    
    计算逻辑:
        1. first_air: 判断足部是否刚刚离地（布尔值）
        2. last_contact_time: 上次接触地面到现在的时间
        3. 如果 last_contact_time < threshold 且 first_air = True
           说明足部快速重新着地，给予奖励
    
    应用场景:
        - 鼓励快速步态（跑步、快走）
        - 防止机器人拖着脚走路
        - 提高运动效率
    
    示例:
        threshold = 0.1
        左脚离地 0.05 秒后着地 → 奖励 +1
        右脚离地 0.15 秒后着地 → 奖励 +0（超过阈值）
        总奖励 = 1.0
    """
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # first_air: 判断是否刚从接触状态变为空中状态
    # [num_envs, num_bodies] - 布尔值张量
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    
    # last_contact_time: 上次接触到现在的时间（秒）
    # [num_envs, num_bodies] - 浮点数张量
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    
    # 计算奖励：离地后快速重新接触地面的足部数量
    # (last_contact_time < threshold): 接触时间短
    # * first_air: 且刚刚离地
    # sum(dim=-1): 统计满足条件的足部数量
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    
    return reward
