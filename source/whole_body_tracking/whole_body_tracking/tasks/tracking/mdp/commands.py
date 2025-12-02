from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
运动指令模块

该模块定义了运动追踪任务的核心组件：
1. MotionLoader: 加载和管理运动捕捉数据（.npz文件）
2. MotionCommand: 运动指令管理器，负责采样、更新和可视化
3. MotionCommandCfg: 运动指令的配置类

主要功能：
- 从.npz文件加载参考运动数据
- 自适应采样（AMP-based curriculum learning）
- 提供当前时刻的参考状态（关节、身体部位）
- 可视化当前状态与目标状态的对比

数据流：
motion.npz → MotionLoader → MotionCommand → observations/rewards
"""

class MotionLoader:
    """
    运动数据加载器
    
    从.npz文件加载运动捕捉数据，并提供按时间步索引的访问接口。
    
    数据格式：
    - joint_pos: [time_steps, num_joints] - 关节位置
    - joint_vel: [time_steps, num_joints] - 关节速度
    - body_pos_w: [time_steps, num_bodies, 3] - 身体部位位置（世界系）
    - body_quat_w: [time_steps, num_bodies, 4] - 身体部位方向（世界系）
    - body_lin_vel_w: [time_steps, num_bodies, 3] - 身体部位线速度
    - body_ang_vel_w: [time_steps, num_bodies, 3] - 身体部位角速度
    - fps: 运动数据的采样频率
    """
    
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        """
        初始化运动数据加载器
        
        参数:
            motion_file: .npz文件路径
            body_indexes: 要使用的身体部位索引列表
            device: 张量所在设备 ("cpu" 或 "cuda")
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """获取选定身体部位的位置 [time_steps, num_selected_bodies, 3]"""
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """获取选定身体部位的方向 [time_steps, num_selected_bodies, 4]"""
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """获取选定身体部位的线速度 [time_steps, num_selected_bodies, 3]"""
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """获取选定身体部位的角速度 [time_steps, num_selected_bodies, 3]"""
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    """
    运动指令管理器
    
    负责管理运动数据的采样、更新和可视化。
    实现了自适应课程学习（Adaptive Curriculum Learning），
    优先采样困难的时间段来加速训练。
    
    核心功能：
    1. 自适应采样：根据失败率动态调整采样概率
    2. 状态提供：提供当前时刻的参考状态
    3. 误差计算：计算追踪误差指标
    4. 可视化：显示当前状态与目标状态
    """
    
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        """
        初始化运动指令管理器
        
        参数:
            cfg: 运动指令配置
            env: 强化学习环境实例
        """
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """获取完整的运动指令（关节位置+速度）[num_envs, num_joints*2]"""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    # ===== 参考运动数据属性 =====
    
    @property
    def joint_pos(self) -> torch.Tensor:
        """参考关节位置 [num_envs, num_joints]"""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """参考关节速度 [num_envs, num_joints]"""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """参考身体部位位置（世界系）[num_envs, num_bodies, 3]"""
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """参考身体部位方向（世界系）[num_envs, num_bodies, 4]"""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """参考身体部位线速度（世界系）[num_envs, num_bodies, 3]"""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """参考身体部位角速度（世界系）[num_envs, num_bodies, 3]"""
        return self.motion.body_ang_vel_w[self.time_steps]

    # ===== 参考锚点数据属性 =====
    
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """参考锚点位置（世界系）[num_envs, 3] - 目标锚点（未来应该到的位置）"""
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """参考锚点方向（世界系）[num_envs, 4]"""
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """参考锚点线速度（世界系）[num_envs, 3]"""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """参考锚点角速度（世界系）[num_envs, 3]"""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    # ===== 机器人当前状态属性 =====
    
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """机器人当前关节位置 [num_envs, num_joints]"""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """机器人当前关节速度 [num_envs, num_joints]"""
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """机器人当前身体部位位置（世界系）[num_envs, num_bodies, 3]"""
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """机器人当前身体部位方向（世界系）[num_envs, num_bodies, 4]"""
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """机器人当前身体部位线速度（世界系）[num_envs, num_bodies, 3]"""
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """机器人当前身体部位角速度（世界系）[num_envs, num_bodies, 3]"""
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    # ===== 机器人当前锚点状态 =====
    
    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """机器人当前锚点位置（世界系）[num_envs, 3] - 参考锚点（现在应该在的位置）"""
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """机器人当前锚点方向（世界系）[num_envs, 4]"""
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """机器人当前锚点线速度（世界系）[num_envs, 3]"""
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """机器人当前锚点角速度（世界系）[num_envs, 3]"""
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        """
        更新追踪误差指标
        
        计算当前状态与参考状态的差异，用于日志记录和分析。
        包括：锚点误差、身体部位误差、关节误差
        """
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """
        自适应采样算法
        
        根据历史失败率动态调整采样概率，优先采样困难的时间段。
        这是一种课程学习策略，可以加速训练。
        
        参数:
            env_ids: 需要重新采样的环境ID列表
        
        算法流程:
        1. 统计当前失败的时间段（bin）
        2. 更新失败率统计
        3. 计算采样概率（失败率高的 bin 概率大）
        4. 平滑采样概率（使用卷积核）
        5. 采样新的时间步
        6. 更新采样统计指标
        """
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        """
        重新采样运动指令并初始化机器人状态
        
        当 episode 结束或运动序列播放完毕时调用。
        
        参数:
            env_ids: 需要重置的环境ID列表
        
        流程:
        1. 使用自适应采样选择新的时间步
        2. 获取该时间步的参考状态
        3. 添加随机扰动（数据增强）
        4. 将机器人重置到新状态
        """
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """
        更新运动指令（每个仿真步调用）
        
        流程:
        1. 时间步前进
        2. 检查是否到达序列末尾，重新采样
        3. 计算相对身体姿态（用于观测）
        4. 更新失败率统计
        """
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        设置调试可视化
        
        参数:
            debug_vis: True = 开启可视化，False = 关闭
        
        可视化内容:
        - 当前锚点（绿色）vs 目标锚点（红色）
        - 当前身体部位（绿色）vs 目标身体部位（红色）
        """
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        """调试可视化回调函数（每帧调用）更新可视化标记的位置和方向"""
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """
    运动指令配置类
    
    定义了 MotionCommand 的所有可配置参数。
    """

    class_type: type = MotionCommand

    asset_name: str = MISSING
    """机器人资产名称（在场景中的名称）"""

    motion_file: str = MISSING
    """运动数据文件路径（.npz格式）"""
    
    anchor_body_name: str = MISSING
    """锚点身体部位名称（例如："pelvis", "torso"）"""
    
    body_names: list[str] = MISSING
    """要追踪的身体部位名称列表"""

    pose_range: dict[str, tuple[float, float]] = {}
    """初始姿态随机化范围 {"x": (min, max), "yaw": (min, max), ...}"""
    
    velocity_range: dict[str, tuple[float, float]] = {}
    """初始速度随机化范围"""

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """关节位置随机扰动范围（弧度）"""

    adaptive_kernel_size: int = 1
    """平滑核大小（用于平滑失败率分布）"""
    
    adaptive_lambda: float = 0.8
    """平滑核的衰减系数（λ^i）"""
    
    adaptive_uniform_ratio: float = 0.1
    """均匀采样比例（防止某些时间段永远不被采样）"""
    
    adaptive_alpha: float = 0.001
    """失败率更新的学习率（指数移动平均系数）"""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
