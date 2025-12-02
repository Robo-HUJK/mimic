from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """
    带有延迟命令应用的理想 PD 执行器。

    此类扩展了 :class:`IdealPDActuator` 类，通过向执行器命令添加延迟来模拟真实的通信或处理延迟。
    延迟是使用循环缓冲区实现的，该缓冲区存储一定数量物理步骤的执行器命令。
    每个物理步骤都会将最新的驱动值推入缓冲区，但应用到仿真的最终驱动值会滞后一定数量的物理步骤。

    时间滞后量是可配置的，并且可以在每次重置时设置为最小和最大时间滞后界限之间的随机值。
    最小和最大时间滞后值在传递给类的配置实例中设置。
    """

    cfg: DelayedImplicitActuatorCfg
    """执行器模型的配置。"""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # 实例化延迟缓冲区
        # 用于存储位置、速度和力矩命令的历史记录
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # 所有环境的索引
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # 获取环境数量 (因为 env_ids 可能是一个切片)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # 为 env_ids 中的环境设置一个新的随机延迟
        # 延迟步数在 [min_delay, max_delay] 之间随机采样
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # 设置延迟
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # 重置缓冲区内容
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # 基于模型设置的延迟，对所有设定点应用延迟
        # 从缓冲区中获取延迟后的命令
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # 计算执行器模型 (调用父类的 compute 方法进行实际的 PD 计算)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """延迟 PD 执行器的配置。"""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """执行器命令可能延迟的最小物理时间步数。默认为 0。"""

    max_delay: int = 0
    """执行器命令可能延迟的最大物理时间步数。默认为 0。"""
