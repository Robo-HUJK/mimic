# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """
    导出动作策略为 ONNX 格式。
    
    此函数封装了 _OnnxMotionPolicyExporter 的使用，将策略网络和相关的运动数据导出为 ONNX 模型。
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    """
    自定义的 ONNX 导出器，用于导出包含运动参考数据的策略。
    
    除了导出策略网络（Actor），它还将整个运动轨迹数据（关节位置、速度、身体位置等）
    嵌入到模型中，使得推理时可以根据时间步（time_step）获取对应的参考动作。
    """
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        # 获取运动命令管理器，读取预加载的动作数据
        cmd: MotionCommand = env.command_manager.get_term("motion")

        # 将所有运动数据移动到 CPU，以便嵌入到 ONNX 模型中
        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        """
        定义 ONNX 模型的前向传播逻辑。
        
        输入:
            x: 观测值 (Observation)
            time_step: 当前动作的时间步索引
            
        输出:
            actions: 策略网络输出的动作
            ...: 对应时间步的参考运动数据（关节位置、速度等）
        """
        # 限制时间步索引，防止越界
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)), # 计算动作
            self.joint_pos[time_step_clamped], # 获取参考关节位置
            self.joint_vel[time_step_clamped], # 获取参考关节速度
            self.body_pos_w[time_step_clamped], # 获取参考身体位置
            self.body_quat_w[time_step_clamped], # 获取参考身体姿态
            self.body_lin_vel_w[time_step_clamped], # 获取参考线速度
            self.body_ang_vel_w[time_step_clamped], # 获取参考角速度
        )

    def export(self, path, filename):
        """执行实际的 ONNX 导出操作。"""
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"], # 定义输入节点名称
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ], # 定义输出节点名称
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    """辅助函数：将列表转换为 CSV 格式的字符串，用于元数据存储。"""
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    """
    向导出的 ONNX 模型添加元数据。
    
    这些元数据包含了环境配置、机器人参数（如关节名称、刚度、阻尼）、
    观测和动作的配置等。这对于在推理环境中正确设置机器人至关重要。
    """
    onnx_path = os.path.join(path, filename)

    # 获取观测项名称
    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    # 获取观测历史长度配置
    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    # 构建元数据字典
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names, # 关节名称
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(), # 关节刚度 (Kp)
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(), # 关节阻尼 (Kd)
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(), # 默认关节位置
        "command_names": env.command_manager.active_terms, # 命令项名称
        "observation_names": observation_names, # 观测项名称
        "observation_history_lengths": observation_history_lengths, # 观测历史长度
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(), # 动作缩放比例
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name, # 锚点身体部位
        "body_names": env.command_manager.get_term("motion").cfg.body_names, # 追踪的身体部位列表
    }

    model = onnx.load(onnx_path)

    # 将元数据写入 ONNX 模型属性
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
