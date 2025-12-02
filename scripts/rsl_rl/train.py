# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 训练强化学习智能体的脚本"""

"""首先启动 Isaac Sim 模拟器"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip

# 添加命令行参数
parser = argparse.ArgumentParser(description="使用 RSL-RL 训练强化学习智能体。")
parser.add_argument("--video", action="store_true", default=False, help="训练期间录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（步数）。")
parser.add_argument("--video_interval", type=int, default=2000, help="视频录制之间的间隔（步数）。")
parser.add_argument("--num_envs", type=int, default=None, help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument("--seed", type=int, default=None, help="环境使用的随机种子")
parser.add_argument("--max_iterations", type=int, default=None, help="强化学习策略训练迭代次数。")

# [修改 1] 将 registry_name 设为可选 (default=None)，并添加 motion_file 参数
parser.add_argument("--registry_name", type=str, default=None, help="wandb 注册表的名称。")
parser.add_argument("--motion_file", type=str, default=None, help="本地 .npz 动作文件的绝对路径。")

# 追加 RSL-RL 命令行参数
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 如果录制视频，始终启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 为 Hydra 清除 sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# 启动 omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下是其余所有内容"""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入扩展以设置环境任务
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

# PyTorch 性能优化设置
torch.backends.cuda.matmul.allow_tf32 = True  # 允许 CUDA 矩阵乘法使用 TF32
torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32
torch.backends.cudnn.deterministic = False  # 不强制确定性（为了性能）
torch.backends.cudnn.benchmark = False  # 不使用 cuDNN benchmark 模式


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """使用 RSL-RL 智能体进行训练。"""
    # 使用非 Hydra CLI 参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境随机种子
    # 注意：某些随机化发生在环境初始化中，所以我们在这里设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # [修改 2] 动作文件加载逻辑：优先使用命令行参数，其次是 wandb，最后是配置文件默认值
    registry_name = "local"  # 默认名称，用于 runner 初始化

    if args_cli.motion_file:
        # 情况 A: 命令行指定了本地文件
        print(f"[INFO] 使用命令行指定的本地动作文件: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        
    elif args_cli.registry_name:
        # 情况 B: 指定了 wandb registry，从云端下载
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib
        import wandb

        print(f"[INFO] 正在从 WandB 下载动作文件: {registry_name}")
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        
    else:
        # 情况 C: 什么都没指定，使用配置文件(flat_env_cfg.py)中的路径
        print(f"[INFO] 未指定动作文件，使用配置文件中的默认路径: {env_cfg.commands.motion.motion_file}")

    # 指定日志实验的目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 在目录中记录实验: {log_root_path}")
    # 指定日志运行的目录: {时间戳}_{运行名称}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 创建 isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 包装以进行视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 训练期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 如果强化学习算法需要，转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 为 rsl-rl 包装环境
    env = RslRlVecEnvWrapper(env)

    # 从 rsl-rl 创建运行器
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # 将 git 状态写入日志
    runner.add_git_repo_to_log(__file__)
    # 在创建新的 log_dir 之前保存恢复路径
    if agent_cfg.resume:
        # 获取之前检查点的路径
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: 从以下位置加载模型检查点: {resume_path}")
        # 加载之前训练的模型
        runner.load(resume_path)

    # 将配置转储到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 运行训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
