import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    """
    自定义的 OnPolicyRunner，用于在保存模型时自动导出 ONNX 并上传到 WandB。
    """
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        # 如果使用 wandb 记录日志，则执行额外的导出操作
        if self.logger_type in ["wandb"]:
            # 获取保存路径的基础目录
            policy_path = path.split("model")[0]
            # 生成 ONNX 文件名，通常基于运行名称或路径
            filename = policy_path.split("/")[-2] + ".onnx"
            # 导出策略网络为 ONNX 格式
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            # 为 ONNX 模型添加元数据（如环境配置等）
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            # 将 ONNX 文件保存到 WandB 云端
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    """
    专门用于动作追踪任务的 Runner，支持导出特定格式的 Motion Policy ONNX。
    """
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        # 用于 WandB Artifact 的注册名称
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # 使用专门的 export_motion_policy_as_onnx 函数导出
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # 如果指定了 registry_name，将此运行链接到 WandB Artifact Registry
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
