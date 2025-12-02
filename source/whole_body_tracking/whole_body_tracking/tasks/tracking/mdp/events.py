"""
事件/随机化模块

该模块定义了训练过程中的随机化事件函数。
随机化用于域随机化（Domain Randomization），增强策略的鲁棒性。

主要功能：
1. 关节默认位置随机化（模拟标定误差）
2. 刚体质心随机化（模拟质量分布不确定性）

应用场景：
- Sim-to-Real迁移：训练时引入随机性，提高真实环境适应性
- 鲁棒性训练：应对不同的机器人个体差异和磨损

调用时机：
- 通常在 reset 事件中调用
- 可以在每个 episode 开始时随机化
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化关节默认位置
    
    模拟机器人的标定误差。真实机器人的零位可能与URDF文件定义的不完全一致，
    通过随机化默认关节位置，可以提高策略对这种差异的鲁棒性。
    
    参数:
        env: 强化学习环境实例
        env_ids: 要随机化的环境ID列表
                None 表示所有环境
        asset_cfg: 资产配置
                  - name: 机器人名称
                  - joint_ids: 要随机化的关节ID列表
        pos_distribution_params: 位置分布参数 (min, max)
                                例如：(-0.1, 0.1) 表示在 ±0.1 弧度范围内随机
        operation: 随机化操作类型
                  - "add": 在原值基础上加上随机值
                  - "scale": 将原值乘以随机值
                  - "abs": 直接使用随机值（绝对值）
        distribution: 随机分布类型
                     - "uniform": 均匀分布
                     - "log_uniform": 对数均匀分布
                     - "gaussian": 高斯分布
    
    功能流程:
        1. 保存标称值（用于后续导出或参考）
        2. 生成随机偏移
        3. 更新 default_joint_pos（关节默认位置）
        4. 同步更新动作管理器的偏移量（因为动作是相对于默认位置的）
    
    应用场景:
        - 模拟不同机器人个体之间的差异
        - 模拟关节磨损导致的零位漂移
        - 提高策略的Sim-to-Real迁移能力
    
    示例:
        # 在配置文件中使用
        reset_joint_default = EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pos_distribution_params": (-0.05, 0.05),  # ±0.05弧度
                "operation": "add",
            }
        )
    """
    # 获取机器人资产（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]

    # 保存标称值（第一个环境的默认值）用于导出
    # 这样可以知道原始的默认位置是什么
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # 解析环境ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 解析关节索引
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # 所有关节（优化性能）
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # 如果提供了分布参数，则进行随机化
    if pos_distribution_params is not None:
        # 克隆当前默认关节位置
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        
        # 根据指定的操作和分布生成随机值
        # _randomize_prop_by_op 是 IsaacLab 提供的工具函数
        pos = _randomize_prop_by_op(
            pos, 
            pos_distribution_params, 
            env_ids, 
            joint_ids, 
            operation=operation, 
            distribution=distribution
        )[env_ids][:, joint_ids]

        # 处理索引（用于高级索引）
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        
        # 更新默认关节位置
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        
        # 重要：同步更新动作管理器的偏移量
        # 因为动作通常定义为相对于默认位置的增量
        # action_target = default_pos + action_offset
        # 如果不更新 offset，会导致动作基准点错误
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """
    随机化刚体质心（Center of Mass）
    
    通过在给定范围内添加随机偏移来改变刚体的质心位置。
    这可以模拟负载变化、质量分布不确定性等情况。
    
    .. note::
        此函数使用 CPU 张量进行质心赋值。
        建议仅在环境初始化期间使用此函数，避免运行时性能开销。
    
    参数:
        env: 强化学习环境实例
        env_ids: 要随机化的环境ID列表
                None 表示所有环境
        com_range: 质心偏移范围（米）
                  格式: {"x": (min, max), "y": (min, max), "z": (min, max)}
                  例如: {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
        asset_cfg: 资产配置
                  - name: 机器人名称
                  - body_ids: 要随机化的刚体ID列表
                             None 或 slice(None) 表示所有刚体
    
    功能流程:
        1. 获取当前所有刚体的质心位置
        2. 为指定刚体生成随机偏移
        3. 将偏移加到原质心位置上
        4. 设置新的质心位置到物理引擎
    
    物理意义:
        - 质心位置影响动力学行为（平衡、惯性、力矩等）
        - 随机化质心可以模拟：
          * 负载变化（背包、工具等）
          * 制造误差（质量分布不均）
          * 部件磨损或更换
    
    应用场景:
        - 提高对负载变化的鲁棒性
        - 模拟不同配置的机器人（带/不带传感器、工具等）
        - Sim-to-Real迁移（真实机器人的质心可能与CAD模型不同）
    
    注意事项:
        - 此函数操作在CPU上执行（为了与PhysX接口兼容）
        - 过大的质心偏移可能导致不稳定的动力学
        - 建议偏移范围在 ±0.05米 以内
    
    示例:
        # 在配置文件中使用
        reset_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_ids=[1, 2, 3]),  # 只随机化特定部位
                "com_range": {
                    "x": (-0.02, 0.02),  # X方向 ±2cm
                    "y": (-0.02, 0.02),  # Y方向 ±2cm
                    "z": (-0.01, 0.01),  # Z方向 ±1cm（通常Z方向影响更大）
                },
            }
        )
    """
    # 获取机器人资产（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 解析环境ID（在CPU上操作）
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析刚体索引
    if asset_cfg.body_ids == slice(None):
        # 所有刚体
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 生成随机质心偏移
    # 从 com_range 字典中提取 x, y, z 的范围
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    
    # 采样随机值：[num_envs, 3]
    # unsqueeze(1) 变为 [num_envs, 1, 3]，用于广播到多个刚体
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],  # 下界
        ranges[:, 1],  # 上界
        (len(env_ids), 3),  # 形状：[num_envs, 3]
        device="cpu"
    ).unsqueeze(1)

    # 获取当前所有刚体的质心位置
    # coms 形状: [num_envs, num_bodies, 3]
    # [:3] 只取位置部分（可能还包含其他信息）
    coms = asset.root_physx_view.get_coms().clone()

    # 为指定的刚体添加随机偏移
    # body_ids 选择要修改的刚体
    # rand_samples 广播到每个选定的刚体
    coms[:, body_ids, :3] += rand_samples

    # 将新的质心位置设置到物理引擎
    # 只影响 env_ids 指定的环境
    asset.root_physx_view.set_coms(coms, env_ids)
