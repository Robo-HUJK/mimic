from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """
    G1 机器人在平坦地面上的动作追踪环境配置。
    继承自 TrackingEnvCfg，针对 G1 机器人进行了特定的参数调整。
    """
    def __post_init__(self):
        super().__post_init__()

        # 配置场景中的机器人资产
        # 使用 G1_CYLINDER_CFG (通常是用圆柱体简化的碰撞模型，计算更快)
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 设置动作缩放比例
        # 神经网络输出的动作通常在 [-1, 1] 之间，需要乘以这个比例转换成实际的关节角度
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
        # 设置动作追踪命令的锚点身体部位
        # "torso_link" 通常作为躯干中心，用于计算相对位置
        self.commands.motion.anchor_body_name = "torso_link"
        
        # 设置需要追踪的关键身体部位列表
        # 奖励函数会计算这些部位与参考动作之间的误差
        self.commands.motion.body_names = [
            "pelvis",                 # 骨盆
            "left_hip_roll_link",     # 左髋滚转
            "left_knee_link",         # 左膝
            "left_ankle_roll_link",   # 左踝滚转
            "right_hip_roll_link",    # 右髋滚转
            "right_knee_link",        # 右膝
            "right_ankle_roll_link",  # 右踝滚转
            "torso_link",             # 躯干
            "left_shoulder_roll_link",# 左肩滚转
            "left_elbow_link",        # 左肘
            "left_wrist_yaw_link",    # 左腕偏航
            "right_shoulder_roll_link",# 右肩滚转
            "right_elbow_link",       # 右肘
            "right_wrist_yaw_link",   # 右腕偏航
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    """
    不带状态估计 (Without State Estimation) 的 G1 平坦地面环境配置。
    
    这个配置模拟了真实世界中可能无法获取精确状态的情况。
    它移除了某些依赖完美状态估计的观测项。
    """
    def __post_init__(self):
        super().__post_init__()
        # 移除锚点相对于基座的位置观测
        self.observations.policy.motion_anchor_pos_b = None
        # 移除基座线速度的观测 (模拟没有速度计的情况)
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    """
    低控制频率的 G1 平坦地面环境配置。
    
    用于训练在较低控制频率下运行的策略。
    """
    def __post_init__(self):
        super().__post_init__()
        # 增加抽取因子 (decimation)，降低控制频率
        # 例如：如果物理引擎是 200Hz，decimation=4，则控制频率为 50Hz
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        
        # 调整动作变化率 (action rate) 的惩罚权重
        # 因为频率变低了，动作变化的幅度可能会变大，所以需要调整权重以保持平衡
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
