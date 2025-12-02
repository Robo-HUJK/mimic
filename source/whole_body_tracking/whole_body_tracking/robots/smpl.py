import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

##
# Configuration
##

# SMPL 人形机器人的关节配置
# 该配置定义了机器人的 USD 路径、初始状态、物理属性以及执行器设置
SMPL_HUMANOID = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot", # 机器人在场景中的 Prim 路径，支持正则表达式
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/smpl/smpl_humanoid.usda", # USD 文件的路径
        activate_contact_sensors=True, # 激活接触传感器
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None, # 是否禁用重力 (None 表示使用默认设置)
            max_depenetration_velocity=10.0, # 最大穿透恢复速度
            enable_gyroscopic_forces=True, # 启用陀螺仪力
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, # 启用自碰撞检测
            solver_position_iteration_count=4, # 位置求解器迭代次数
            solver_velocity_iteration_count=0, # 速度求解器迭代次数
            sleep_threshold=0.005, # 休眠阈值
            stabilization_threshold=0.001, # 稳定阈值
        ),
        copy_from_source=False, # 是否从源复制 USD 文件
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95), # 初始位置 (x, y, z)
        joint_pos={".*": 0.0}, # 初始关节位置 (所有关节设为 0)
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"], # 匹配所有关节
            stiffness=None, # 刚度 (将在其他地方设置或从 USD 读取)
            damping=None, # 阻尼 (将在其他地方设置或从 USD 读取)
        ),
    },
)
