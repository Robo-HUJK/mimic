from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境在策略更新前收集的时间步数（越大数据越多但更新越慢）###修改24
    num_steps_per_env = 24
    
    # 最大训练迭代次数（总训练轮数，控制训练时长）###修改30000
    max_iterations = 3000
    
    # 模型检查点保存间隔（每50次迭代保存一次模型）###修改500
    save_interval = 500
    
    # 实验名称（用于日志目录命名）
    experiment_name = "g1_flat"
    
    # 是否使用经验归一化（True表示使用运行统计来归一化观测值）
    empirical_normalization = True
    
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        # 初始动作噪声标准差（用于探索，训练初期动作随机性较大）
        init_noise_std=1.0,
        
        # Actor网络隐藏层维度（策略网络：观测->动作，3层神经网络）
        actor_hidden_dims=[512, 256, 128],
        
        # Critic网络隐藏层维度（价值网络：观测->价值估计，3层神经网络）
        critic_hidden_dims=[512, 256, 128],
        
        # 激活函数类型（ELU: Exponential Linear Unit）
        activation="elu",
    )
    
    # PPO算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值函数损失的权重系数（平衡策略损失和价值损失）
        value_loss_coef=1.0,
        
        # 是否对价值函数损失使用裁剪（防止价值函数更新过大）
        use_clipped_value_loss=True,
        
        # PPO裁剪参数ε（限制策略更新幅度在[1-0.2, 1+0.2]范围内，防止更新过大）
        clip_param=0.2,
        
        # 熵正则化系数（鼓励探索，防止策略过早收敛到次优解）
        entropy_coef=0.005,
        
        # 每次数据收集后的训练轮数（用同一批数据训练网络的次数）###修改5
        num_learning_epochs=5,
        
        # 小批次数量（将收集的数据分成4个批次进行训练，提高训练稳定性）###修改4
        num_mini_batches=4,
        
        # 学习率（神经网络参数更新步长）
        learning_rate=1.0e-3,
        
        # 学习率调度策略（adaptive表示根据KL散度自适应调整学习率）
        schedule="adaptive",
        
        # 折扣因子（决定未来奖励的重要性，0.99表示非常重视长期回报）
        gamma=0.99,
        
        # GAE的λ参数（Generalized Advantage Estimation，权衡偏差和方差）
        lam=0.95,
        
        # 期望的KL散度目标（用于自适应学习率调整，控制策略更新幅度）
        desired_kl=0.01,
        
        # 梯度裁剪的最大范数（防止梯度爆炸，提高训练稳定性）
        max_grad_norm=1.0,
    )

LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
