import numpy as np

# 你的文件路径
file_path = "/home/wyb/humanoid/video_CR7_level1_filter_amass.npz"

try:
    data = np.load(file_path)
    print(f"文件: {file_path}")
    print("包含的键 (Keys):", list(data.keys()))
    
    if 'fps' in data:
        print(f"FPS: {data['fps']}")
    else:
        print("❌ 错误: 缺少 'fps' 键！")
        
except Exception as e:
    print(f"无法加载文件: {e}")