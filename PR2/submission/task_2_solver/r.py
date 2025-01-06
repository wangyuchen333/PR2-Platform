import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前文件的目录
dir_path = os.path.dirname(os.path.realpath(__file__))
cmd_file = os.path.join(dir_path, 'task2_cmd.npz')

# 加载 .npz 文件
cmds_npz = np.load(cmd_file)

# 访问 'cmd' 数组
cmds = cmds_npz['cmd']

# 检查数据有效性
if np.isnan(cmds).any() or np.isinf(cmds).any():
    raise ValueError("数据包含 NaN 或无限值，请检查数据源。")

# 设置统一的样式
plt.style.use('seaborn-darkgrid')

# 绘制每个维度的折线图
fig, axs = plt.subplots(4, 1, figsize=(12, 10))

for i in range(4):
    axs[i].plot(cmds[:, i], label=f'维度 {i+1}', color=f'C{i}', linestyle='-', marker='o')
    axs[i].set_title(f'命令维度 {i+1} 的变化')
    axs[i].set_xlabel('索引')
    axs[i].set_ylabel('值')
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.show()

# 绘制每个维度的直方图
fig, axs = plt.subplots(4, 1, figsize=(12, 10))

for i in range(4):
    axs[i].hist(cmds[:, i], bins=30, color=f'C{i}', alpha=0.7, edgecolor='black')
    axs[i].set_title(f'命令维度 {i+1} 的分布')
    axs[i].set_xlabel('值')
    axs[i].set_ylabel('频率')
    axs[i].grid(True)

plt.tight_layout()
plt.show()
