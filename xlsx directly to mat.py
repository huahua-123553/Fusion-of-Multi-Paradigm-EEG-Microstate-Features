import numpy as np
import pandas as pd
from scipy.io import savemat
import os

# 读取 Excel 文件
xlsx_file = r'C:\Users\Administrator\Desktop\MCI机器学习\静息态\MCI.xlsx'  # 你的MCI文件路径
df = pd.read_excel(xlsx_file)

# 确认数据：第一列是被试编号，后面都是特征
ids = df.iloc[:, 0].values.reshape(-1, 1)   # 被试编号
features = df.iloc[:, 1:].values           # 特征

# 拼接 ID 和特征，形成最终矩阵
data_array = np.hstack((ids, features))

# 保存为 .mat 文件
save_path = r'C:\Users\Administrator\Desktop\MCI机器学习\静息态\MCI.mat'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
savemat(save_path, {'data': data_array})

print(f"✅ MCI.mat 已保存，data.shape = {data_array.shape}")
