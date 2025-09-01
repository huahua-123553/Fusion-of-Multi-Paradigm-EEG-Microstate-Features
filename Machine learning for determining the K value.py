# -*- coding: utf-8 -*-
"""
K值扫描 LOSO-CV + mRMR + SVM
自动识别病人标签，计算平均值和方差，并画图
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mrmr import mrmr_classif
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ========== 1️⃣ 设置工作目录 & 读取数据 ==========
os.chdir(r'C:\Users\Administrator\Desktop\MCI机器学习\静息态')

data_hc = loadmat("HC.mat")['data']   # 健康组
data_pod = loadmat("MCI.mat")['data'] # 病人组

labels_hc = np.zeros(data_hc.shape[0], dtype=int)
labels_pod = np.ones(data_pod.shape[0], dtype=int)

ids_hc = data_hc[:, 0]
ids_pod = data_pod[:, 0]

features = np.vstack((data_hc[:, 1:], data_pod[:, 1:]))
labels = np.concatenate((labels_hc, labels_pod))
groups = np.concatenate((ids_hc, ids_pod))

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_features = pd.DataFrame(features_scaled)

# ========== 2️⃣ K值扫描 ==========
K_values = range(1, 14)
results_list = []

for K in K_values:
    selected_features = mrmr_classif(X=df_features, y=labels, K=K)
    selected_data = df_features.iloc[:, selected_features].values
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    acc_list, sens_list, spec_list = [], [], []
    
    for train_idx, test_idx in sgkf.split(selected_data, labels, groups=groups):
        X_train, X_test = selected_data[train_idx], selected_data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # SVM 调参
        svm = SVC(kernel='rbf')
        param_grid = {'C': np.linspace(0.1, 10, 10), 'gamma': np.linspace(0.01, 1, 10)}
        clf = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)

        best_svm = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
        best_svm.fit(X_train, y_train)
        predictions = best_svm.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0,1]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        acc_list.append(accuracy)
        sens_list.append(sensitivity)
        spec_list.append(specificity)
    
    results_list.append({
        'K': K,
        'Acc_mean': np.mean(acc_list),
        'Acc_std': np.std(acc_list, ddof=1),
        'Sens_mean': np.mean(sens_list),
        'Sens_std': np.std(sens_list, ddof=1),
        'Spec_mean': np.mean(spec_list),
        'Spec_std': np.std(spec_list, ddof=1)
    })
    print(f"K={K} 完成")

# ========== 3️⃣ 保存结果 ==========
results_df = pd.DataFrame(results_list)
results_df.to_csv("K值扫描结果.csv", index=False)
print("✅ K值扫描结果已保存到 K值扫描结果.csv")

# ========== 4️⃣ 作图 ==========
plt.figure(figsize=(10,6))
plt.errorbar(results_df['K'], results_df['Acc_mean'], yerr=results_df['Acc_std'], label='Accuracy', fmt='-o')
plt.errorbar(results_df['K'], results_df['Sens_mean'], yerr=results_df['Sens_std'], label='Sensitivity', fmt='-s')
plt.errorbar(results_df['K'], results_df['Spec_mean'], yerr=results_df['Spec_std'], label='Specificity', fmt='-^')
plt.xlabel("mRMR K")
plt.ylabel("Mean and standard deviation")
plt.title(" Mean and standard deviation of accuracy, sensitivity, and specificity for each K value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
