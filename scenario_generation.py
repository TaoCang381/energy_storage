import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def generate_scenarios(num_samples, num_scenarios, mean, cov_matrix):
    """
    使用拉丁超立方抽样(LHS)和K-means聚类生成并削减随机场景。

    参数:
    - num_samples (int): 初始生成的样本数量。
    - num_scenarios (int): 希望削减到的最终场景数量。
    - mean (np.array): 预测误差的均值向量 (通常为0)。
    - cov_matrix (np.array): 预测误差的协方差矩阵。

    返回:
    - scenarios (np.array): 削减后的典型场景 (shape: [num_scenarios, num_variables])。
    - probabilities (np.array): 每个典型场景对应的概率。
    """
    # 检查输入维度
    num_variables = len(mean)
    if cov_matrix.shape != (num_variables, num_variables):
        raise ValueError("均值向量和协方差矩阵的维度不匹配。")

    # 1. 使用拉丁超立方抽样生成均匀分布的样本 [0, 1]
    sampler = LatinHypercube(d=num_variables, seed=42)
    uniform_samples = sampler.random(n=num_samples)

    # 2. 将均匀分布的样本转换为正态分布样本
    # 使用逆累计分布函数 (ppf)
    # 假设变量之间是独立的，如果不是，需要使用Cholesky分解
    # L = np.linalg.cholesky(cov_matrix)
    # norm_samples = mean + uniform_samples @ L.T # Cholesky分解方法

    # 这里我们使用更通用的方法，直接从多元正态分布中生成
    # 但为了应用LHS，我们先独立转换，然后引入相关性
    # 这种方法更直观地体现了LHS的优势
    norm_dist = norm(loc=0, scale=1)  # 标准正态分布
    standard_norm_samples = norm_dist.ppf(uniform_samples)

    # 引入相关性
    L = np.linalg.cholesky(cov_matrix)
    correlated_samples = standard_norm_samples @ L.T + mean

    # 3. 使用K-means算法进行场景削减
    kmeans = KMeans(n_clusters=num_scenarios, random_state=42, n_init='auto')
    kmeans.fit(correlated_samples)

    # 获取削减后的场景（聚类中心）
    scenarios = kmeans.cluster_centers_

    # 计算每个场景的概率
    labels = kmeans.labels_
    probabilities = np.array([np.sum(labels == i) for i in range(num_scenarios)]) / num_samples

    return scenarios, probabilities


def plot_scenarios(initial_samples, reduced_scenarios, probabilities):
    """
    可视化初始样本和削减后的场景（仅适用于2D情况）。
    """
    if initial_samples.shape[1] != 2 or reduced_scenarios.shape[1] != 2:
        print("警告: 只能对二维数据进行可视化。")
        return

    plt.figure(figsize=(10, 8))
    # 绘制初始样本点
    plt.scatter(initial_samples[:, 0], initial_samples[:, 1], c='lightblue', alpha=0.5,
                label=f'{initial_samples.shape[0]} Initial Samples')
    # 绘制削减后的场景点，点的大小与概率成正比
    plt.scatter(reduced_scenarios[:, 0], reduced_scenarios[:, 1], s=probabilities * 5000, c='red', edgecolor='black',
                label=f'{reduced_scenarios.shape[0]} Reduced Scenarios')

    plt.title('Scenario Generation and Reduction using LHS and K-means')
    plt.xlabel('Uncertain Variable 1 (e.g., Wind Power Error)')
    plt.ylabel('Uncertain Variable 2 (e.g., PV Power Error)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    # --- 示例：如何使用 ---
    # 假设我们有3个不确定性变量：风电、光伏、电负荷的预测误差
    # 它们的预测误差均值为0
    num_uncertain_variables = 3
    mean_error = np.zeros(num_uncertain_variables)

    # 假设它们的标准差分别为预测值的15%, 20%, 5%
    # 并且风和光伏有一定的负相关性
    # 注意：这是一个示例协方差矩阵，你需要根据你的实际数据来确定
    cov = np.array([
        [0.15 ** 2, -0.01, 0.005],  # Var(Wind), Cov(Wind, PV), Cov(Wind, Load)
        [-0.01, 0.20 ** 2, 0.002],  # Cov(PV, Wind), Var(PV), Cov(PV, Load)
        [0.005, 0.002, 0.05 ** 2]  # Cov(Load, Wind), Cov(Load, PV), Var(Load)
    ])

    # 设置生成参数
    N_initial_samples = 1000  # 初始生成1000个样本
    S_reduced_scenarios = 10  # 削减为10个典型场景

    # 生成场景
    reduced_scenarios, scenario_probabilities = generate_scenarios(
        num_samples=N_initial_samples,
        num_scenarios=S_reduced_scenarios,
        mean=mean_error,
        cov_matrix=cov
    )

    # 打印结果
    print(f"--- 削减后的 {S_reduced_scenarios} 个典型场景 ---")
    print("场景 (行) vs. 不确定性变量 (列):")
    np.set_printoptions(precision=4, suppress=True)
    print(reduced_scenarios)

    print("\n--- 每个场景对应的概率 ---")
    print(scenario_probabilities)

    print(f"\n概率之和: {np.sum(scenario_probabilities):.2f}")

    # 对于二维情况的可视化示例
    if num_uncertain_variables == 2:
        # 重新生成初始样本用于绘图
        sampler_2d = LatinHypercube(d=2, seed=42)
        uniform_samples_2d = sampler_2d.random(n=N_initial_samples)
        L_2d = np.linalg.cholesky(cov[:2, :2])
        initial_samples_2d = norm.ppf(uniform_samples_2d) @ L_2d.T

        plot_scenarios(initial_samples_2d, reduced_scenarios, scenario_probabilities)