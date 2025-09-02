# file: main_simulation_hierarchical.py (V2.0 - WPT-Powered Functional Decoupling)
# 备注：引入小波包变换(WPT)对净负荷进行功能解耦，为分层MPC提供清晰的任务信号。

import numpy as np
import matplotlib.pyplot as plt
import pywt  # --- 修改区域 ---: 导入小波包变换库
import pandas as pd  # 导入pandas用于数据处理

from hess_system import HybridEnergyStorageSystem
from mpc_ems_hierarchical import HierarchicalMPCEms

# (导入储能模型部分与之前一致，此处省略)
from high_power_density_group.flywheel_simulation import FlywheelModel
from high_power_density_group.supercapacitor_simulation import Supercapacitor
from high_power_density_group.Superconducting_magnetic_energy_storage_simulation import \
    SuperconductingMagneticEnergyStorage
from Medium_power_density_group.electrochemical_energy_storage import ElectrochemicalEnergyStorage
from low_power_density_group.pumped_storage_simulation import PumpedHydroStorage
from low_power_density_group.hydrogen_storage import HydrogenStorage
from low_power_density_group.thermal_storage import ThermalEnergyStorage
from low_power_density_group.caes_system import DiabaticCAES

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 修改区域 START: 增加小波包分解函数 ---
def decompose_power_signal(power_signal, wavelet='db4', level=3):
    """
    使用小波包变换将功率信号分解为不同频段的子信号。

    参数:
    power_signal (np.array): 输入的净负荷功率序列。
    wavelet (str): 使用的小波基函数，'db4'是一个常用的好选择。
    level (int): 分解的层数。3层会得到 2^3 = 8 个频带。

    返回:
    dict: 一个字典，包含了重构后的低频、中频、高频功率序列。
    """
    # 1. 执行小波包分解
    wp = pywt.WaveletPacket(data=power_signal, wavelet=wavelet, mode='symmetric', maxlevel=level)

    # 2. 获取分解后各个频带的节点
    #    对于3层分解，我们会得到 'aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd' 8个节点
    #    'a'代表低频(approximate), 'd'代表高频(detail)。从左到右代表从粗到细的分解。
    #    例如'aaa'是最低频部分，'ddd'是最高频部分。
    nodes = wp.get_level(level, order='freq')  # 按频率排序节点

    # 3. 重构不同频带的信号
    #    我们可以根据需要，将这8个频带组合成我们想要的“高、中、低”三部分

    # 定义频带分配 (这是一个关键的设计点，可以根据您的储能特性调整)
    # 假设8个频带 (从0到7)，我们这样分配：
    low_freq_bands = nodes[:2]  # 最低的2个频带 (0, 1) -> 慢速储能
    mid_freq_bands = nodes[2:4]  # 中间的2个频带 (2, 3) -> 电化学储能
    high_freq_bands = nodes[4:]  # 最高的4个频带 (4, 5, 6, 7) -> 快速响应组

    # 初始化空的信号数组
    low_freq_signal = np.zeros_like(power_signal)
    mid_freq_signal = np.zeros_like(power_signal)
    high_freq_signal = np.zeros_like(power_signal)

    # 将属于各个部分的频带信号相加重构
    for band in low_freq_bands:
        low_freq_signal += wp[band.path].reconstruct(update=False)

    for band in mid_freq_bands:
        mid_freq_signal += wp[band.path].reconstruct(update=False)

    for band in high_freq_bands:
        high_freq_signal += wp[band.path].reconstruct(update=False)

    # 确保重构信号的长度与原始信号一致
    if len(low_freq_signal) != len(power_signal):
        low_freq_signal = pywt.waverec(wp.get_level(level, order='freq'), wavelet, 'symmetric')[:len(power_signal)]
        # This is a simplified reconstruction. For a more accurate one, you'd reconstruct each band and sum them up as done above, handling potential length mismatches.
        # The above logic for summing up bands is generally better. If length issues persist, it's often due to padding in the dwt algorithm.
        # A robust way is to trim the reconstructed signal to the original length.
        low_freq_signal = low_freq_signal[:len(power_signal)]
        mid_freq_signal = mid_freq_signal[:len(power_signal)]
        high_freq_signal = high_freq_signal[:len(power_signal)]

    return {
        "low": low_freq_signal,
        "mid": mid_freq_signal,
        "high": high_freq_signal
    }


# --- 修改区域 END ---


# (数据生成函数与之前一致，此处省略)
def generate_wind_power_data(duration_s, dt_s):
    # ... (no changes here)
    pass


# ... (other data generation functions are unchanged)


if __name__ == '__main__':
    # (系统参数设置与之前一致，此处省略)
    # ...

    # 1. 初始化混合储能系统
    # ... (no changes in HESS initialization)
    hess = HybridEnergyStorageSystem(dt_lower)
    # ... (add units as before)

    # 2. 初始化分层MPC控制器
    ems = HierarchicalMPCEms(hess, horizon_upper, horizon_lower)

    # 3. 主仿真循环
    # (初始化结果记录与之前一致)
    results = {
        # ...
    }

    for k_lower, t_lower in enumerate(time_series_lower):
        current_soc = hess.get_all_soc()

        # --- 修改区域 START: 在MPC调用前进行信号分解 ---

        # 1. 准备用于分解的净负荷预测序列
        net_load_fine_forecast = net_load_fine[k_lower: k_lower + ems.PH_lower]

        # 如果预测序列长度不足一个下层MPC时域，则用最后一个值填充
        if len(net_load_fine_forecast) < ems.PH_lower:
            padding_size = ems.PH_lower - len(net_load_fine_forecast)
            net_load_fine_forecast = np.pad(net_load_fine_forecast, (0, padding_size), 'edge')

        # 2. 调用小波包分解函数，生成未来一个下层时域长度的任务信号
        decomposed_signals = decompose_power_signal(net_load_fine_forecast, wavelet='db4', level=3)

        slow_task_signal = decomposed_signals["low"]
        mid_task_signal = decomposed_signals["mid"]
        high_task_signal = decomposed_signals["high"]

        # --- 修改区域 END ---

        # 4. 上层MPC求解
        dispatch_upper = None
        if t_lower % dt_upper == 0:
            k_upper = int(t_lower / dt_upper)

            # 准备上层MPC所需的预测数据
            net_load_upper_forecast = net_load_upper[k_upper: k_upper + ems.PH_upper]
            prices_upper_forecast = grid_prices_upper[k_upper: k_upper + ems.PH_upper]

            # 将下层时间尺度的低频信号降采样至上层时间尺度
            # 例如，每15个点取一个点作为上层MPC的跟踪目标
            downsample_ratio = int(dt_upper / dt_lower)
            slow_task_signal_upper = slow_task_signal[::downsample_ratio]

            # 确保输入长度匹配
            if len(slow_task_signal_upper) < ems.PH_upper:
                padding_size = ems.PH_upper - len(slow_task_signal_upper)
                slow_task_signal_upper = np.pad(slow_task_signal_upper, (0, padding_size), 'edge')

            dispatch_upper = ems.solve_upper_level(
                current_soc,
                net_load_upper_forecast,
                prices_upper_forecast,
                slow_task_signal_upper  # 传递新的低频任务信号
            )

        # 5. 下层MPC求解
        # 从上层结果中获取慢速储能的计划出力，作为下层的参考信号
        # 这个参考信号现在的作用更像是“已知扰动”，而不是跟踪目标
        slow_asset_dispatch_plan = {}
        if dispatch_upper and dispatch_upper.get("status") == "optimal":
            slow_asset_dispatch_plan = dispatch_upper["dispatch"]
        else:  # 如果上层未求解或失败，则假设慢速储能不出力
            slow_asset_dispatch_plan = {unit.id: np.zeros(ems.PH_upper) for unit in ems.energy_assets}

        # 将上层计划插值到下层时间尺度
        reference_signals = {"slow_asset_dispatch": {}}
        for uid, dispatch_plan_upper in slow_asset_dispatch_plan.items():
            t_upper_future = np.arange(ems.PH_upper) * dt_upper
            t_lower_future = np.arange(ems.PH_lower) * dt_lower
            dispatch_plan_lower = np.interp(t_lower_future, t_upper_future, dispatch_plan_upper)
            reference_signals["slow_asset_dispatch"][uid] = dispatch_plan_lower

        dispatch_lower = ems.solve_lower_level(
            current_soc,
            reference_signals,  # 包含了慢速储能的计划
            mid_task_signal,  # 传递中频任务信号
            high_task_signal  # 传递高频任务信号
        )

        # 6. 更新系统状态和记录结果
        # (这部分逻辑与之前基本一致，确保从dispatch_lower中正确提取各储能的出力)
        # ...

    # 7. 绘制结果
    # (绘图部分需要增加对分解信号和各储能组出力的可视化)
    # ...