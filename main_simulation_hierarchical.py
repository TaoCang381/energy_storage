# 文件: main_simulation_hierarchical.py (V2.3 - 最终防错健壮版)
# 备注：这是一个完整的、可直接运行的版本，修复了所有已知的变量定义和逻辑衔接问题。
#       增加了边界检查，防止仿真末尾出现索引越界错误。
#       集成了小波包变换(WPT)对净负荷进行功能解耦，为分层MPC提供清晰的任务信号。
#       更新了绘图部分以更好地可视化新框架的运行结果。

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd

from hess_system import HybridEnergyStorageSystem
from mpc_ems_hierarchical import HierarchicalMPCEms

# 导入所有储能模型类
from high_power_density_group.flywheel_simulation import FlywheelModel
from high_power_density_group.supercapacitor_simulation import Supercapacitor
from high_power_density_group.Superconducting_magnetic_energy_storage_simulation import \
    SuperconductingMagneticEnergyStorage
from Medium_power_density_group.electrochemical_energy_storage import ElectrochemicalEnergyStorage
from low_power_density_group.pumped_storage_simulation import PumpedHydroStorage
from low_power_density_group.hydrogen_storage import HydrogenStorage
from low_power_density_group.thermal_storage import ThermalEnergyStorage
from low_power_density_group.caes_system import DiabaticCAES

# 设置绘图参数，确保中文和负号能正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 核心工具函数：小波包分解
# =============================================================================
def decompose_power_signal(power_signal, wavelet='db4', level=3):
    """
    【健壮版】使用小波包变换将功率信号分解为不同频段的子信号。
    本版本采用更可靠的系数重构方法，确保输出的稳定性和正确性。
    """
    power_signal = np.asarray(power_signal)  # 确保输入是numpy数组
    original_len = len(power_signal)

    # 执行小波包分解
    wp = pywt.WaveletPacket(data=power_signal, wavelet=wavelet, mode='symmetric', maxlevel=level)

    # 获取按频率排序的终端节点路径
    nodes = wp.get_level(level, order='freq')
    node_paths = [node.path for node in nodes]

    # 定义频带分配 (这是一个关键的设计点，可以根据您的储能特性调整)
    # 3层分解得到8个频带，路径从 'aaa' (最低频) 到 'ddd' (最高频)
    low_freq_paths = node_paths[:2]  # 最低的2个频带 -> 能量型储能
    mid_freq_paths = node_paths[2:4]  # 中间的2个频带 -> 平滑型储能 (EES)
    high_freq_paths = node_paths[4:]  # 最高的4个频带 -> 功率型储能

    # 通过系数筛选和重构，生成各频带信号
    def reconstruct_from_paths(full_wp, target_paths, max_level):
        # 创建一个空的小波包结构
        temp_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        for path in target_paths:
            # 仅将目标频带的系数填入
            temp_wp[path] = full_wp[path].data
        # 重构信号
        return temp_wp.reconstruct(update=False)

    low_freq_signal = reconstruct_from_paths(wp, low_freq_paths, level)
    mid_freq_signal = reconstruct_from_paths(wp, mid_freq_paths, level)
    high_freq_signal = reconstruct_from_paths(wp, high_freq_paths, level)

    # 裁剪信号以匹配原始长度，防止因小波包内部padding导致长度不一致
    return {
        "low": low_freq_signal[:original_len],
        "mid": mid_freq_signal[:original_len],
        "high": high_freq_signal[:original_len]
    }


# =============================================================================
# 2. 数据生成函数 (为确保完整性，此处包含所有函数定义)
# =============================================================================
def generate_wind_power_data(duration_s, dt_s):
    time_series = np.arange(0, duration_s, dt_s)
    base_wind = 5e6 * (np.sin(2 * np.pi * time_series / 86400) + 1.5)
    noise = 1e6 * np.random.randn(len(time_series))
    raw_wind_fine = base_wind + noise
    raw_wind_fine[raw_wind_fine < 0] = 0
    downsample_ratio = int(900 / dt_s)
    raw_wind_upper = raw_wind_fine[::downsample_ratio]
    return raw_wind_fine, raw_wind_upper


def generate_solar_power_data(duration_s, dt_s):
    time_series = np.arange(0, duration_s, dt_s)
    solar_noon = duration_s / 2
    solar_intensity = - (time_series - solar_noon) ** 2 / (solar_noon ** 2) + 1
    solar_intensity[solar_intensity < 0] = 0
    base_solar = 8e6 * solar_intensity
    cloud_effect = np.ones(len(time_series), dtype=float)
    for _ in range(5):
        start = np.random.randint(0, len(time_series) - 3600)
        end = start + np.random.randint(300, 1800)
        cloud_effect[start:end] *= np.random.uniform(0.2, 0.6)
    solar_power_fine = base_solar * cloud_effect
    downsample_ratio = int(900 / dt_s)
    solar_power_upper = solar_power_fine[::downsample_ratio]
    return solar_power_fine, solar_power_upper


def generate_load_data(duration_s, dt_s):
    time_series = np.arange(0, duration_s, dt_s)
    base_load = 10e6 + 5e6 * np.sin(2 * np.pi * (time_series - 6 * 3600) / 86400) + \
                3e6 * np.sin(4 * np.pi * (time_series - 9 * 3600) / 86400)
    noise = 0.5e6 * np.random.randn(len(time_series))
    load_power_fine = base_load + noise
    downsample_ratio = int(900 / dt_s)
    load_power_upper = load_power_fine[::downsample_ratio]
    return load_power_fine, load_power_upper


def generate_grid_price_data(duration_s, dt_s):
    time_series = np.arange(0, duration_s, dt_s)
    prices = np.ones(len(time_series)) * 300
    hour = (time_series / 3600) % 24
    prices[(hour >= 10) & (hour < 15)] = 600
    prices[(hour >= 18) & (hour < 21)] = 700
    prices[(hour >= 0) & (hour < 6)] = 150
    downsample_ratio = int(900 / dt_s)
    return prices[::downsample_ratio]


# =============================================================================
# 3. 主仿真程序
# =============================================================================
if __name__ == '__main__':
    # 系统参数设置
    duration = 24 * 3600
    dt_lower = 1
    dt_upper = 15 * 60

    horizon_lower_s = 10 * 60
    horizon_upper_s = 6 * 3600

    horizon_lower = int(horizon_lower_s / dt_lower)
    horizon_upper = int(horizon_upper_s / dt_upper)

    time_series_lower = np.arange(0, duration, dt_lower)
    time_series_upper = np.arange(0, duration, dt_upper)

    # 生成原始数据
    raw_wind_fine, raw_wind_upper = generate_wind_power_data(duration, dt_lower)
    solar_power_fine, solar_power_upper = generate_solar_power_data(duration, dt_lower)
    load_power_fine, load_power_upper = generate_load_data(duration, dt_lower)
    grid_prices_upper = generate_grid_price_data(duration, dt_lower)

    # 计算净负荷
    net_load_fine = load_power_fine - (raw_wind_fine + solar_power_fine)
    net_load_upper = load_power_upper - (raw_wind_upper + solar_power_upper)

    # 1. 初始化混合储能系统 (HESS)
    hess = HybridEnergyStorageSystem(dt_lower)
    hess.add_unit(FlywheelModel(ess_id='fw', dt_s=dt_lower))
    hess.add_unit(Supercapacitor(ess_id='sc', dt_s=dt_lower))
    hess.add_unit(SuperconductingMagneticEnergyStorage(ess_id='smes', dt_s=dt_lower))
    hess.add_unit(ElectrochemicalEnergyStorage(ess_id='ees', dt_s=dt_lower))
    hess.add_unit(PumpedHydroStorage(ess_id='phs', dt_s=dt_lower))
    hess.add_unit(HydrogenStorage(ess_id='hes', dt_s=dt_lower))
    hess.add_unit(ThermalEnergyStorage(ess_id='tes', dt_s=dt_lower))
    hess.add_unit(DiabaticCAES(ess_id='caes', dt_s=dt_lower))

    # 2. 初始化分层MPC控制器
    ems = HierarchicalMPCEms(hess, horizon_upper, horizon_lower, dt_upper, dt_lower)

    # 3. 初始化结果记录字典
    results = {
        'p_hess_total': [], 'p_grid_exchange': [],
        'soc': {uid: [] for uid in hess.all_units.keys()},
        'dispatch': {uid: [] for uid in hess.all_units.keys()}
    }

    # 4. 主仿真循环
    p_grid_plan_upper = np.zeros(ems.PH_upper)
    slow_asset_dispatch_plan_upper = {unit.id: np.zeros(ems.PH_upper) for unit in ems.energy_assets}

    for k_lower, t_lower in enumerate(time_series_lower):
        if k_lower % 900 == 0:
            print(f"仿真进行中... 时间: {t_lower / 3600:.2f}h / {duration / 3600}h")

        current_soc = hess.get_all_soc()

        # A. 信号分解
        net_load_fine_forecast = net_load_fine[k_lower: k_lower + ems.PH_lower]
        if len(net_load_fine_forecast) < ems.PH_lower:
            padding_size = ems.PH_lower - len(net_load_fine_forecast)
            net_load_fine_forecast = np.pad(net_load_fine_forecast, (0, padding_size), 'edge')

        decomposed_signals = decompose_power_signal(net_load_fine_forecast, wavelet='db4', level=3)
        slow_task_signal = decomposed_signals["low"]
        mid_task_signal = decomposed_signals["mid"]
        high_task_signal = decomposed_signals["high"]

        # B. 上层MPC求解 (周期性执行)
        if k_lower % int(dt_upper / dt_lower) == 0:
            k_upper = int(t_lower // dt_upper)

            # ======================= ## 核心修正 ## =======================
            # 在进行数据切片前，先检查索引是否会越界
            if k_upper < len(net_load_upper):
                # 准备上层MPC所需的预测数据
                net_load_upper_forecast = net_load_upper[k_upper: k_upper + ems.PH_upper]
                prices_upper_forecast = grid_prices_upper[k_upper: k_upper + ems.PH_upper]

                # 如果预测时域超出了数据末尾，则用最后一个有效值进行填充
                if len(net_load_upper_forecast) < ems.PH_upper:
                    pad_size = ems.PH_upper - len(net_load_upper_forecast)
                    net_load_upper_forecast = np.pad(net_load_upper_forecast, (0, pad_size), 'edge')
                    prices_upper_forecast = np.pad(prices_upper_forecast, (0, pad_size), 'edge')

                # 降采样低频信号
                downsample_ratio = int(dt_upper / dt_lower)
                slow_task_signal_upper = slow_task_signal[::downsample_ratio]
                if len(slow_task_signal_upper) < ems.PH_upper:
                    pad_size = ems.PH_upper - len(slow_task_signal_upper)
                    slow_task_signal_upper = np.pad(slow_task_signal_upper, (0, pad_size), 'edge')

                # 调用上层MPC求解器
                dispatch_upper = ems.solve_upper_level(
                    current_soc, net_load_upper_forecast,
                    prices_upper_forecast, slow_task_signal_upper
                )

                # 更新上层计划
                if dispatch_upper and dispatch_upper.get("status") == "optimal":
                    p_grid_plan_upper = dispatch_upper["grid_exchange"]
                    slow_asset_dispatch_plan_upper = dispatch_upper["dispatch"]
                else:
                    p_grid_plan_upper.fill(0)
                    for plan in slow_asset_dispatch_plan_upper.values():
                        plan.fill(0)
            # 如果 k_upper 越界，则不执行任何操作，沿用上一次的计划
            # =============================================================

        # C. 下层MPC求解
        slow_asset_dispatch_ref_lower = {}
        t_upper_future = np.arange(ems.PH_upper) * dt_upper
        t_lower_future = np.arange(ems.PH_lower) * dt_lower
        for uid, dispatch_plan_upper in slow_asset_dispatch_plan_upper.items():
            dispatch_plan_lower = np.interp(t_lower_future, t_upper_future, dispatch_plan_upper)
            slow_asset_dispatch_ref_lower[uid] = dispatch_plan_lower

        dispatch_lower = ems.solve_lower_level(
            current_soc, slow_asset_dispatch_ref_lower,
            mid_task_signal, high_task_signal
        )

        # D. 更新系统状态和记录结果
        current_dispatch = {}
        for uid, pwr in dispatch_lower.items():
            current_dispatch[uid] = pwr
        for uid, pwr_plan in slow_asset_dispatch_ref_lower.items():
            current_dispatch[uid] = pwr_plan[0]

        hess.update_all_states(current_dispatch)

        total_hess_power = 0
        for uid, unit in hess.all_units.items():
            power = current_dispatch.get(uid, 0)
            results['dispatch'][uid].append(power)
            results['soc'][uid].append(unit.soc)
            total_hess_power += power
        results['p_hess_total'].append(total_hess_power)

        actual_grid_exchange = net_load_fine[k_lower] - total_hess_power
        results["p_grid_exchange"].append(actual_grid_exchange)

    # 5. 绘制仿真结果
    time_h_lower = time_series_lower / 3600
    time_h_upper = time_series_upper / 3600

    # 图1：整体功率平衡情况
    fig1, axs1 = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axs1[0].plot(time_h_lower, net_load_fine / 1e6, label="本地净负荷 (MW)", color='lightblue', zorder=1)
    axs1[0].set_title("系统整体功率平衡与电价", fontsize=16)
    axs1[0].set_ylabel("功率 (MW)")
    axs1[0].grid(True)
    ax_price = axs1[0].twinx()
    ax_price.plot(time_h_upper, grid_prices_upper, 'r--', label="电价 (元/MWh)", alpha=0.7, zorder=2)
    ax_price.set_ylabel("电价 (元/MWh)")
    lines, labels = axs1[0].get_legend_handles_labels()
    lines2, labels2 = ax_price.get_legend_handles_labels()
    ax_price.legend(lines + lines2, labels + labels2, loc='upper right')
    axs1[1].plot(time_h_lower, np.array(results["p_hess_total"]) / 1e6, 'k-', label="HESS总出力 (MW)", zorder=3)
    axs1[1].fill_between(time_h_lower, np.array(results["p_grid_exchange"]) / 1e6,
                         label="电网购电(正)/售电(负) (MW)", color='gray', alpha=0.5, zorder=1)
    axs1[1].set_xlabel("时间 (小时)", fontsize=12)
    axs1[1].set_ylabel("功率 (MW)")
    axs1[1].grid(True)
    axs1[1].legend()
    fig1.tight_layout()

    # 图2：分解的任务信号
    fig2, axs2 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    plot_len = int(2 * 3600 / dt_lower)
    time_h_plot = time_h_lower[:plot_len]
    decomposed_plot_signals = decompose_power_signal(net_load_fine[:plot_len], wavelet='db4', level=3)
    axs2[0].plot(time_h_plot, decomposed_plot_signals['low'] / 1e6, label="低频任务信号 (能量型储能)")
    axs2[0].set_title("小波包分解后的功率任务信号 (前2小时)", fontsize=16)
    axs2[1].plot(time_h_plot, decomposed_plot_signals['mid'] / 1e6, label="中频任务信号 (平滑型储能)")
    axs2[2].plot(time_h_plot, decomposed_plot_signals['high'] / 1e6, label="高频任务信号 (功率型储能)")
    for ax in axs2:
        ax.set_ylabel("功率 (MW)")
        ax.grid(True)
        ax.legend()
    axs2[2].set_xlabel("时间 (小时)", fontsize=12)
    fig2.tight_layout()

    # 图3：各储能组出力情况
    fig3, axs3 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    p_energy = np.sum([results['dispatch'][u.id] for u in ems.energy_assets], axis=0)
    p_smoothing = np.sum([results['dispatch'][u.id] for u in ems.smoothing_assets], axis=0)
    p_power = np.sum([results['dispatch'][u.id] for u in ems.power_assets], axis=0)
    axs3[0].plot(time_h_lower, p_energy / 1e6, label="能量型储能组总出力")
    axs3[0].set_title("各功能储能组出力", fontsize=16)
    axs3[1].plot(time_h_lower, p_smoothing / 1e6, label="平滑型储能组总出力")
    axs3[2].plot(time_h_lower, p_power / 1e6, label="功率型储能组总出力")
    for ax in axs3:
        ax.set_ylabel("功率 (MW)")
        ax.grid(True)
        ax.legend()
    axs3[2].set_xlabel("时间 (小时)", fontsize=12)
    fig3.tight_layout()

    # 图4：各类储能SOC变化
    fig4, axs4 = plt.subplots(len(hess.all_units), 1, figsize=(16, 20), sharex=True)
    fig4.suptitle("所有储能单元SOC变化", fontsize=16)
    for i, (uid, unit) in enumerate(hess.all_units.items()):
        axs4[i].plot(time_h_lower, results['soc'][uid], label=f"{unit.id} 的SOC曲线")
        axs4[i].set_ylabel(f"SOC_{uid}")
        axs4[i].grid(True)
        axs4[i].legend()
        axs4[i].set_ylim(-0.05, 1.05)
    axs4[-1].set_xlabel("时间 (小时)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()