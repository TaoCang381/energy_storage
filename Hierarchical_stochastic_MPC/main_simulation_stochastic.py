# file: main_simulation_hierarchical.py (V3.3 - 优化单位统一修正版)
# 备注：本版本将优化计算中的所有功率单位统一为兆瓦(MW)以提高求解器稳定性，
#       但在与储能物理模型交互时，仍将单位转换回瓦特(W)。
import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from tqdm import tqdm
# 导入所有必要的模块
from hess_system import HybridEnergyStorageSystem
from Hierarchical_stochastic_MPC.mpc_ems_stochastic import HierarchicalMPCEms
from scenario_generation import generate_scenarios

from base_storage_model import BaseStorageModel

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

# 设置绘图字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 核心工具函数：小波包分解
# =============================================================================
def decompose_power_signal(power_signal, wavelet='db4', level=3):
    """
    使用小波包变换将功率信号分解为不同频段的子信号。
    """
    power_signal = np.asarray(power_signal)
    original_len = len(power_signal)

    # 执行完整的小波包分解
    wp_full = pywt.WaveletPacket(data=power_signal, wavelet=wavelet, mode='symmetric', maxlevel=level)

    # 获取按频率排序的所有最深层的节点路径
    nodes = wp_full.get_level(level, order='freq')
    node_paths = [str(node.path) for node in nodes]

    # 增强鲁棒性：验证节点数量
    expected_node_count = 2 ** level
    if len(node_paths) != expected_node_count:
        if original_len < expected_node_count:
            print(f"警告: 信号长度({original_len})不足以支持{level}层分解，将返回零信号。")
            return {"low": np.zeros(original_len), "mid": np.zeros(original_len), "high": np.zeros(original_len)}
        else:
            raise ValueError(f"小波包分解节点数量异常！预期 {expected_node_count}，实际 {len(node_paths)}")

    # 定义频带分配
    low_freq_paths = node_paths[:2]
    mid_freq_paths = node_paths[2:4]
    high_freq_paths = node_paths[4:]

    # 通过系数筛选和重构，生成各频带信号
    def reconstruct_from_paths(full_wp, target_paths, max_level):
        temp_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        for path_str in target_paths:
            try:
                temp_wp[path_str] = full_wp[path_str].data
            except TypeError:
                path_tuple = tuple(path_str)
                if path_tuple in full_wp:
                    temp_wp[path_tuple] = full_wp[path_tuple].data
        return temp_wp.reconstruct(update=False)

    low_freq_signal = reconstruct_from_paths(wp_full, low_freq_paths, level)
    mid_freq_signal = reconstruct_from_paths(wp_full, mid_freq_paths, level)
    high_freq_signal = reconstruct_from_paths(wp_full, high_freq_paths, level)

    # 裁剪信号以匹配原始长度
    return {
        "low": low_freq_signal[:original_len],
        "mid": mid_freq_signal[:original_len],
        "high": high_freq_signal[:original_len]
    }


# =============================================================================
# 2. 数据生成函数 (单位: W)
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
    cloud_effect = np.ones_like(time_series, dtype=float)

    # --- 核心修正：添加一个判断条件 ---
    # 只有当时间序列的点数足够多时，才添加云层效果
    # 这里的 3600 是原代码中硬编码的数值，我们保留它作为判断依据
    if len(time_series) > 3600:
        for _ in range(10):
            # 确保随机起始点不会导致数组越界
            start = np.random.randint(0, len(time_series) - 3600)
            end = start + np.random.randint(300, 1800)
            cloud_effect[start:end] *= np.random.uniform(0.2, 0.6)
    # --- 修正结束 ---

    solar_power_fine = base_solar * cloud_effect
    downsample_ratio = int(900 / dt_s) if dt_s > 0 else 1 # 避免除零错误
    if downsample_ratio == 0: downsample_ratio = 1
    solar_power_upper = solar_power_fine[::downsample_ratio]
    return solar_power_fine, solar_power_upper


def generate_load_data(duration_s, dt_s):
    time_series = np.arange(0, duration_s, dt_s)
    base_load = 10e6 + 5e6 * np.sin(2 * np.pi * (time_series - 6 * 3600) / 86400) + 3e6 * np.sin(
        4 * np.pi * (time_series - 9 * 3600) / 86400)
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
    return prices


# =============================================================================
# 3. 主仿真程序
# =============================================================================
if __name__ == '__main__':
    # --- 仿真参数设置 (不变) ---
    duration = 1 * 3600
    dt_lower = 5
    dt_upper = 15 * 60
    horizon_lower = int(60 / dt_lower)
    horizon_upper = int(6 * 3600 / dt_upper)

    # --- 数据生成 (不变) ---
    time_series_lower = np.arange(0, duration, dt_lower)
    time_series_upper = np.arange(0, duration, dt_upper)
    raw_wind_fine, raw_wind_upper = generate_wind_power_data(duration, dt_lower)
    solar_power_fine, solar_power_upper = generate_solar_power_data(duration, dt_lower)
    load_power_fine, load_power_upper = generate_load_data(duration, dt_lower)
    grid_prices_upper = generate_grid_price_data(duration, dt_upper)
    net_load_fine = load_power_fine - (raw_wind_fine + solar_power_fine)
    net_load_upper = load_power_upper - (raw_wind_upper + solar_power_upper)

    # --- HESS 和 EMS 初始化 (不变) ---
    hess = HybridEnergyStorageSystem(dt_lower)
    # ... (请保留你原来的 hess.add_unit(...) 代码)
    hess.add_unit(FlywheelModel(id='fw', dt_s=dt_lower))
    hess.add_unit(Supercapacitor(id='sc', dt_s=dt_lower))
    hess.add_unit(SuperconductingMagneticEnergyStorage(id='smes', dt_s=dt_lower))
    hess.add_unit(ElectrochemicalEnergyStorage(id='ees', dt_s=dt_lower))
    hess.add_unit(PumpedHydroStorage(id='phs', dt_s=dt_lower))
    hess.add_unit(HydrogenStorage(id='hes', dt_s=dt_lower))
    hess.add_unit(ThermalEnergyStorage(id='tes', dt_s=dt_lower))
    hess.add_unit(DiabaticCAES(id='caes', dt_s=dt_lower))
    ems = HierarchicalMPCEms(hess, horizon_upper, horizon_lower)

    # --- 场景生成 (不变) ---
    print("正在生成不确定性场景...")
    num_scenarios = 10
    net_load_std_dev_mw = np.mean(np.abs(net_load_upper)) / 1e6 * 0.15
    cov_matrix = np.array([[net_load_std_dev_mw ** 2]])
    scenarios_base, probabilities = generate_scenarios(
        num_samples=1000, num_scenarios=num_scenarios,
        mean=np.array([0]), cov_matrix=cov_matrix
    )
    scenarios_mw = np.random.randn(num_scenarios, horizon_upper) * scenarios_base
    print(f"{num_scenarios}个场景生成完毕。")

    # --- 结果记录初始化 (不变) ---
    results = {'p_hess_total': [], 'p_grid_exchange': [], 'soc': {uid: [] for uid in hess.all_units.keys()},
               'dispatch': {uid: [] for uid in hess.all_units.keys()}}
    p_grid_plan_upper = np.zeros(ems.PH_upper)
    slow_asset_dispatch_plan_upper = {unit.id: np.zeros(ems.PH_upper) for unit in ems.energy_assets}

    # --- 仿真循环开始 ---
    for k_lower, t_lower in tqdm(enumerate(time_series_lower), total=len(time_series_lower), desc="HESS 仿真进行中"):
        current_soc = hess.get_all_soc()
        for uid, soc_val in current_soc.items():
            if soc_val is None: current_soc[uid] = 0.5

        # 1. 上层MPC决策 (每15分钟执行一次，制定经济计划)
        if t_lower % dt_upper == 0:
            k_upper = int(t_lower // dt_upper)
            net_load_upper_forecast = net_load_upper[k_upper: k_upper + ems.PH_upper]
            if len(net_load_upper_forecast) < ems.PH_upper:
                net_load_upper_forecast = np.pad(net_load_upper_forecast,
                                                 (0, ems.PH_upper - len(net_load_upper_forecast)), 'edge')
            net_load_upper_forecast_mw = net_load_upper_forecast / 1e6

            prices_upper_forecast = grid_prices_upper[k_upper: k_upper + ems.PH_upper]
            if len(prices_upper_forecast) < ems.PH_upper:
                prices_upper_forecast = np.pad(prices_upper_forecast, (0, ems.PH_upper - len(prices_upper_forecast)),
                                               'edge')

            dispatch_upper = ems.solve_stochastic_upper_level(
                current_soc, prices_upper_forecast, net_load_upper_forecast_mw, scenarios_mw, probabilities
            )
            if dispatch_upper and dispatch_upper.get("status") == "optimal":
                p_grid_plan_upper = dispatch_upper["grid_exchange"]
                slow_asset_dispatch_plan_upper = dispatch_upper["dispatch"]
            else:
                p_grid_plan_upper = np.zeros(ems.PH_upper)
                slow_asset_dispatch_plan_upper = {unit.id: np.zeros(ems.PH_upper) for unit in ems.energy_assets}

        # <--- CORRECTED LOGIC: 实时控制层 ---

        # 2. 获取当前时刻的上层“经济计划”功率 (单位: W)
        k_in_upper_plan_float = (t_lower % dt_upper) / dt_upper

        # 慢速储能的计划功率 (W)
        planned_slow_dispatch_watts = {}
        for uid, plan in slow_asset_dispatch_plan_upper.items():
            if plan is not None and len(plan) > 0:
                p_mw = np.interp(k_in_upper_plan_float, np.linspace(0, 1, len(plan), endpoint=False), plan)
                planned_slow_dispatch_watts[uid] = p_mw * 1e6
        total_planned_slow_dispatch_watts = sum(planned_slow_dispatch_watts.values())

        # 电网的计划功率 (W)
        planned_grid_exchange_mw = np.interp(k_in_upper_plan_float,
                                             np.linspace(0, 1, len(p_grid_plan_upper), endpoint=False),
                                             p_grid_plan_upper)
        planned_grid_exchange_watts = planned_grid_exchange_mw * 1e6

        # 3. 计算需要快速储能来平衡的“高频不平衡功率” (单位: W)
        # 这个差值是上层计划未能覆盖的、需要实时响应的部分
        actual_net_load_watts = net_load_fine[k_lower]
        imbalance_watts = actual_net_load_watts - (total_planned_slow_dispatch_watts + planned_grid_exchange_watts)

        # 4. 对这个“不平衡功率”进行小波包分解，交给下层MPC处理
        imbalance_forecast_watts = np.full(ems.PH_lower, imbalance_watts)
        decomposed_signals = decompose_power_signal(imbalance_forecast_watts, wavelet='db4', level=3)
        mid_task_signal_mw = decomposed_signals["mid"] / 1e6
        high_task_signal_mw = decomposed_signals["high"] / 1e6

        dispatch_lower_mw = ems.solve_lower_level(current_soc, mid_task_signal_mw, high_task_signal_mw)

        # 5. 合成最终的、发送给物理模型的调度指令 (单位: W)
        current_dispatch_watts = {uid: p_mw * 1e6 for uid, p_mw in dispatch_lower_mw.items()}
        current_dispatch_watts.update(planned_slow_dispatch_watts)

        # 6. 更新HESS状态并记录结果
        hess.update_all_states(current_dispatch_watts)

        total_hess_power_watts = sum(current_dispatch_watts.values())
        results['p_hess_total'].append(total_hess_power_watts)
        results["p_grid_exchange"].append(planned_grid_exchange_watts)  # 记录计划的电网功率
        for uid, unit in hess.all_units.items():
            results['dispatch'][uid].append(current_dispatch_watts.get(uid, 0))
            results['soc'][uid].append(unit.soc)

    # --- 仿真结束，开始绘图 ---
    print("仿真完成，正在生成结果图像...")
    # ... (绘图部分代码保持不变, 此处省略) ...
    time_h_lower = time_series_lower / 3600
    time_h_upper = time_series_upper / 3600

    # 图1
    plt.figure(figsize=(12, 6))
    plt.plot(time_h_lower, np.array(net_load_fine) / 1e6, label='净负荷 (MW)', alpha=0.7)
    plt.plot(time_h_lower, np.array(results['p_hess_total']) / 1e6, label='HESS总输出功率 (MW)', linestyle='--')
    plt.plot(time_h_lower, np.array(results['p_grid_exchange']) / 1e6, label='计划电网交换功率 (MW)', linestyle=':')
    plt.xlabel('时间 (小时)')
    plt.ylabel('功率 (MW)')
    plt.title('净负荷与混合储能系统响应（随机优化+实时控制）')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 图2
    plt.figure(figsize=(12, 8))
    for unit in ems.energy_assets:
        plt.plot(time_h_lower, results['soc'][unit.id], label=f'SOC - {unit.id.upper()}')
    for unit in ems.smoothing_assets:
        plt.plot(time_h_lower, results['soc'][unit.id], label=f'SOC - {unit.id.upper()}', linestyle='--')
    for unit in ems.power_assets:
        plt.plot(time_h_lower, results['soc'][unit.id], label=f'SOC - {unit.id.upper()}', linestyle=':')
    plt.xlabel('时间 (小时)')
    plt.ylabel('SOC (荷电状态)')
    plt.title('储能单元SOC变化曲线')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 图3
    plt.figure(figsize=(12, 8))
    for unit in ems.energy_assets:
        plt.plot(time_h_lower, np.array(results['dispatch'][unit.id]) / 1e6, label=f'功率 - {unit.id.upper()}')
    for unit in ems.smoothing_assets:
        plt.plot(time_h_lower, np.array(results['dispatch'][unit.id]) / 1e6, label=f'功率 - {unit.id.upper()}',
                 linestyle='--')
    for unit in ems.power_assets:
        plt.plot(time_h_lower, np.array(results['dispatch'][unit.id]) / 1e6, label=f'功率 - {unit.id.upper()}',
                 linestyle=':')
    plt.xlabel('时间 (小时)')
    plt.ylabel('功率 (MW)')
    plt.title('储能单元调度功率曲线')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()