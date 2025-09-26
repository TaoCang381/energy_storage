# file: main_simulation_hierarchical.py (V3.3 - 优化单位统一修正版)
# 备注：本版本将优化计算中的所有功率单位统一为兆瓦(MW)以提高求解器稳定性，
#       但在与储能物理模型交互时，仍将单位转换回瓦特(W)。

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from tqdm import tqdm
# 导入所有必要的模块
from hess_system import HybridEnergyStorageSystem
from mpc_ems_hierarchical import HierarchicalMPCEms
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
    # --- 仿真参数设置 ---
    duration = 1 * 3600  # 仿真总时长 (s)
    dt_lower = 1  # 下层控制时间步长 (s)
    dt_upper = 1 * 60  # 上层控制时间步长 (s)

    # --- 预测时域设置 ---
    horizon_lower = int(60 / dt_lower)  # 下层MPC预测时域 (60秒)
    horizon_upper = int(6 * 3600 / dt_upper)  # 上层MPC预测时域 (6小时)

    # --- 时间序列生成 ---
    time_series_lower = np.arange(0, duration, dt_lower)
    time_series_upper = np.arange(0, duration, dt_upper)

    # --- 生成原始数据 (单位: W) ---
    raw_wind_fine, raw_wind_upper = generate_wind_power_data(duration, dt_lower)
    solar_power_fine, solar_power_upper = generate_solar_power_data(duration, dt_lower)
    load_power_fine, load_power_upper = generate_load_data(duration, dt_lower)
    grid_prices_upper = generate_grid_price_data(duration, dt_upper)

    # --- 计算净负荷 (单位: W) ---
    net_load_fine = load_power_fine - (raw_wind_fine + solar_power_fine)
    net_load_upper = load_power_upper - (raw_wind_upper + solar_power_upper)

    # --- 初始化混合储能系统 (HESS) ---
    hess = HybridEnergyStorageSystem(dt_lower)
    hess.add_unit(FlywheelModel(id='fw', dt_s=dt_lower))
    hess.add_unit(Supercapacitor(id='sc', dt_s=dt_lower))
    hess.add_unit(SuperconductingMagneticEnergyStorage(id='smes', dt_s=dt_lower))
    hess.add_unit(ElectrochemicalEnergyStorage(id='ees', dt_s=dt_lower))
    hess.add_unit(PumpedHydroStorage(id='phs', dt_s=dt_lower))
    hess.add_unit(HydrogenStorage(id='hes', dt_s=dt_lower))
    hess.add_unit(ThermalEnergyStorage(id='tes', dt_s=dt_lower))
    hess.add_unit(DiabaticCAES(id='caes', dt_s=dt_lower))

    # --- 初始化分层模型预测控制器 (EMS) ---
    ems = HierarchicalMPCEms(hess, horizon_upper, horizon_lower)

    # --- 初始化结果记录字典 ---
    results = {'p_hess_total': [], 'p_grid_exchange': [], 'soc': {uid: [] for uid in hess.all_units.keys()},
               'dispatch': {uid: [] for uid in hess.all_units.keys()}}
    p_grid_plan_upper = np.zeros(ems.PH_upper)  # 单位: MW
    slow_asset_dispatch_plan_upper = {unit.id: np.zeros(ems.PH_upper) for unit in ems.energy_assets}  # 单位: MW

    # --- 仿真循环开始 ---
    for k_lower, t_lower in tqdm(enumerate(time_series_lower), total=len(time_series_lower), desc="HESS 仿真进行中"):
     #   if k_lower % 3600 == 0:
      #      print(f"仿真进行中... 时间: {t_lower / 3600:.2f}h / {duration / 3600:.0f}h")

        # 1. 获取当前所有储能单元的SOC
        current_soc = hess.get_all_soc()
        for uid, soc_val in current_soc.items():
            if soc_val is None:
                print(f"警告: 在 t={t_lower}s, 单元 {uid} 的 SOC 为 None。将使用默认值 0.5。")
                current_soc[uid] = 0.5
        # 2. 准备下层MPC所需的净负荷预测
        net_load_fine_forecast = net_load_fine[k_lower: k_lower + ems.PH_lower]
        if len(net_load_fine_forecast) < ems.PH_lower:
            padding_size = ems.PH_lower - len(net_load_fine_forecast)
            net_load_fine_forecast = np.pad(net_load_fine_forecast, (0, padding_size), 'edge')

        # 3. 对净负荷进行小波包分解
        decomposed_signals = decompose_power_signal(net_load_fine_forecast, wavelet='db4', level=3)

        # 核心修正：将分解后的信号从 W 转换为 MW，以用于优化计算
        slow_task_signal_mw = decomposed_signals["low"] / 1e6
        mid_task_signal_mw = decomposed_signals["mid"] / 1e6
        high_task_signal_mw = decomposed_signals["high"] / 1e6

        # 4. 上层MPC决策 (每15分钟执行一次)
        if t_lower % dt_upper == 0:
            k_upper = int(t_lower // dt_upper)

            # 准备上层MPC所需的价格和低频任务信号预测
            prices_upper_forecast = grid_prices_upper[k_upper: k_upper + ems.PH_upper]
            if len(prices_upper_forecast) < ems.PH_upper:
                prices_upper_forecast = np.pad(prices_upper_forecast, (0, ems.PH_upper - len(prices_upper_forecast)),
                                               'edge')

            downsample_ratio = int(dt_upper / dt_lower)
            slow_task_signal_upper_mw = slow_task_signal_mw[::downsample_ratio]
            if len(slow_task_signal_upper_mw) < ems.PH_upper:
                padding_size = ems.PH_upper - len(slow_task_signal_upper_mw)
                slow_task_signal_upper_mw = np.pad(slow_task_signal_upper_mw, (0, padding_size), 'edge')

            # 调用上层求解器 (所有功率单位均为 MW)
            dispatch_upper = ems.solve_upper_level(current_soc, prices_upper_forecast, slow_task_signal_upper_mw)

            if dispatch_upper and dispatch_upper.get("status") == "optimal":
                p_grid_plan_upper = dispatch_upper["grid_exchange"]  # 结果是 MW
                slow_asset_dispatch_plan_upper = dispatch_upper["dispatch"]  # 结果是 MW
            else:
                p_grid_plan_upper = np.zeros(ems.PH_upper)
                slow_asset_dispatch_plan_upper = {unit.id: np.zeros(ems.PH_upper) for unit in ems.energy_assets}

        # 5. 下层MPC决策 (每个时间步都执行)
        # 调用下层求解器 (所有功率单位均为 MW)
        dispatch_lower_mw = ems.solve_lower_level(current_soc, mid_task_signal_mw, high_task_signal_mw)

        # 6. 合成最终调度指令
        current_dispatch_mw = dispatch_lower_mw.copy()
        k_in_upper_plan_float = (t_lower % dt_upper) / dt_upper

        # 从上层计划中插值得到当前时刻的低频部分指令
        for uid, plan in slow_asset_dispatch_plan_upper.items():
            if plan is not None and len(plan) > 0:
                current_dispatch_mw[uid] = np.interp(k_in_upper_plan_float,
                                                     np.linspace(0, 1, len(plan), endpoint=False), plan)
            else:
                current_dispatch_mw[uid] = 0

        # 核心修正：将最终的调度指令从 MW 转换回 W，以供储能物理模型使用
        current_dispatch_watts = {uid: p_mw * 1e6 for uid, p_mw in current_dispatch_mw.items() if p_mw is not None}

        # 7. 更新HESS状态并记录结果
        hess.update_all_states(current_dispatch_watts)

        total_hess_power_watts = 0
        for uid, unit in hess.all_units.items():
            power_watts = current_dispatch_watts.get(uid, 0)
            results['dispatch'][uid].append(power_watts)
            results['soc'][uid].append(unit.soc)
            total_hess_power_watts += power_watts
        results['p_hess_total'].append(total_hess_power_watts)

        # 从上层电网计划中插值得到当前时刻的交换功率 (单位: W)
        current_p_grid_from_plan_mw = 0
        if p_grid_plan_upper is not None and len(p_grid_plan_upper) > 0:
            current_p_grid_from_plan_mw = np.interp(k_in_upper_plan_float,
                                                    np.linspace(0, 1, len(p_grid_plan_upper), endpoint=False),
                                                    p_grid_plan_upper)
        results["p_grid_exchange"].append(current_p_grid_from_plan_mw * 1e6)

    # --- 仿真结束，开始绘图 ---
    print("仿真完成，正在生成结果图像...")

    time_h_lower = time_series_lower / 3600
    time_h_upper = time_series_upper / 3600

    # 图1：净负荷与HESS总响应功率
    plt.figure(figsize=(12, 6))
    plt.plot(time_h_lower, np.array(net_load_fine) / 1e6, label='净负荷 (MW)', alpha=0.7)
    plt.plot(time_h_lower, np.array(results['p_hess_total']) / 1e6, label='HESS总输出功率 (MW)', linestyle='--')
    plt.plot(time_h_lower, np.array(results['p_grid_exchange']) / 1e6, label='计划电网交换功率 (MW)', linestyle=':')
    plt.xlabel('时间 (小时)')
    plt.ylabel('功率 (MW)')
    plt.title('净负荷与混合储能系统响应')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 图2：各类储能单元的SOC变化
    plt.figure(figsize=(12, 8))
    # 能量型
    for unit in ems.energy_assets:
        plt.plot(time_h_lower, results['soc'][unit.id], label=f'SOC - {unit.id.upper()}')
    # 平滑型
    for unit in ems.smoothing_assets:
        plt.plot(time_h_lower, results['soc'][unit.id], label=f'SOC - {unit.id.upper()}', linestyle='--')
    # 功率型
    for unit in ems.power_assets:
        plt.plot(time_h_lower, results['soc'][unit.id], label=f'SOC - {unit.id.upper()}', linestyle=':')
    plt.xlabel('时间 (小时)')
    plt.ylabel('SOC (荷电状态)')
    plt.title('储能单元SOC变化曲线')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 图3：各类储能单元的调度功率
    plt.figure(figsize=(12, 8))
    # 能量型
    for unit in ems.energy_assets:
        plt.plot(time_h_lower, np.array(results['dispatch'][unit.id]) / 1e6, label=f'功率 - {unit.id.upper()}')
    # 平滑型
    for unit in ems.smoothing_assets:
        plt.plot(time_h_lower, np.array(results['dispatch'][unit.id]) / 1e6, label=f'功率 - {unit.id.upper()}',
                 linestyle='--')
    # 功率型
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