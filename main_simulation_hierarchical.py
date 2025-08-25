# file: main_simulation_hierarchical.py (V1.2 - 最终健壮版)
# 备注：修正了主循环逻辑，确保在MPC求解失败时也能记录数据，避免绘图错误。

import numpy as np
import matplotlib.pyplot as plt
from hess_system import HybridEnergyStorageSystem
from mpc_ems_hierarchical import HierarchicalMPCEms

# (导入储能模型部分与之前一致，此处省略)
from high_power_density_group.flywheel_simulation import FlywheelModel
from high_power_density_group.supercapacitor_simulation import Supercapacitor
from high_power_density_group.Superconducting_magnetic_energy_storage_simulation import SuperconductingMagneticEnergyStorage
from Medium_power_density_group.electrochemical_energy_storage import ElectrochemicalEnergyStorage
from low_power_density_group.pumped_storage_simulation import PumpedHydroStorage
from low_power_density_group.hydrogen_storage import HydrogenStorage
from low_power_density_group.thermal_storage import ThermalEnergyStorage
from low_power_density_group.caes_system import DiabaticCAES

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# (数据生成函数与之前一致，此处省略)
def generate_wind_power_data(duration_s, dt_s):
    window_duration_s = 3600
    window_size = max(1, int(window_duration_s / dt_s))
    num_points = int(duration_s / dt_s)
    time_for_generation = np.arange(num_points + window_size) * dt_s
    base_power = 50e6
    low_freq = 20e6 * np.sin(2 * np.pi * time_for_generation / (12 * 3600))
    medium_freq = 10e6 * np.sin(2 * np.pi * time_for_generation / (2 * 3600))
    high_freq_variance = 3e6
    high_freq = high_freq_variance * np.random.randn(num_points + window_size)
    raw_wind_power_full = base_power + low_freq + medium_freq + high_freq
    raw_wind_power_full = np.maximum(0, raw_wind_power_full)
    target_grid_full = np.convolve(raw_wind_power_full, np.ones(window_size) / window_size, mode='valid')
    time_series = time_for_generation[:num_points]
    raw_wind_power = raw_wind_power_full[:num_points]
    return time_series, raw_wind_power, target_grid_full[:num_points]

def generate_price_signal(time_series):
    prices = np.ones_like(time_series)
    hours = (time_series / 3600) % 24
    prices[((hours >= 8) & (hours < 12)) | ((hours >= 18) & (hours < 22))] = 800
    prices[((hours >= 12) & (hours < 18)) | ((hours >= 22) & (hours < 24))] = 500
    prices[(hours >= 0) & (hours < 8)] = 200
    return prices

# (HESS系统初始化与之前一致，此处省略)
hess = HybridEnergyStorageSystem()
hess.add_unit(FlywheelModel(ess_id="fw_01"), group='fast')
hess.add_unit(Supercapacitor(ess_id="sc_01"), group='fast')
hess.add_unit(SuperconductingMagneticEnergyStorage(ess_id="smes_01"), group='fast')
hess.add_unit(ElectrochemicalEnergyStorage(ess_id="ees_01"), group='medium')
hess.add_unit(PumpedHydroStorage(ess_id="phs_01"), group='long')
hess.add_unit(HydrogenStorage(ess_id="hes_01"), group='long')
hess.add_unit(ThermalEnergyStorage(ess_id="tes_01"), group='long')
hess.add_unit(DiabaticCAES(ess_id="caes_01"), group='long')

# (仿真参数准备与之前一致，此处省略)
dt_upper_s = 60 * 15
dt_lower_s = 60 * 1
steps_per_upper = int(dt_upper_s / dt_lower_s)
duration_s = 3600 * 24
PH_upper = 8
PH_lower = 15
time_series_upper, raw_wind, _ = generate_wind_power_data(duration_s, dt_upper_s)
time_series_lower, raw_wind_fine, _ = generate_wind_power_data(duration_s, dt_lower_s)
solar_power_fine = np.zeros_like(raw_wind_fine)
load_power_fine = (np.mean(raw_wind_fine) * 0.4) + 40e6 + 20e6 * np.sin(2 * np.pi * time_series_lower / (24 * 3600))
grid_prices_upper = generate_price_signal(time_series_upper)
h_mpc_ems = HierarchicalMPCEms(hess, PH_upper, PH_lower)
num_upper_steps = int(duration_s / dt_upper_s)
all_units_list = list(hess.all_units.values())
results = {"p_grid_exchange": [], "p_hess_total": []}
for unit in all_units_list:
    results[f"p_{unit.id.split('_')[0]}"] = []
    results[f"soc_{unit.id.split('_')[0]}"] = []

print("--- 开始分层MPC经济调度仿真 ---")
lower_step_counter = 0
for i in range(num_upper_steps):
    print(f"仿真进度: {(i / num_upper_steps) * 100:.1f}% (Upper Layer Planning...)")
    current_soc_dict = {u.id: u.get_soc() for u in all_units_list}

    pred_wind = raw_wind[i: i + PH_upper]
    pred_solar = np.zeros_like(pred_wind)
    pred_load = (np.mean(raw_wind) * 0.4) + 40e6 + 20e6 * np.sin(
        2 * np.pi * time_series_upper[i: i + PH_upper] / (24 * 3600))
    pred_prices = grid_prices_upper[i: i + PH_upper]

    if len(pred_wind) < PH_upper:
        pad_len = PH_upper - len(pred_wind)
        pred_wind = np.pad(pred_wind, (0, pad_len), 'edge')
        pred_load = np.pad(pred_load, (0, pad_len), 'edge')
        pred_solar = np.pad(pred_solar, (0, pad_len), 'edge')
        pred_prices = np.pad(pred_prices, (0, pad_len), 'edge')

    reference_signals = h_mpc_ems.solve_upper_layer(current_soc_dict, pred_wind, pred_solar, pred_load, pred_prices, dt_upper_s / 3600.0)

    # ========================= 核心修正部分 (开始) =========================
    # main_simulation_hierarchical.py
    # ...
    if not reference_signals:
        print("上层MPC求解失败，本轮采用零功率计划！")
        # 如果上层求解失败，创建一个默认的零功率参考信号，确保内层循环可以执行
        reference_signals = {
            # 修正：使用单元的ID作为字典键
            "slow_asset_dispatch": {uid.id: np.zeros(PH_upper) for uid in h_mpc_ems.slow_assets},
            "fast_asset_net_power_ref": np.zeros(PH_upper)
        }
    # ========================= 核心修正部分 (结束) =========================

    for j in range(steps_per_upper):
        current_soc_dict_fast = {u.id: u.get_soc() for u in h_mpc_ems.fast_assets}
        dispatch_plan = h_mpc_ems.solve_lower_layer(current_soc_dict_fast, reference_signals, dt_lower_s / 3600.0, steps_per_upper)

        # 即使下层求解失败（返回None），也要保证应用零功率并记录数据
        if not dispatch_plan:
            dispatch_plan = {f"{unit.id}_power": 0 for unit in all_units_list}

        for unit in all_units_list:
            power_cmd = dispatch_plan.get(f"{unit.id}_power", 0)
            if power_cmd > 0:
                unit.discharge(power_cmd, dt_lower_s)
            elif power_cmd < 0:
                unit.charge(abs(power_cmd), dt_lower_s)
            else:
                unit.idle_loss(dt_lower_s)

        total_power = 0
        for unit in all_units_list:
            key = unit.id.split('_')[0]
            power = dispatch_plan.get(f"{unit.id}_power", 0)
            results[f'p_{key}'].append(power)
            results[f'soc_{key}'].append(unit.get_soc())
            total_power += power

        net_load_step = load_power_fine[lower_step_counter] - (raw_wind_fine[lower_step_counter] + solar_power_fine[lower_step_counter])
        grid_power_step = net_load_step - total_power
        results["p_grid_exchange"].append(grid_power_step)
        results["p_hess_total"].append(total_power)
        lower_step_counter += 1

print("--- 仿真结束 ---")

# (结果可视化部分与之前一致，此处省略)
fig, axs = plt.subplots(4, 1, figsize=(15, 18), sharex=True)
time_h_lower = time_series_lower / 3600
net_load_fine = load_power_fine - (raw_wind_fine + solar_power_fine)
axs[0].plot(time_h_lower, net_load_fine / 1e6, label="本地净负荷 (MW)", color='lightblue')
axs[0].set_title("净负荷、电价与电网交互", fontsize=16)
axs[0].set_ylabel("功率 (MW)")
axs[0].grid(True)
ax_price = axs[0].twinx()
ax_price.plot(time_series_upper / 3600, grid_prices_upper, 'r--', label="电价 (元/MWh)", alpha=0.7)
ax_price.set_ylabel("电价 (元/MWh)")
axs[0].legend(loc='upper left')
ax_price.legend(loc='upper right')
axs[1].bar(time_h_lower, np.array(results["p_grid_exchange"]) / 1e6, width=1/3600, label="电网交互功率 (购电为负)")
axs[1].plot(time_h_lower, np.array(results["p_hess_total"]) / 1e6, 'k-', label="HESS总功率 (放电为正)")
axs[1].set_title("电网交互与HESS总出力", fontsize=16)
axs[1].set_ylabel("功率 (MW)")
axs[1].legend()
axs[1].grid(True)
for unit in all_units_list:
    unit_type = unit.id.split('_')[0]
    axs[2].plot(time_h_lower, np.array(results[f"p_{unit_type}"]) / 1e6, label=f"{unit_type} 功率")
axs[2].set_title("各储能单元出力", fontsize=16)
axs[2].set_ylabel("功率 (MW)")
axs[2].legend()
axs[2].grid(True)
for unit in all_units_list:
    unit_type = unit.id.split('_')[0]
    axs[3].plot(time_h_lower, results[f"soc_{unit_type}"], label=f"{unit_type} SOC")
axs[3].set_title("各储能单元SOC", fontsize=16)
axs[3].set_ylabel("SOC")
axs[3].set_xlabel("时间 (小时)")
axs[3].legend()
axs[3].grid(True)
plt.tight_layout()
plt.savefig("hmpc_dispatch_results_8_units.png")
plt.show()