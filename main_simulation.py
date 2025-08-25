# file: PythonProject/main_simulation.py (V5.0 - 八储能完整系统版)

import numpy as np
import matplotlib.pyplot as plt
from hess_system import HybridEnergyStorageSystem
from mpc_ems import MPCEnergyManagementSystem

# 导入所有8种储能模型
from high_power_density_group.flywheel_simulation import FlywheelModel
from high_power_density_group.supercapacitor_simulation import Supercapacitor
from high_power_density_group.Superconducting_magnetic_energy_storage_simulation import SuperconductingMagneticEnergyStorage
from Medium_power_density_group.electrochemical_energy_storage import ElectrochemicalEnergyStorage
from low_power_density_group.pumped_storage_simulation import PumpedHydroStorage
from low_power_density_group.hydrogen_storage import HydrogenStorage
from low_power_density_group.thermal_storage import ThermalEnergyStorage
from low_power_density_group.caes_system import DiabaticCAES

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 数据生成函数 (保持不变) ---
def generate_wind_power_data(duration_s, dt_s):
    # ... (此处省略，保持你原来的函数不变)
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
    # ... (此处省略，保持你原来的函数不变)
    prices = np.ones_like(time_series)
    hours = (time_series / 3600) % 24
    prices[((hours >= 8) & (hours < 12)) | ((hours >= 18) & (hours < 22))] = 800
    prices[((hours >= 12) & (hours < 18)) | ((hours >= 22) & (hours < 24))] = 500
    prices[(hours >= 0) & (hours < 8)] = 200
    return prices

# --- 1. 初始化HESS系统 (加载全部8种储能) ---
hess = HybridEnergyStorageSystem()
print("Initializing HESS with all 8 storage units...")

# 快速响应组
hess.add_unit(FlywheelModel(ess_id="fw_01", cost_per_kwh=0.05), group='fast')
hess.add_unit(Supercapacitor(ess_id="sc_01", cost_per_kwh=0.02), group='fast')
hess.add_unit(SuperconductingMagneticEnergyStorage(ess_id="smes_01", cost_per_kwh=0.02), group='fast')
# 中速响应组
hess.add_unit(ElectrochemicalEnergyStorage(ess_id="ees_01", cost_per_kwh=0.08), group='medium')
# 慢速/长时储能组
hess.add_unit(PumpedHydroStorage(ess_id="phs_01", cost_per_kwh=0.005), group='long')
hess.add_unit(HydrogenStorage(ess_id="hes_01", cost_per_kwh=0.1), group='long')
hess.add_unit(ThermalEnergyStorage(ess_id="tes_01", cost_per_kwh=0.01), group='long')
hess.add_unit(DiabaticCAES(ess_id="caes_01", cost_per_kwh_fuel=0.3), group='long')

print("HESS setup complete.")

# --- 2. 准备仿真数据和MPC控制器 ---
duration_s = 3600 * 24
dt_s = 60 * 15
time_steps = int(duration_s / dt_s)
time_series, raw_wind, _ = generate_wind_power_data(duration_s, dt_s)
solar_power = np.zeros_like(raw_wind)
load_power = (np.mean(raw_wind) * 0.4) + 40e6 + 20e6 * np.sin(2 * np.pi * time_series / (24 * 3600))
grid_prices = generate_price_signal(time_series)
prediction_horizon = 4
mpc_ems = MPCEnergyManagementSystem(hess, prediction_horizon)

# --- 3. 仿真主循环 ---
all_units_list = list(hess.all_units.values())
results = { "p_grid_exchange": [], "p_hess_total": [] }
for unit in all_units_list:
    unit_type = unit.id.split('_')[0]
    results[f"p_{unit_type}"] = []
    results[f"soc_{unit_type}"] = []

print("--- 开始MPC经济调度仿真 ---")
for i in range(time_steps):
    current_soc = {unit.id: unit.get_soc() for unit in all_units_list}

    end_idx = i + prediction_horizon
    # ... (准备预测数据，此处省略)
    pred_wind = raw_wind[i:end_idx]
    pred_solar = solar_power[i:end_idx]
    pred_load = load_power[i:end_idx]
    pred_prices = grid_prices[i:end_idx]
    if len(pred_wind) < prediction_horizon:
        pad_width = prediction_horizon - len(pred_wind)
        pred_wind = np.pad(pred_wind, (0, pad_width), 'edge')
        pred_solar = np.pad(pred_solar, (0, pad_width), 'edge')
        pred_load = np.pad(pred_load, (0, pad_width), 'edge')
        pred_prices = np.pad(pred_prices, (0, pad_width), 'edge')

    dispatch_plan = mpc_ems.solve(current_soc, pred_wind, pred_solar, pred_load, pred_prices)

    if dispatch_plan:
        for unit in all_units_list:
            power_cmd = dispatch_plan.get(f"{unit.id}_power", 0)
            if power_cmd < 0: unit.charge(abs(power_cmd), dt_s)
            else: unit.discharge(power_cmd, dt_s)
    else:
        for unit in all_units_list: unit.idle_loss(dt_s)

    # 记录结果
    p_hess_total = 0
    for unit in all_units_list:
        unit_type = unit.id.split('_')[0]
        power_key = f"p_{unit_type}"
        soc_key = f"soc_{unit_type}"
        dispatched_power = dispatch_plan.get(f"{unit.id}_power", 0) if dispatch_plan else 0
        results[power_key].append(dispatched_power)
        results[soc_key].append(unit.get_soc())
        p_hess_total += dispatched_power

    results["p_hess_total"].append(p_hess_total)
    results["p_grid_exchange"].append(dispatch_plan.get("grid_power")[0] if dispatch_plan else 0)

    if i % 4 == 0:
        print(f"仿真进度: {(i / time_steps) * 100:.1f}%")

print("--- 仿真结束 ---")

# --- 4. 结果可视化 ---
fig, axs = plt.subplots(4, 1, figsize=(15, 18), sharex=True)
time_h = time_series / 3600

# (可视化代码与五储能版本类似，此处省略以节约篇幅，请使用五储能版本中的可视化代码，并自行添加新储能的绘图)
# 图1: 净负荷与电价
axs[0].plot(time_h, (raw_wind + solar_power - load_power) / 1e6, label="本地净负荷 (MW)", color='lightblue')
axs[0].set_title("净负荷、电价与电网交互", fontsize=16)
axs[0].set_ylabel("功率 (MW)")
axs[0].grid(True)
ax_price = axs[0].twinx()
ax_price.plot(time_h, grid_prices, 'r--', label="电价 (元/MWh)", alpha=0.7)
ax_price.set_ylabel("电价 (元/MWh)")
axs[0].legend(loc='upper left')
ax_price.legend(loc='upper right')

# 图2: 电网交互与HESS总功率
axs[1].bar(time_h, np.array(results["p_grid_exchange"]) / 1e6, width=0.05, label="电网交互功率 (购电为正)")
axs[1].plot(time_h, np.array(results["p_hess_total"]) / 1e6, 'k-', label="HESS总功率 (放电为正)")
axs[1].set_title("电网交互与HESS总出力", fontsize=16)
axs[1].set_ylabel("功率 (MW)")
axs[1].legend()
axs[1].grid(True)

# 图3: 各储能单元出力
for unit in all_units_list:
    unit_type = unit.id.split('_')[0]
    axs[2].plot(time_h, np.array(results[f"p_{unit_type}"]) / 1e6, label=f"{unit_type} 功率")
axs[2].set_title("各储能单元出力", fontsize=16)
axs[2].set_ylabel("功率 (MW)")
axs[2].legend()
axs[2].grid(True)

# 图4: 各储能单元SOC
for unit in all_units_list:
    unit_type = unit.id.split('_')[0]
    axs[3].plot(time_h, results[f"soc_{unit_type}"], label=f"{unit_type} SOC")
axs[3].set_title("各储能单元SOC", fontsize=16)
axs[3].set_ylabel("SOC")
axs[3].set_xlabel("时间 (小时)")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.savefig("economic_dispatch_results_8_units.png")
plt.show()