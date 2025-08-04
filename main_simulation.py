# file: PythonProject/main_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from hess_system import HybridEnergyStorageSystem
from ems import HierarchicalEMS

# --- 导入您所有的储能模型 ---
# (请确保文件路径正确，并且所有模型都已修改为继承基类)
from high_power_density_group.flywheel_simulation import FlywheelModel
from high_power_density_group.supercapacitor_simulation import Supercapacitor
from high_power_density_group.Superconducting_magnetic_energy_storage_simulation import SuperconductingMagneticEnergyStorage
from Medium_power_density_group.lithium_ion_battery_simulation import LithiumIonBattery
from Medium_power_density_group.sodium_ion_battery_simulation import SodiumIonBattery
from Medium_power_density_group.lead_batteries_simulation import LeadAcidBattery
from Medium_power_density_group.flow_battery_simulation import FlowBattery
from low_power_density_group.pumped_storage_simulation import PumpedHydroStorage
from low_power_density_group.thermal_storage import ThermalEnergyStorage
from low_power_density_group.caes_system import DiabaticCAES
from low_power_density_group.hydrogen_storage import HydrogenStorage


def generate_wind_power_data(duration_s, dt_s):
    """生成包含多频段波动的模拟风电数据"""
    num_points = int(duration_s / dt_s)
    time = np.arange(num_points) * dt_s

    # 基础功率 + 小时级 + 分钟级 + 秒级波动
    base_power = 50e6  # 50MW
    low_freq = 20e6 * np.sin(2 * np.pi * time / (3600 * 0.5))
    medium_freq = 10e6 * np.sin(2 * np.pi * time / 600)
    high_freq = 5e6 * (np.random.rand(num_points) - 0.5)

    raw_wind_power = base_power + low_freq + medium_freq + high_freq
    raw_wind_power = np.maximum(0, raw_wind_power)

    # 目标：将风电平滑为10分钟移动平均值
    window_size = int(600 / dt_s)
    if window_size < 1: window_size = 1
    target_grid_power = np.convolve(raw_wind_power, np.ones(window_size) / window_size, mode='same')

    return time, raw_wind_power, target_grid_power


# 1. 初始化HESS系统
hess = HybridEnergyStorageSystem()

# 2. 配置并添加所有储能单元 (这是关键的参数配置步骤)
# --- 快响应组 (高功率、低容量，用于高频) ---
hess.add_unit(Supercapacitor(ess_id="sc_01", initial_soc=0.5, rated_current=5000, initial_capacitance=5000),
              group='fast')
hess.add_unit(
    SuperconductingMagneticEnergyStorage(ess_id="smes_01", initial_soc=0.5, inductance=5, pcs_rated_power=5e6),
    group='fast')

# --- 中响应组 (中等功率、中等容量，用于中频) ---
hess.add_unit(FlywheelModel(ess_id="fw_01", initial_soc=0.5, rated_power=2e6, mass=2000), group='medium')
hess.add_unit(LithiumIonBattery(ess_id="li_ion_01", initial_soc=0.5, nominal_capacity_ah=20000, max_c_rate_discharge=2),
              group='medium')
# ... 您可以根据需要取消注释以添加更多储能
# hess.add_unit(SodiumIonBattery(ess_id="na_ion_01", initial_soc=0.5, nominal_capacity_ah=20000), group='medium')
# hess.add_unit(LeadAcidBattery(ess_id="lead_acid_01", initial_soc=0.8, nominal_capacity_ah=30000), group='medium')

# --- 长时组 (低功率、高容量，本次仿真备用，不参与高频平抑) ---
# (为简化本次仿真，暂时不实例化它们)
# hess.add_unit(FlowBattery(ess_id="flow_batt_01", initial_soc=0.5), group='long')
# hess.add_unit(PumpedHydroStorage(ess_id="phs_01", initial_soc=0.5), group='long')
# ...

# 3. 准备数据和EMS
duration_s = 3600  # 模拟1小时
dt_s = 1  # 时间步长1秒
time_series, raw_wind, target_grid = generate_wind_power_data(duration_s, dt_s)

ems = HierarchicalEMS(hess)

# 4. 仿真主循环
actual_grid_power_history = []
hess_power_history = []

print("--- 开始HESS仿真 ---")

# ========================== 错误修正点 ==========================
# 预先计算好整个波动序列，以传递给新的decompose_signal方法
total_fluctuation_series = raw_wind - target_grid
# ==============================================================

for i in range(len(time_series)):
    # EMS分解信号 (新的、正确的调用方式)
    p_for_fast, p_for_medium = ems.decompose_signal(total_fluctuation_series, i)

    # EMS下发指令, HESS响应
    # 储能需要吸收(充电)的功率 = -波动。例如风电多了，波动为正，储能吸收，功率为负。
    hess_power_fast = ems.dispatch_power_to_group('fast', -p_for_fast, dt_s)
    hess_power_medium = ems.dispatch_power_to_group('medium', -p_for_medium, dt_s)

    # HESS总出力
    p_hess_total = hess_power_fast + hess_power_medium

    # 实际并网功率 = 原始风电 + HESS总出力
    p_actual_grid = raw_wind[i] + p_hess_total

    # 记录历史数据
    actual_grid_power_history.append(p_actual_grid)
    hess_power_history.append(p_hess_total)

    if i % 300 == 0:  # 每300秒打印一次进度
        print(f"仿真进度: {i / duration_s * 100:.1f}%")

print("--- 仿真结束 ---")

# 5. 结果可视化
plt.figure(figsize=(15, 8))
plt.plot(time_series / 60, raw_wind / 1e6, label="原始风电功率", color='lightblue', alpha=0.8)
plt.plot(time_series / 60, target_grid / 1e6, 'k--', label="目标并网功率", lw=2)
plt.plot(time_series / 60, np.array(actual_grid_power_history) / 1e6, label="HESS平抑后功率", color='darkgreen', lw=2)
plt.title("HESS对风电波动的平抑效果", fontsize=16)
plt.xlabel("时间 (分钟)", fontsize=12)
plt.ylabel("功率 (MW)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()