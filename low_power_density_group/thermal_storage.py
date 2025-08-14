# file: PythonProject/low power density group/thermal_storage.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit


class ThermalEnergyStorage(EnergyStorageUnit):
    """
    热储能 (TES) 模型 (HESS集成版 - 严格对应论文公式)
    模型基于显热储能物理公式，包含非对称转换效率和散热损失。
    """

    def __init__(self,
                 ess_id="thermal_storage_01",
                 initial_soc=0.5,
                 initial_soh=1.0,  # TES的SOH几乎不衰减
                 # ========================== 合理化参数配置 (开始) ==========================
                 # --- 储热罐物理参数 (配置为 100MW / 800MWh_e 系统) ---
                 storage_medium_mass_kg=8e7,  # 储热介质质量 (kg), e.g., 8万吨熔盐
                 specific_heat_capacity_j_kgk=1500,  # 介质比热容 (J/(kg·K)), e.g., 熔盐

                 # --- 温度与运行限制 ---
                 max_temp_k=838.15,  # 最高工作温度 (K), e.g., 565°C
                 min_temp_k=563.15,  # 最低工作温度 (K), e.g., 290°C

                 # --- 功率转换系统(PCS)参数 ---
                 heater_rated_power_w=110e6,  # 电加热器额定功率 (W), e.g., 110MW
                 heat_engine_rated_power_w=100e6,  # 热机额定输出功率 (W), e.g., 100MW

                 # --- 效率与损耗 ---
                 elec_to_heat_efficiency=0.98,  # 电->热 转换效率
                 heat_to_elec_efficiency=0.42,  # 热->电 转换效率
                 heat_loss_rate_percent_hr=0.04,  # 每小时散热损失率 (% of max stored energy)

                 soc_upper_limit=1.0,
                 soc_lower_limit=0.0,
                 # ========================== 合理化参数配置 (结束) ==========================
                 cost_per_kwh=0.01  # 运维成本较低
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)
        self.soh = 1.0

        # --- 物理与性能参数 ---
        self.m = storage_medium_mass_kg
        self.c = specific_heat_capacity_j_kgk
        self.T_max = max_temp_k
        self.T_min = min_temp_k
        self.P_heater_rated = heater_rated_power_w
        self.P_gen_rated = heat_engine_rated_power_w
        self.eta_e2h = elec_to_heat_efficiency
        self.eta_h2e = heat_to_elec_efficiency
        self.theta_loss = (heat_loss_rate_percent_hr / 100) / 3600.0  # 转换为每秒损失率
        self.soc_max = soc_upper_limit
        self.soc_min = soc_lower_limit
        self.cost_per_kwh = cost_per_kwh

        # --- 核心状态变量：储存的热量 H_tes ---
        self.H_tes_max_J = self.m * self.c * (self.T_max - self.T_min)
        self.H_tes_J = self.H_tes_max_J * self.soc

        self.energy_history = []
        self.temp_history = []

    def get_soc(self):
        """根据储存的热量计算并更新SOC"""
        if self.H_tes_max_J > 0:
            self.soc = self.H_tes_J / self.H_tes_max_J
        else:
            self.soc = 0
        return self.soc

    def get_current_temp_k(self):
        """根据热量计算当前温度"""
        return self.T_min + self.H_tes_J / (self.m * self.c)

    def get_available_charge_power(self):
        """获取当前可用的充电(加热)功率 (W_e)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_heater_rated

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W_e)"""
        if self.get_soc() <= self.soc_min: return 0
        return self.P_gen_rated

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (加热)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return
        self.state = 'charging'

        # 1. 计算注入的热功率
        power_heat_in = power_elec * self.eta_e2h

        # 2. 更新储存的热量 (动态方程)
        delta_heat = power_heat_in * time_s
        self.H_tes_J += delta_heat

        # 3. 应用储热容量约束
        self.H_tes_J = min(self.H_tes_J, self.H_tes_max_J * self.soc_max)

        self._record_history_tes(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，并计算所需热功率"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return
        self.state = 'discharging'

        # 1. 计算需要从储罐中提取的热功率
        power_heat_out = power_elec / self.eta_h2e

        # 2. 更新储存的热量 (动态方程)
        delta_heat = power_heat_out * time_s
        self.H_tes_J -= delta_heat

        # 3. 应用储热容量约束
        self.H_tes_J = max(self.H_tes_J, self.H_tes_max_J * self.soc_min)

        self._record_history_tes(time_s, -power_elec)

    def idle_loss(self, time_s):
        """模拟闲置时的散热损失"""
        self.state = 'idle'

        # 损失的热量 = 最大储热量 * 损失率 * 时间
        lost_heat = self.H_tes_max_J * self.theta_loss * time_s
        self.H_tes_J -= lost_heat
        self.H_tes_J = max(0, self.H_tes_J)  # 确保热量不为负

        self._record_history_tes(time_s, 0)

    def _record_history_tes(self, time_delta, power):
        """记录TES特有的历史数据"""
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.energy_history.append(self.H_tes_J)
        self.temp_history.append(self.get_current_temp_k())


# --- 单元测试用的示例函数 ---
def simulate_tes_test():
    tes = ThermalEnergyStorage(initial_soc=0.2)

    max_energy_gwh_th = tes.H_tes_max_J / 3.6e12
    max_energy_gwh_e = max_energy_gwh_th * tes.eta_h2e
    print(
        f"TES Initialized. Gen Power: {tes.P_gen_rated / 1e6} MW, Usable Elec Energy: {max_energy_gwh_e * 1000:.2f} MWh")
    print(f"Initial SOC: {tes.get_soc():.3f}\n")

    # 模拟以110MW功率连续充电6小时
    charge_power = 110e6
    charge_time = 6 * 3600
    print(f"--- Charging with {charge_power / 1e6} MW for {charge_time / 3600:.1f}h ---")
    tes.charge(charge_power, charge_time)
    print(f"After charging, SOC: {tes.get_soc():.3f}, Temp: {tes.get_current_temp_k() - 273.15:.1f}°C\n")

    # 模拟闲置2小时
    idle_time = 2 * 3600
    print(f"--- Idling for {idle_time / 3600:.1f}h ---")
    tes.idle_loss(idle_time)
    print(f"After idling, SOC: {tes.get_soc():.3f}, Temp: {tes.get_current_temp_k() - 273.15:.1f}°C\n")

    # 模拟以100MW电功率连续发电4小时
    discharge_power = 100e6
    discharge_time = 4 * 3600
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for {discharge_time / 3600:.1f}h ---")
    tes.discharge(discharge_power, discharge_time)
    print(f"After discharging, SOC: {tes.get_soc():.3f}, Temp: {tes.get_current_temp_k() - 273.15:.1f}°C\n")


if __name__ == "__main__":
    simulate_tes_test()