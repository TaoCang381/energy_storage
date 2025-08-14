# file: PythonProject/low power density group/pumped_storage_simulation.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit

# --- 物理常数 ---
WATER_DENSITY_KG_M3 = 1000
GRAVITY_G = 9.81


class PumpedHydroStorage(EnergyStorageUnit):
    """
    抽水蓄能 (PHS) 模型 (HESS集成版 - 简化版)
    模型基于水力学和重力势能，核心状态为水量。已移除爬坡约束以简化模型。
    """

    def __init__(self,
                 ess_id="phs_01",
                 initial_soc=0.5,
                 initial_soh=1.0,
                 # --- 核心物理参数 ---
                 upper_reservoir_volume_m3=1.835e7,
                 effective_head_m=400,

                 # --- 机组性能参数 ---
                 turbine_rated_power_w=300e6,
                 pump_rated_power_w=300e6,
                 turbine_efficiency=0.9,
                 pump_efficiency=0.9,

                 # --- 运行限制 ---
                 min_power_ratio=0.2,
                 soc_upper_limit=0.98,
                 soc_lower_limit=0.02,
                 cost_per_kwh=0.005
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)
        self.soh = 1.0

        # --- 规格参数 ---
        self.V_ur_max = upper_reservoir_volume_m3
        self.V_ur_min = self.V_ur_max * soc_lower_limit
        self.h_eff = effective_head_m
        self.P_gen_rated = turbine_rated_power_w
        self.P_pump_rated = pump_rated_power_w
        self.eta_gen = turbine_efficiency
        self.eta_pump = pump_efficiency
        self.soc_max = soc_upper_limit
        self.soc_min = soc_lower_limit
        self.cost_per_kwh = cost_per_kwh

        self.P_gen_min = self.P_gen_rated * min_power_ratio
        self.P_pump_min = self.P_pump_rated * min_power_ratio

        # --- 核心状态变量：上水库水量 V_ur ---
        self.V_ur_m3 = self.V_ur_max * self.soc

        self.volume_history = []

    def get_soc(self):
        """根据水量计算并更新SOC"""
        if self.V_ur_max > 0:
            self.soc = self.V_ur_m3 / self.V_ur_max
        else:
            self.soc = 0
        return self.soc

    def _power_to_flow(self, power, is_charging):
        """根据功率计算流量"""
        denominator = WATER_DENSITY_KG_M3 * GRAVITY_G * self.h_eff
        if denominator == 0: return 0
        if is_charging:
            return (power * self.eta_pump) / denominator
        else:
            return power / (denominator * self.eta_gen)

    def _update_volume(self, flow_rate_m3s, time_s, is_charging):
        """根据流量更新水量"""
        delta_volume = flow_rate_m3s * time_s
        if is_charging:
            self.V_ur_m3 += delta_volume
        else:
            self.V_ur_m3 -= delta_volume

        self.V_ur_m3 = max(self.V_ur_min, min(self.V_ur_m3, self.V_ur_max))

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (W) - 已移除爬坡约束"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_pump_rated

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (W) - 已移除爬坡约束"""
        if self.get_soc() <= self.soc_min: return 0
        return self.P_gen_rated

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (抽水)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec < self.P_pump_min:
            self.idle_loss(time_s)
            return

        self.state = 'charging'

        flow_rate = self._power_to_flow(power_elec, is_charging=True)
        self._update_volume(flow_rate, time_s, is_charging=True)

        self._record_history_phs(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec < self.P_gen_min:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'

        flow_rate = self._power_to_flow(power_elec, is_charging=False)
        self._update_volume(flow_rate, time_s, is_charging=False)

        self._record_history_phs(time_s, -power_elec)

    def idle_loss(self, time_s):
        self.state = 'idle'
        self._record_history_phs(time_s, 0)

    def _record_history_phs(self, time_delta, power):
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.volume_history.append(self.V_ur_m3)


# --- 单元测试用的示例函数 ---
def simulate_phs_test():
    phs = PumpedHydroStorage(initial_soc=0.5)

    max_energy_mwh = phs.V_ur_max * WATER_DENSITY_KG_M3 * GRAVITY_G * phs.h_eff / 3.6e9
    print(f"PHS Initialized. Rated Power: {phs.P_gen_rated / 1e6} MW, Usable Energy: {max_energy_mwh:.2f} MWh")
    print(f"Initial SOC: {phs.get_soc():.3f}\n")

    charge_power = 200e6
    charge_time = 4 * 3600
    print(f"--- Charging with {charge_power / 1e6} MW for {charge_time / 3600:.1f}h ---")
    phs.charge(charge_power, charge_time)
    print(f"After charging, SOC: {phs.get_soc():.3f}\n")

    discharge_power = 300e6
    discharge_time = 2 * 3600
    print(f"--- Discharging with {discharge_power / 1e6} MW for {discharge_time / 3600:.1f}h ---")
    phs.discharge(discharge_power, discharge_time)
    print(f"After discharging, SOC: {phs.get_soc():.3f}\n")


if __name__ == "__main__":
    simulate_phs_test()