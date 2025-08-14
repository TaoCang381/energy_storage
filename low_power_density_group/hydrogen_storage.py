# file: PythonProject/low power density group/hydrogen_storage.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit

# --- 物理常数 ---
LHV_H2_J_PER_KG = 120e6  # 氢气低热值 (J/kg)
LHV_H2_KWH_PER_KG = 33.3  # 氢气低热值 (kWh/kg)


class HydrogenStorage(EnergyStorageUnit):
    """
    氢储能 (HES) 模型 (HESS集成版 - 严格对应论文公式)
    包含电解槽、储氢罐、燃料电池三个核心部分，并考虑热电联产。
    """

    def __init__(self,
                 ess_id="hydrogen_storage_01",
                 initial_soc=0.5,
                 initial_soh=1.0,
                 # ========================== 合理化参数配置 (开始) ==========================
                 # --- 制氢系统 (充电) ---
                 electrolyzer_rated_power_w=50e6,  # 电解槽额定功率 (W), e.g., 50 MW
                 electrolyzer_efficiency_kwh_kg=50,  # 制氢电耗 (kWh/kg), e.g., 50度电制1公斤氢气

                 # --- 储氢系统 ---
                 tank_max_capacity_kg=100000,  # 储氢罐最大容量 (kg), e.g., 100吨
                 compressor_power_ratio=0.08,  # 压缩机功率占电解槽功率的比例, e.g., 8%

                 # --- 发电系统 (放电) ---
                 fuel_cell_rated_power_w=40e6,  # 燃料电池额定电功率 (W), e.g., 40 MW
                 fc_elec_efficiency=0.55,  # 燃料电池发电效率 (%)
                 fc_heat_recovery_efficiency=0.35,  # 燃料电池余热回收效率 (%)

                 # --- 运行限制 ---
                 soc_upper_limit=0.95,
                 soc_lower_limit=0.05,
                 # ========================== 合理化参数配置 (结束) ==========================
                 cost_per_kwh_cycle=0.001  # 寿命成本主要体现在启停和运行时长
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)

        # --- 核心组件参数 ---
        self.P_ely_rated = electrolyzer_rated_power_w
        self.eta_ely_kwh_kg = electrolyzer_efficiency_kwh_kg
        self.M_tank_max = tank_max_capacity_kg
        self.compressor_ratio = compressor_power_ratio
        self.P_fc_rated = fuel_cell_rated_power_w
        self.eta_fc_elec = fc_elec_efficiency
        self.eta_fc_heat = fc_heat_recovery_efficiency
        self.soc_max = soc_upper_limit
        self.soc_min = soc_lower_limit
        self.cost_per_kwh_cycle = cost_per_kwh_cycle

        # --- 实时工作参数 (受SOH影响) ---
        self.current_fc_efficiency = self.eta_fc_elec * self.soh
        self.current_ely_efficiency = self.eta_ely_kwh_kg / self.soh

        # --- 核心状态变量：储氢质量 M_H2 ---
        self.M_H2_kg = self.M_tank_max * self.soc

        self.mass_history = []
        self.heat_power_history = []  # 新增：记录热功率

    def get_soc(self):
        """根据氢气质量计算并更新SOC"""
        if self.M_tank_max > 0:
            self.soc = self.M_H2_kg / self.M_tank_max
        else:
            self.soc = 0
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电(制氢)总功率 (W)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_ely_rated

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_min: return 0
        return self.P_fc_rated

    def charge(self, power_elec, time_s):
        """按指定总电功率充电 (制氢 + 压缩)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return
        self.state = 'charging'

        # 1. 计算压缩机功耗
        compressor_power = power_elec * self.compressor_ratio
        # 2. 实际用于电解槽的功率
        power_to_ely = power_elec - compressor_power
        if power_to_ely <= 0:
            self.idle_loss(time_s)
            return

        # 3. 计算产氢速率
        power_to_ely_kw = power_to_ely / 1000
        m_dot_ely = power_to_ely_kw / self.current_ely_efficiency  # kg/h

        # 4. 更新储氢质量 (动态方程)
        time_h = time_s / 3600.0
        self.M_H2_kg += m_dot_ely * time_h
        self.M_H2_kg = min(self.M_H2_kg, self.M_tank_max * self.soc_max)

        self._record_history_hes(time_s, power_elec, 0)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，并计算伴生的热功率"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return (0, 0)  # 返回电功率和热功率
        self.state = 'discharging'

        # 1. 计算耗氢速率
        power_elec_kw = power_elec / 1000
        time_h = time_s / 3600.0
        m_dot_fc = (power_elec_kw) / (LHV_H2_KWH_PER_KG * self.current_fc_efficiency)  # kg/h

        # 2. 计算总耗氢量并检查余量
        mass_consumed_kg = m_dot_fc * time_h
        if mass_consumed_kg > self.M_H2_kg:
            mass_consumed_kg = self.M_H2_kg
            # 根据实际可用氢气反算实际发电量
            power_elec = (mass_consumed_kg / time_h) * LHV_H2_KWH_PER_KG * self.current_fc_efficiency * 1000

        # 3. 更新储氢质量 (动态方程)
        self.M_H2_kg -= mass_consumed_kg
        self.M_H2_kg = max(self.M_H2_kg, self.M_tank_max * self.soc_min)

        # 4. 计算伴生的热功率 (热电联产)
        power_heat = power_elec * (self.eta_fc_heat / self.eta_fc_elec)

        self._record_history_hes(time_s, -power_elec, power_heat)
        return power_elec, power_heat

    def idle_loss(self, time_s):
        """模拟闲置时的氢气泄漏"""
        self.state = 'idle'
        daily_loss_ratio = 0.0001
        loss_per_second_kg = (self.M_tank_max * daily_loss_ratio) / (24 * 3600)
        self.M_H2_kg -= loss_per_second_kg * time_s
        self._record_history_hes(time_s, 0, 0)

    def _record_history_hes(self, time_delta, power, heat_power):
        """记录HES特有的历史数据"""
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.mass_history.append(self.M_H2_kg)
        self.heat_power_history.append(heat_power)


# --- 单元测试用的示例函数 ---
def simulate_hydrogen_test():
    h2_storage = HydrogenStorage(initial_soc=0.5)

    max_energy_gwh = (h2_storage.M_tank_max * LHV_H2_KWH_PER_KG) / 1e6
    print(f"HES Initialized. Ely Power: {h2_storage.P_ely_rated / 1e6} MW, FC Power: {h2_storage.P_fc_rated / 1e6} MW")
    print(f"Max Energy Capacity (equivalent): {max_energy_gwh:.2f} GWh")
    print(f"Initial SOC: {h2_storage.get_soc():.3f}\n")

    # 模拟以50MW功率连续制氢24小时
    charge_power = 50e6
    charge_time = 24 * 3600
    print(f"--- Charging with {charge_power / 1e6} MW for {charge_time / 3600:.1f}h ---")
    h2_storage.charge(charge_power, charge_time)
    print(f"After charging, SOC: {h2_storage.get_soc():.3f}, H2 Mass: {h2_storage.M_H2_kg / 1000:.2f} tons\n")

    # 模拟以40MW电功率连续发电12小时
    discharge_power = 40e6
    discharge_time = 12 * 3600
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for {discharge_time / 3600:.1f}h ---")
    actual_p_elec, actual_p_heat = h2_storage.discharge(discharge_power, discharge_time)
    print(f"After discharging, SOC: {h2_storage.get_soc():.3f}, H2 Mass: {h2_storage.M_H2_kg / 1000:.2f} tons")
    print(
        f"  > Actual Elec Power: {actual_p_elec / 1e6:.2f} MW, Co-generated Heat Power: {actual_p_heat / 1e6:.2f} MW_th\n")


if __name__ == "__main__":
    simulate_hydrogen_test()