# file: low_power_density_group/hydrogen_storage.py (修正并简化SOH版 V2.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_storage_model import BaseStorageModel

# --- 物理常数 ---
LHV_H2_KWH_PER_KG = 33.3  # 氢气低热值 (kWh/kg)


class HydrogenStorage(BaseStorageModel):
    """
    氢储能 (HES) 模型 (SOH简化版)
    - 移除了SOH衰减模型，严格对应您docx中的简化公式。
    - 参数已按基准表进行校准和反向推算。
    - 修正了因移除SOH导致的AttributeError。
    """

    def __init__(self,
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 electrolyzer_rated_power_mw=50.0,
                 fuel_cell_rated_power_mw=50.0,
                 rated_capacity_mwh=2000.0,
                 electrolyzer_efficiency_kwh_kg=48,
                 fc_elec_efficiency=0.55,
                 om_cost_per_mwh=50,
                 soc_upper_limit=0.95,
                 soc_lower_limit=0.05,
                 compressor_power_ratio=0.08,
                 fc_heat_recovery_efficiency=0.35
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = fuel_cell_rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh

        # 核心计算：往返效率 (Round-trip efficiency)
        kg_per_kwh_ely = 1 / electrolyzer_efficiency_kwh_kg
        kwh_per_kg_fc = LHV_H2_KWH_PER_KG * fc_elec_efficiency
        self.efficiency = kg_per_kwh_ely * kwh_per_kg_fc

        # 3. 保留HES特有的物理参数
        self.P_ely_rated_w = electrolyzer_rated_power_mw * 1e6
        self.eta_ely_kwh_kg = electrolyzer_efficiency_kwh_kg
        self.compressor_ratio = compressor_power_ratio
        self.P_fc_rated_w = self.power_m_w * 1e6
        self.eta_fc_elec = fc_elec_efficiency
        self.eta_fc_heat = fc_heat_recovery_efficiency

        # 核心推算：根据能量公式 E = M_H2 * LHV, 反算储氢罐的最大质量容量
        energy_kwh = self.capacity_mwh * 1000
        if LHV_H2_KWH_PER_KG > 1e-6:
            self.M_tank_max = energy_kwh / LHV_H2_KWH_PER_KG
        else:
            self.M_tank_max = 0

        # 4. 初始化核心状态变量：储氢质量 M_H2 (kg)
        self.M_H2_kg = self.M_tank_max * self.soc

        self.mass_history = []
        self.heat_power_history = []
        self.state = 'idle'

    def update_state(self, dispatch_power_w):
        """
        根据调度指令（单位：W）更新储能状态。
        """
        if dispatch_power_w > 0:
            self.discharge(dispatch_power_w, self.dt_s)
        elif dispatch_power_w < 0:
            self.charge(abs(dispatch_power_w), self.dt_s)
        else:
            self.idle_loss(self.dt_s)

    def get_soc(self):
        """根据氢气质量计算并更新SOC"""
        if self.M_tank_max > 1e-6:
            self.soc = self.M_H2_kg / self.M_tank_max
        else:
            self.soc = self.soc_min
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电(制氢)总功率 (W)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_ely_rated_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_min: return 0
        return self.P_fc_rated_w

    def charge(self, power_elec, time_s):
        """按指定总电功率充电 (制氢 + 压缩)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'

        compressor_power = power_elec * self.compressor_ratio
        power_to_ely = power_elec - compressor_power
        if power_to_ely <= 0:
            self.idle_loss(time_s)
            return

        power_to_ely_kw = power_to_ely / 1000
        # <--- BUG修复：使用固定的效率参数 ---
        m_dot_ely = power_to_ely_kw / self.eta_ely_kwh_kg  # kg/h

        time_h = time_s / 3600.0
        self.M_H2_kg += m_dot_ely * time_h
        self.M_H2_kg = min(self.M_H2_kg, self.M_tank_max * self.soc_max)

        self.heat_power_history.append(0)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，并计算伴生的热功率"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'
        power_elec_kw = power_elec / 1000
        time_h = time_s / 3600.0

        # <--- BUG修复：使用固定的效率参数 ---
        m_dot_fc = power_elec_kw / (LHV_H2_KWH_PER_KG * self.eta_fc_elec)

        mass_consumed_kg = m_dot_fc * time_h

        if mass_consumed_kg > (self.M_H2_kg - self.M_tank_max * self.soc_min):
            available_mass = self.M_H2_kg - self.M_tank_max * self.soc_min
            if time_h > 0:
                m_dot_fc = available_mass / time_h
                power_elec_kw = m_dot_fc * LHV_H2_KWH_PER_KG * self.eta_fc_elec
                power_elec = power_elec_kw * 1000
            else:
                power_elec = 0
            mass_consumed_kg = available_mass

        self.M_H2_kg -= mass_consumed_kg
        self.M_H2_kg = max(self.M_H2_kg, self.M_tank_max * self.soc_min)

        power_heat = power_elec * (self.eta_fc_heat / self.eta_fc_elec)
        self.heat_power_history.append(power_heat)

    def idle_loss(self, time_s):
        """模拟闲置时的氢气泄漏"""
        self.state = 'idle'
        daily_loss_ratio = 0.0001
        loss_per_second_kg = (self.M_tank_max * daily_loss_ratio) / (24 * 3600)
        self.M_H2_kg -= loss_per_second_kg * time_s
        self.M_H2_kg = max(self.M_H2_kg, self.M_tank_max * self.soc_min)
        self.heat_power_history.append(0)


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    h2_storage = HydrogenStorage(id='hes_test', dt_s=3600, initial_soc=0.5)

    max_energy_gwh = (h2_storage.capacity_mwh) / 1000
    print(f"HES Initialized. Ely Power: {h2_storage.P_ely_rated_w / 1e6} MW, FC Power: {h2_storage.power_m_w} MW")
    print(f"Max Energy Capacity (equivalent): {max_energy_gwh:.2f} GWh")
    print(f"Initial SOC: {h2_storage.get_soc():.3f}\n")

    charge_power = 50e6
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    h2_storage.update_state(-charge_power)
    print(f"After charging, SOC: {h2_storage.get_soc():.3f}, H2 Mass: {h2_storage.M_H2_kg / 1000:.2f} tons\n")

    discharge_power = 40e6
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for 1h ---")
    h2_storage.update_state(discharge_power)
    print(f"After discharging, SOC: {h2_storage.get_soc():.3f}, H2 Mass: {h2_storage.M_H2_kg / 1000:.2f} tons")