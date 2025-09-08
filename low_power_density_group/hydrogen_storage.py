# file: low_power_density_group/hydrogen_storage.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel

# --- 物理常数 ---
LHV_H2_KWH_PER_KG = 33.3  # 氢气低热值 (kWh/kg)


# --- 修改区域 2: 让 HES 继承 BaseStorageModel ---
class HydrogenStorage(BaseStorageModel):
    """
    氢储能 (HES) 模型 (HESS集成版 - 严格对应论文公式)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,  # <--- 标准接口参数
                 dt_s,  # <--- 标准接口参数
                 initial_soc=0.5,
                 initial_soh=1.0,
                 electrolyzer_rated_power_mw=50.0,  # 电解槽额定功率 (MW)
                 electrolyzer_efficiency_kwh_kg=50,  # 制氢电耗 (kWh/kg)
                 tank_max_capacity_kg=100000,  # 储氢罐最大容量 (kg)
                 compressor_power_ratio=0.08,  # 压缩机功率占比
                 fuel_cell_rated_power_mw=40.0,  # 燃料电池额定电功率 (MW)
                 fc_elec_efficiency=0.55,  # 燃料电池发电效率
                 fc_heat_recovery_efficiency=0.35,  # 余热回收效率
                 soc_upper_limit=0.95,
                 soc_lower_limit=0.05,
                 om_cost_per_mwh=10  # 元/MWh
                 ):

        # --- 关键改动 3: 调用父类的构造函数 ---
        super().__init__(id, dt_s)

        # --- 关键改动 4: 将参数赋值给父类中的标准属性 ---
        self.soc = initial_soc
        self.power_m_w = fuel_cell_rated_power_mw  # 以燃料电池功率作为额定功率
        self.capacity_mwh = tank_max_capacity_kg * LHV_H2_KWH_PER_KG / 1000  # 转换为 MWh
        # 计算往返效率: 电->氢->电
        kg_per_kwh_ely = 1 / electrolyzer_efficiency_kwh_kg
        kwh_per_kg_fc = LHV_H2_KWH_PER_KG * fc_elec_efficiency
        self.efficiency = kg_per_kwh_ely * kwh_per_kg_fc

        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh
        self.soh = initial_soh

        # --- 保留HES特有的物理参数 ---
        self.P_ely_rated_w = electrolyzer_rated_power_mw * 1e6
        self.eta_ely_kwh_kg = electrolyzer_efficiency_kwh_kg
        self.M_tank_max = tank_max_capacity_kg
        self.compressor_ratio = compressor_power_ratio
        self.P_fc_rated_w = self.power_m_w * 1e6
        self.eta_fc_elec = fc_elec_efficiency
        self.eta_fc_heat = fc_heat_recovery_efficiency

        # --- 实时工作参数 (受SOH影响) ---
        self.current_fc_efficiency = self.eta_fc_elec * self.soh
        self.current_ely_efficiency_kwh_kg = self.eta_ely_kwh_kg / self.soh

        # --- 核心状态变量：储氢质量 M_H2 ---
        self.M_H2_kg = self.M_tank_max * self.soc

        self.mass_history = []
        self.heat_power_history = []
        self.state = 'idle'

    # ==============================================================================
    # --- 新增：核心标准接口 update_state ---
    # ==============================================================================
    def update_state(self, dispatch_power_w):
        """
        根据调度指令（单位：W）更新储能状态。
        """
        if dispatch_power_w > 0:
            # 正功率表示放电 (发电)
            self.discharge(dispatch_power_w, self.dt_s)
        elif dispatch_power_w < 0:
            # 负功率表示充电 (制氢)
            self.charge(abs(dispatch_power_w), self.dt_s)
        else:
            # 零功率表示闲置
            self.idle_loss(self.dt_s)

    # ==============================================================================
    # --- 模型核心物理方法 (完全保留您原有的代码) ---
    # ==============================================================================

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
        m_dot_ely = power_to_ely_kw / self.current_ely_efficiency_kwh_kg  # kg/h

        time_h = time_s / 3600.0
        self.M_H2_kg += m_dot_ely * time_h
        self.M_H2_kg = min(self.M_H2_kg, self.M_tank_max * self.soc_max)

        self.heat_power_history.append(0)  # 充电时不产热

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，并计算伴生的热功率"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'

        power_elec_kw = power_elec / 1000
        time_h = time_s / 3600.0
        # 计算耗氢速率 (kg/h)
        m_dot_fc = power_elec_kw / (LHV_H2_KWH_PER_KG * self.current_fc_efficiency)

        mass_consumed_kg = m_dot_fc * time_h
        # 检查氢气余量，如果不足则按比例降低发电功率
        if mass_consumed_kg > (self.M_H2_kg - self.M_tank_max * self.soc_min):
            available_mass = self.M_H2_kg - self.M_tank_max * self.soc_min
            if time_h > 0:
                m_dot_fc = available_mass / time_h
                power_elec_kw = m_dot_fc * LHV_H2_KWH_PER_KG * self.current_fc_efficiency
                power_elec = power_elec_kw * 1000
            else:
                power_elec = 0
            mass_consumed_kg = available_mass

        self.M_H2_kg -= mass_consumed_kg
        self.M_H2_kg = max(self.M_H2_kg, self.M_tank_max * self.soc_min)

        # 计算伴生的热功率
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

    # 模拟以50MW功率连续制氢24小时
    charge_power = 50e6
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    h2_storage.update_state(-charge_power)  # 充电
    print(f"After charging, SOC: {h2_storage.get_soc():.3f}, H2 Mass: {h2_storage.M_H2_kg / 1000:.2f} tons\n")

    # 模拟以40MW电功率连续发电12小时
    discharge_power = 40e6
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for 1h ---")
    h2_storage.update_state(discharge_power)  # 放电
    print(f"After discharging, SOC: {h2_storage.get_soc():.3f}, H2 Mass: {h2_storage.M_H2_kg / 1000:.2f} tons")