# file: low_power_density_group/thermal_storage.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel


# --- 修改区域 2: 让 TES 继承 BaseStorageModel ---
class ThermalEnergyStorage(BaseStorageModel):
    """
    热储能 (TES) 模型 (HESS集成版 - 严格对应论文公式)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,  # <--- 标准接口参数
                 dt_s,  # <--- 标准接口参数
                 initial_soc=0.5,
                 storage_medium_mass_kg=8e7,
                 specific_heat_capacity_j_kgk=1500,
                 max_temp_k=838.15,  # 565°C
                 min_temp_k=563.15,  # 290°C
                 heater_rated_power_mw=110.0,
                 heat_engine_rated_power_mw=100.0,
                 elec_to_heat_efficiency=0.98,
                 heat_to_elec_efficiency=0.42,
                 heat_loss_rate_percent_hr=0.04,
                 soc_upper_limit=1.0,
                 soc_lower_limit=0.0,
                 om_cost_per_mwh=10  # 元/MWh
                 ):

        # --- 关键改动 3: 调用父类的构造函数 ---
        super().__init__(id, dt_s)

        # SOH对于热储能基本不变
        self.soh = 1.0

        # --- 关键改动 4: 将参数赋值给父类中的标准属性 ---
        self.soc = initial_soc
        self.power_m_w = heat_engine_rated_power_mw  # 以热机发电功率作为额定功率
        # 计算最大储热量 (焦耳)
        h_tes_max_j = storage_medium_mass_kg * specific_heat_capacity_j_kgk * (max_temp_k - min_temp_k)
        # 计算等效的电容量 (MWh)
        self.capacity_mwh = (h_tes_max_j * heat_to_elec_efficiency) / 3.6e9
        # 计算电-热-电往返效率
        self.efficiency = elec_to_heat_efficiency * heat_to_elec_efficiency
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh

        # --- 保留TES特有的物理参数 ---
        self.m = storage_medium_mass_kg
        self.c = specific_heat_capacity_j_kgk
        self.T_max = max_temp_k
        self.T_min = min_temp_k
        self.P_heater_rated_w = heater_rated_power_mw * 1e6
        self.P_gen_rated_w = self.power_m_w * 1e6
        self.eta_e2h = elec_to_heat_efficiency
        self.eta_h2e = heat_to_elec_efficiency
        self.theta_loss = (heat_loss_rate_percent_hr / 100) / 3600.0  # 转换为每秒损失率

        # --- 核心状态变量：储存的热量 H_tes ---
        self.H_tes_max_J = h_tes_max_j
        self.H_tes_J = self.H_tes_max_J * self.soc

        self.energy_history = []
        self.temp_history = []
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
            # 负功率表示充电 (加热)
            self.charge(abs(dispatch_power_w), self.dt_s)
        else:
            # 零功率表示闲置 (但有散热)
            self.idle_loss(self.dt_s)

    # ==============================================================================
    # --- 模型核心物理方法 (完全保留您原有的代码) ---
    # ==============================================================================

    def get_soc(self):
        """根据储存的热量计算并更新SOC"""
        if self.H_tes_max_J > 1e-6:
            self.soc = self.H_tes_J / self.H_tes_max_J
        else:
            self.soc = self.soc_min
        return self.soc

    def get_current_temp_k(self):
        """根据热量计算当前温度"""
        if self.m * self.c < 1e-6: return self.T_min
        return self.T_min + self.H_tes_J / (self.m * self.c)

    def get_available_charge_power(self):
        """获取当前可用的充电(加热)功率 (W_e)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_heater_rated_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W_e)"""
        if self.get_soc() <= self.soc_min: return 0
        return self.P_gen_rated_w

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (加热)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'
        power_heat_in = power_elec * self.eta_e2h
        delta_heat = power_heat_in * time_s
        self.H_tes_J += delta_heat
        # 充电时也考虑散热
        self.idle_loss(time_s)
        self.H_tes_J = min(self.H_tes_J, self.H_tes_max_J * self.soc_max)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'
        power_heat_out = power_elec / self.eta_h2e
        delta_heat = power_heat_out * time_s

        # 检查余热是否足够
        if delta_heat > self.H_tes_J:
            self.idle_loss(time_s)
            return

        self.H_tes_J -= delta_heat
        # 放电时也考虑散热
        self.idle_loss(time_s)
        self.H_tes_J = max(self.H_tes_J, self.H_tes_max_J * self.soc_min)

    def idle_loss(self, time_s):
        """模拟闲置时的散热损失"""
        # 只有在充电和放电之外的状态才标记为idle
        if self.state not in ['charging', 'discharging']:
            self.state = 'idle'

        lost_heat = self.H_tes_max_J * self.theta_loss * time_s
        self.H_tes_J -= lost_heat
        self.H_tes_J = max(0, self.H_tes_J)


# --- 单元测试用的示例函数 (保持不变) ---
if __name__ == "__main__":
    tes = ThermalEnergyStorage(id='tes_test', dt_s=3600, initial_soc=0.2)

    max_energy_gwh_e = tes.capacity_mwh / 1000
    print(f"TES Initialized. Gen Power: {tes.power_m_w} MW, Usable Elec Energy: {max_energy_gwh_e * 1000:.2f} MWh")
    print(f"Initial SOC: {tes.get_soc():.3f}\n")

    # 模拟以110MW功率连续充电6小时
    charge_power = 110e6
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    tes.update_state(-charge_power)  # 充电
    print(f"After charging, SOC: {tes.get_soc():.3f}, Temp: {tes.get_current_temp_k() - 273.15:.1f}°C\n")

    # 模拟闲置2小时
    print(f"--- Idling for 1h ---")
    tes.update_state(0)  # 闲置
    print(f"After idling, SOC: {tes.get_soc():.3f}, Temp: {tes.get_current_temp_k() - 273.15:.1f}°C\n")

    # 模拟以100MW电功率连续发电4小时
    discharge_power = 100e6
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for 1h ---")
    tes.update_state(discharge_power)  # 放电
    print(f"After discharging, SOC: {tes.get_soc():.3f}, Temp: {tes.get_current_temp_k() - 273.15:.1f}°C\n")