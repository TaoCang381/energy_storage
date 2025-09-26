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
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 heater_rated_power_mw=30.0,  # 充电（加热）功率
                 heat_engine_rated_power_mw=30.0,  # 放电（发电）功率
                 rated_capacity_mwh=300.0,  # 额定等效电容量

                 # 效率与成本参数
                 elec_to_heat_efficiency=0.98,  # 电加热效率非常高
                 heat_to_elec_efficiency=0.45,  # 热转电效率（朗肯循环等），受卡诺循环限制
                 om_cost_per_mwh=8,

                 # --- 其他关键参数 ---
                 soc_upper_limit=0.95,
                 soc_lower_limit=0.1,

                 # --- 物理特性参数（可选择性提供，或使用默认值）---
                 # 典型熔融盐工作温度
                 max_temp_k=838.15,  # 565°C
                 min_temp_k=563.15,  # 290°C
                 # 典型储热介质比热容
                 specific_heat_capacity_j_kgk=1500,  # J/(kg·K)
                 # 每小时热损失率
                 heat_loss_rate_percent_hr=0.04
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = heat_engine_rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh
        # 电-热-电往返效率
        self.efficiency = elec_to_heat_efficiency * heat_to_elec_efficiency
        self.soh = 1.0  # 热储能SOH衰减很小，简化为1

        # 3. 保留TES特有的物理参数
        self.P_heater_rated_w = heater_rated_power_mw * 1e6
        self.P_gen_rated_w = self.power_m_w * 1e6
        self.eta_e2h = elec_to_heat_efficiency
        self.eta_h2e = heat_to_elec_efficiency
        self.T_max = max_temp_k
        self.T_min = min_temp_k
        self.c = specific_heat_capacity_j_kgk
        self.theta_loss = (heat_loss_rate_percent_hr / 100) / 3600.0

        # 4. 根据顶层参数，反向推算内部物理参数
        # 核心推算：根据能量公式 E_elec = E_heat * eta_h2e, E_heat = m * c * dT
        # 反算需要多大质量的储热介质
        # E_elec (J) = capacity_mwh * 3.6e9
        energy_elec_joules = self.capacity_mwh * 3.6e9
        if self.eta_h2e > 1e-6:
            self.H_tes_max_J = energy_elec_joules / self.eta_h2e
        else:
            self.H_tes_max_J = 0

        delta_T = self.T_max - self.T_min
        if self.c * delta_T > 1e-6:
            self.m = self.H_tes_max_J / (self.c * delta_T)
        else:
            self.m = 0

        # 5. 初始化核心状态变量：储存的热量 H_tes (单位: 焦耳)
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