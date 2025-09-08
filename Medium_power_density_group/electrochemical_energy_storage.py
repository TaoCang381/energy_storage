# file: Medium_power_density_group/electrochemical_energy_storage.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel


# --- 修改区域 2: 让 EES 继承 BaseStorageModel ---
class ElectrochemicalEnergyStorage(BaseStorageModel):
    """
    通用电化学储能 (EES) 模型 (HESS集成版 - 严格对应论文公式)
    已按照BaseStorageModel进行接口标准化，并完整保留SOH衰减模型。
    """

    def __init__(self,
                 id,  # <--- 标准接口参数
                 dt_s,  # <--- 标准接口参数
                 initial_soc=0.5,
                 initial_soh=1.0,
                 rated_power_mw=100.0,  # 额定功率, 100 MW
                 nominal_capacity_mwh=200.0,  # 额定能量容量, 200 MWh
                 charge_efficiency=0.95,
                 discharge_efficiency=0.95,
                 soc_upper_limit=0.9,
                 soc_lower_limit=0.1,
                 cycle_life_model={
                     0.1: 10000, 0.3: 6000, 0.5: 4000,
                     0.8: 3000, 1.0: 2500
                 },
                 om_cost_per_mwh=50  # 元/MWh
                 ):

        # --- 关键改动 3: 调用父类的构造函数 ---
        super().__init__(id, dt_s)

        # --- 关键改动 4: 将参数赋值给父类中的标准属性 ---
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        self.capacity_mwh = nominal_capacity_mwh
        self.efficiency = np.sqrt(charge_efficiency * discharge_efficiency)
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh

        # --- 保留EES特有的物理参数和SOH模型 ---
        self.soh = initial_soh
        self.nominal_capacity_mwh = nominal_capacity_mwh
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency
        self.cycle_life_model = cycle_life_model

        # --- 实时工作参数 (受SOH影响) ---
        self.current_capacity_mwh = self.nominal_capacity_mwh * self.soh

        # --- 核心状态变量：储存的能量 E_ees ---
        self.E_ees_mwh = self.current_capacity_mwh * self.soc

        # SOH衰减计算相关状态
        self.last_cycle_soc_min = self.soc
        self.last_cycle_soc_max = self.soc
        self.state = 'idle'

        self.soh_history = []

    # ==============================================================================
    # --- 新增：核心标准接口 update_state ---
    # ==============================================================================
    def update_state(self, dispatch_power_w):
        """
        根据调度指令（单位：W）更新储能状态。
        这是被HESS系统统一调用的接口方法。
        """
        if dispatch_power_w > 0:
            # 正功率表示放电
            self.discharge(dispatch_power_w, self.dt_s)
        elif dispatch_power_w < 0:
            # 负功率表示充电
            self.charge(abs(dispatch_power_w), self.dt_s)
        else:
            # 零功率表示闲置
            self.idle_loss(self.dt_s)

    # ==============================================================================
    # --- 模型核心物理方法 (完全保留您原有的代码) ---
    # ==============================================================================

    def get_soc(self):
        """根据储存的能量计算并更新SOC"""
        # SOH变化后，当前可用容量会变
        self.current_capacity_mwh = self.nominal_capacity_mwh * self.soh
        if self.current_capacity_mwh > 1e-6:
            self.soc = self.E_ees_mwh / self.current_capacity_mwh
        else:
            self.soc = 0
        return self.soc

    def get_available_charge_power(self):
        if self.get_soc() >= self.soc_max: return 0
        return self.power_m_w * 1e6  # 返回单位 W

    def get_available_discharge_power(self):
        if self.get_soc() <= self.soc_min: return 0
        return self.power_m_w * 1e6  # 返回单位 W

    def charge(self, power_elec, time_s):
        """按指定电功率充电，对应动态方程"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        # 当从放电转为充电时，一个循环结束，计算SOH损耗
        if self.state == 'discharging':
            self._update_soh()
        self.state = 'charging'

        time_h = time_s / 3600.0
        delta_energy_mwh = (power_elec / 1e6) * self.eta_ch * time_h

        self.E_ees_mwh += delta_energy_mwh
        self.E_ees_mwh = min(self.E_ees_mwh, self.current_capacity_mwh * self.soc_max)

        self.get_soc()
        self.last_cycle_soc_max = max(self.last_cycle_soc_max, self.soc)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电，对应动态方程"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'

        time_h = time_s / 3600.0
        delta_energy_mwh = (power_elec / 1e6) / self.eta_dis * time_h

        self.E_ees_mwh -= delta_energy_mwh
        self.E_ees_mwh = max(self.E_ees_mwh, self.current_capacity_mwh * self.soc_min)

        self.get_soc()
        self.last_cycle_soc_min = min(self.last_cycle_soc_min, self.soc)

    def idle_loss(self, time_s):
        self.state = 'idle'
        # 简化模型，暂不考虑自放电

    def _update_soh(self):
        """核心方法：根据上一个循环的DoD来更新SOH"""
        dod = self.last_cycle_soc_max - self.last_cycle_soc_min
        if dod < 0.01:
            self.last_cycle_soc_min = self.get_soc()
            self.last_cycle_soc_max = self.get_soc()
            return

        # 使用插值来获得更精确的循环寿命
        dod_points = sorted(self.cycle_life_model.keys())
        life_points = [self.cycle_life_model[d] for d in dod_points]
        total_cycles_at_this_dod = np.interp(dod, dod_points, life_points)

        # 每完成这样一个充放循环，SOH就损耗 1 / total_cycles
        soh_loss = (1.0 / total_cycles_at_this_dod)

        self.soh -= soh_loss
        self.soh = max(0.8, self.soh)  # 假设SOH最低为80%

        # SOH变化后，更新工作参数
        self.current_capacity_mwh = self.nominal_capacity_mwh * self.soh

        # 重置本轮循环的SOC记录
        self.get_soc()
        self.last_cycle_soc_min = self.soc
        self.last_cycle_soc_max = self.soc


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    ees = ElectrochemicalEnergyStorage(id='ees_test', dt_s=3600, initial_soc=0.5)

    print(f"EES Initialized. Rated Power: {ees.power_m_w} MW, Rated Capacity: {ees.capacity_mwh} MWh")
    print(f"Initial SOC: {ees.get_soc():.3f}, Initial SOH: {ees.soh:.4f}\n")

    # 模拟充电: 以50MW功率充电1小时
    charge_power = 50e6
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    ees.update_state(-charge_power)  # 充电
    print(f"After charging, SOC: {ees.get_soc():.3f}, SOH: {ees.soh:.4f}\n")

    # 模拟放电: 以80MW功率放电1.5小时 (分两次，以模拟dt)
    discharge_power = 80e6
    print(f"--- Discharging with {discharge_power / 1e6} MW for 1h ---")
    ees.update_state(discharge_power)  # 放电
    print(f"After 1h discharging, SOC: {ees.get_soc():.3f}, SOH: {ees.soh:.4f}")

    # 再次充电，触发SOH更新
    print(f"--- Charging again to trigger SOH update ---")
    ees.update_state(-charge_power)  # 再次充电
    print(f"After 2nd charging, SOC: {ees.get_soc():.3f}, SOH: {ees.soh:.4f}\n")