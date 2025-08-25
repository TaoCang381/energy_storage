# file: PythonProject/Medium power density group/electrochemical_energy_storage.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit


class ElectrochemicalEnergyStorage(EnergyStorageUnit):
    """
    通用电化学储能 (EES) 模型 (HESS集成版 - 严格对应论文公式)
    代表系统中的所有电池储能，包含非线性OCV、与DoD相关的SOH衰减等关键特性。
    """

    def __init__(self,
                 ess_id="ees_01",
                 initial_soc=0.5,
                 initial_soh=1.0,
                 # ========================== 合理化参数配置 (开始) ==========================
                 # --- 核心规格 (配置为 100MW / 200MWh 系统) ---
                 rated_power_w=100e6,  # 额定功率 (W), e.g., 100 MW
                 nominal_capacity_kwh=200e3,  # 额定能量容量 (kWh), e.g., 200 MWh
                 charge_efficiency=0.95,  # 充电效率
                 discharge_efficiency=0.95,  # 放电效率

                 # --- 运行限制 ---
                 soc_upper_limit=0.9,
                 soc_lower_limit=0.1,

                 # --- OCV-SOC 模型参数 (以锂电池为参考) ---
                 ocv_params={'P1': 3.5, 'P2': 0.05, 'P3': 0.1, 'P4': 0.15},

                 # --- DoD-Cycle Life 衰减模型 (核心创新点) ---
                 cycle_life_model={
                     0.1: 10000,  # 10% DoD: 可循环10000次
                     0.3: 6000,  # 30% DoD: 可循环6000次
                     0.5: 4000,  # 50% DoD: 可循环4000次
                     0.8: 3000,  # 80% DoD: 可循环3000次
                     1.0: 2500  # 100% DoD: 可循环2500次
                 },
                 # ========================== 合理化参数配置 (结束) ==========================

                 cost_per_kwh=0.0002  # 每循环一度电的等效寿命成本
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)

        # --- 物理与性能参数 ---
        self.rated_power_w = rated_power_w
        self.nominal_capacity_kwh = nominal_capacity_kwh
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency
        self.soc_max = soc_upper_limit
        self.soc_min = soc_lower_limit
        self.ocv_params = ocv_params
        self.cycle_life_model = cycle_life_model
        self.cost_per_kwh = cost_per_kwh

        # --- 实时工作参数 (受SOH影响) ---
        self.capacity_kwh = self.nominal_capacity_kwh * self.soh

        # --- 核心状态变量：储存的能量 E_ees ---
        self.E_ees_kwh = self.capacity_kwh * self.soc

        # SOH衰减计算相关状态
        self.last_cycle_soc_min = self.soc
        self.last_cycle_soc_max = self.soc

        self.energy_history = []
        self.soh_history = []

    def get_soc(self):
        """根据储存的能量计算并更新SOC"""
        if self.capacity_kwh > 0:
            self.soc = self.E_ees_kwh / self.capacity_kwh
        else:
            self.soc = 0
        return self.soc

    def get_available_charge_power(self):
        if self.get_soc() >= self.soc_max: return 0
        return self.rated_power_w

    def get_available_discharge_power(self):
        if self.get_soc() <= self.soc_min: return 0
        return self.rated_power_w

    def charge(self, power_elec, time_s):
        """按指定电功率充电，对应动态方程"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return

        # 当从放电转为充电时，一个循环结束，计算SOH损耗
        if self.state == 'discharging':
            self._update_soh()
        self.state = 'charging'

        time_h = time_s / 3600.0
        delta_energy = power_elec * self.eta_ch * time_h / 1000  # convert W to kWh

        self.E_ees_kwh += delta_energy

        # 应用容量约束
        self.E_ees_kwh = min(self.E_ees_kwh, self.capacity_kwh * self.soc_max)

        # 更新并记录本轮循环的SOC最大值
        self.get_soc()  # 更新self.soc
        self.last_cycle_soc_max = max(self.last_cycle_soc_max, self.soc)

        self._record_history_ees(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电，对应动态方程"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return
        self.state = 'discharging'

        time_h = time_s / 3600.0
        delta_energy = (power_elec / self.eta_dis) * time_h / 1000  # convert W to kWh

        self.E_ees_kwh -= delta_energy

        # 应用容量约束
        self.E_ees_kwh = max(self.E_ees_kwh, self.capacity_kwh * self.soc_min)

        # 更新并记录本轮循环的SOC最小值
        self.get_soc()  # 更新self.soc
        self.last_cycle_soc_min = min(self.last_cycle_soc_min, self.soc)

        self._record_history_ees(time_s, -power_elec)

    def idle_loss(self, time_s):
        self.state = 'idle'
        # 简化模型，暂不考虑自放电
        self._record_history_ees(time_s, 0)

    def _update_soh(self):
        """核心方法：根据上一个循环的DoD来更新SOH"""
        dod = self.last_cycle_soc_max - self.last_cycle_soc_min
        if dod < 0.01:  # 忽略太浅的循环
            # 重置本轮循环的SOC记录
            self.last_cycle_soc_min = self.soc
            self.last_cycle_soc_max = self.soc
            return

        # 查找该DoD对应的循环寿命
        life_key = min([k for k in self.cycle_life_model.keys() if k >= dod], default=1.0)
        total_cycles_at_this_dod = self.cycle_life_model[life_key]

        # 每完成这样一个循环，SOH就损耗 1 / total_cycles
        # 我们这里完成的只是一个部分循环，等效于 0.5 个完整循环
        soh_loss = (1.0 / total_cycles_at_this_dod) * 0.5

        self.soh -= soh_loss
        self.soh = max(0, self.soh)  # SOH不小于0

        # SOH变化后，更新工作参数
        self.capacity_kwh = self.nominal_capacity_kwh * self.soh

        # 重置本轮循环的SOC记录
        self.last_cycle_soc_min = self.soc
        self.last_cycle_soc_max = self.soc

    def _record_history_ees(self, time_delta, power):
        """记录EES特有的历史数据"""
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.energy_history.append(self.E_ees_kwh)
        self.soh_history.append(self.soh)


# --- 单元测试用的示例函数 (使用合理化参数) ---
def simulate_ees_test():
    ees = ElectrochemicalEnergyStorage(initial_soc=0.5)

    print(
        f"EES Initialized. Rated Power: {ees.rated_power_w / 1e6} MW, Rated Capacity: {ees.nominal_capacity_kwh / 1000} MWh")
    print(f"Initial SOC: {ees.get_soc():.3f}, Initial SOH: {ees.soh:.4f}\n")

    # 模拟充电: 以50MW功率充电1小时
    charge_power = 50e6
    charge_time = 3600
    print(f"--- Charging with {charge_power / 1e6} MW for {charge_time / 3600}h ---")
    ees.charge(charge_power, charge_time)
    print(f"After charging, SOC: {ees.get_soc():.3f}, SOH: {ees.soh:.4f}\n")

    # 模拟放电: 以80MW功率放电1.5小时
    discharge_power = 80e6
    discharge_time = 5400
    print(f"--- Discharging with {discharge_power / 1e6} MW for {discharge_time / 3600}h ---")
    ees.discharge(discharge_power, discharge_time)
    print(f"After discharging, SOC: {ees.get_soc():.3f}, SOH: {ees.soh:.4f}\n")

    # 再次充电，触发SOH更新
    print(f"--- Charging again to trigger SOH update ---")
    ees.charge(charge_power, charge_time)
    print(f"After 2nd charging, SOC: {ees.get_soc():.3f}, SOH: {ees.soh:.4f}\n")


if __name__ == "__main__":
    simulate_ees_test()