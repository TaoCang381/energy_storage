# file: PythonProject/high_power_density_group/supercapacitor_simulation.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit


class Supercapacitor(EnergyStorageUnit):
    """
    超级电容器模型 (HESS集成版 - 依据Word文档简化模型)
    模型严格按照Word文档中定义的宏观动态方程和约束构建。
    """

    def __init__(self,
                 ess_id="supercap_01",
                 initial_soc=0.5,
                 initial_soh=1.0,
                 # ========================== 参数修改区 (开始) ==========================
                 # --- 核心物理参数 (调整为更合理的工业模组级别) ---
                 capacitance_F=120000,  # 电容量 (F), e.g., 120 kF. 对应约100Wh的储能.
                 max_voltage=48,  # 最大工作电压 (V), e.g., 一个48V的模组.
                 min_voltage=24,  # 最小工作电压 (V).
                 esr_ohm=0.005,  # 等效串联电阻 (Ohm, ESR).

                 # --- 运行限制 ---
                 rated_power=50000,  # 额定功率 (W), e.g., 50kW.
                 rated_current=1000,  # 额定电流 (A).

                 # --- 损耗参数 ---
                 self_discharge_rate_sigma=1e-7,  # 小时自放电率 (s^-1).
                 # ========================== 参数修改区 (结束) ==========================

                 # --- HESS集成参数 ---
                 cost_per_kwh=0.08
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)

        # --- 物理与性能参数 ---
        self.C_sc = capacitance_F * self.soh
        self.V_max = max_voltage
        self.V_min = min_voltage
        self.R_esr = esr_ohm / self.soh
        self.rated_power_elec = rated_power
        self.rated_current_sc = rated_current
        self.sigma = self_discharge_rate_sigma
        self.cost_per_kwh = cost_per_kwh

        # --- 核心状态变量：电压 V_sc ---
        v_range_sq = self.V_max ** 2 - self.V_min ** 2
        self.V_sc = math.sqrt(self.soc * v_range_sq + self.V_min ** 2)

        self.voltage_history = []

    def get_soc(self):
        """根据电压计算并更新SOC (基于能量)"""
        v_range_sq = self.V_max ** 2 - self.V_min ** 2
        if v_range_sq <= 0: return 0
        self.soc = (self.V_sc ** 2 - self.V_min ** 2) / v_range_sq
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (W)，依据Word文档约束"""
        if self.V_sc >= self.V_max: return 0

        power_limit_by_p = self.rated_power_elec
        power_limit_by_i = self.V_sc * self.rated_current_sc + self.rated_current_sc ** 2 * self.R_esr

        return min(power_limit_by_p, power_limit_by_i)

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (W)，依据Word文档约束"""
        if self.V_sc <= self.V_min: return 0

        power_limit_by_p = self.rated_power_elec
        power_limit_by_i = self.V_sc * self.rated_current_sc - self.rated_current_sc ** 2 * self.R_esr

        return min(power_limit_by_p, power_limit_by_i)

    def charge(self, power_elec, time_s):
        """按指定电功率充电"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return
        self.state = 'charging'

        current = power_elec / self.V_sc if self.V_sc > 1e-3 else self.rated_current_sc
        current = min(current, self.rated_current_sc)

        delta_v = (current * time_s) / self.C_sc
        self.V_sc += delta_v

        self.V_sc = min(self.V_sc, self.V_max)

        self._record_history_sc(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return
        self.state = 'discharging'

        current = power_elec / self.V_sc if self.V_sc > 1e-3 else 0
        current = min(current, self.rated_current_sc)

        delta_v = (current * time_s) / self.C_sc
        self.V_sc -= delta_v

        self.V_sc = max(self.V_sc, self.V_min)

        self._record_history_sc(time_s, -power_elec)

    def idle_loss(self, time_s):
        """计算闲置时的自放电损耗，对应公式中的 sigma 项"""
        self.state = 'idle'

        self.V_sc *= (1 - self.sigma * time_s)

        self._record_history_sc(time_s, 0)

    def _record_history_sc(self, time_delta, power):
        """记录超级电容特有的历史数据"""
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.voltage_history.append(self.V_sc)


# --- 单元测试用的示例函数 (修改了测试工况) ---
def simulate_sc_test():
    # 使用默认的、经过合理化调整的参数来创建实例
    sc = Supercapacitor(initial_soc=0.5)

    # 打印初始状态和可用能量
    total_energy_kj = 0.5 * sc.C_sc * (sc.V_max ** 2 - sc.V_min ** 2) / 1000
    print(f"Supercapacitor Initialized. Usable Energy: {total_energy_kj:.2f} kJ")
    print(f"Initial SOC: {sc.get_soc():.3f}, Initial Voltage: {sc.V_sc:.3f} V\n")

    # 测试工况：使用与设备额定功率相匹配的功率进行测试
    charge_power = 40000  # 40 kW
    charge_time = 20  # 20 s
    print(f"--- Charging with {charge_power / 1000} kW for {charge_time}s ---")
    sc.charge(charge_power, charge_time)
    print(f"After charging, SOC: {sc.get_soc():.3f}, Voltage: {sc.V_sc:.3f} V\n")

    idle_time = 30  # 30 s
    print(f"--- Idling for {idle_time}s ---")
    sc.idle_loss(idle_time)
    print(f"After idling, SOC: {sc.get_soc():.3f}, Voltage: {sc.V_sc:.3f} V\n")

    discharge_power = 50000  # 50 kW
    discharge_time = 15  # 15 s
    print(f"--- Discharging with {discharge_power / 1000} kW for {discharge_time}s ---")
    sc.discharge(discharge_power, discharge_time)
    print(f"After discharging, SOC: {sc.get_soc():.3f}, Voltage: {sc.V_sc:.3f} V\n")


if __name__ == "__main__":
    simulate_sc_test()