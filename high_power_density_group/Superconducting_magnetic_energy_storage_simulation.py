# file: PythonProject/high_power_density_group/Superconducting_magnetic_energy_storage_simulation.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit


class SuperconductingMagneticEnergyStorage(EnergyStorageUnit):
    """
    超导磁储能 (SMES) 模型 (物理升级版 - 严格对应论文公式)
    模型基于电磁感应定律，核心状态为线圈电流，并包含制冷系统损耗。
    """

    def __init__(self,
                 ess_id="smes_01",
                 initial_soc=0.5,
                 initial_soh=1.0,
                 # ========================== 合理化参数配置 (开始) ==========================
                 # --- 核心物理参数 (配置为 5MW / 5MJ 系统) ---
                 inductance_H=2.5,  # 电感 (H, 亨利)
                 max_current_A=2000.0,  # 最大/临界电流 (A)
                 min_current_A=200.0,  # 最小工作电流 (A), 避免奇异性

                 # --- PCS 和制冷系统参数 ---
                 pcs_rated_power_w=5e6,  # PCS额定功率 (W), e.g., 5 MW
                 pcs_efficiency=0.97,  # PCS转换效率
                 cryogenic_power_w=50000,  # 低温系统维持功率 (W), e.g., 50 kW (恒定损耗)
                 pcs_max_voltage_v=3000,  # PCS最大输出电压 (V)
                 # ========================== 合理化参数配置 (结束) ==========================

                 cost_per_kwh=0.02  # 运行成本主要为电费
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)
        self.soc_min = 0.0
        self.soc_max = 1.0
        # --- 物理与性能参数 ---
        self.L_smes = inductance_H
        self.I_max = max_current_A
        self.I_min = min_current_A
        self.rated_power_w = pcs_rated_power_w
        self.eta_pcs = pcs_efficiency  # 假设充放电效率相同
        self.P_cryo = cryogenic_power_w
        self.V_pcs_max = pcs_max_voltage_v
        self.cost_per_kwh = cost_per_kwh

        # --- 核心状态变量：线圈电流 I_smes ---
        # SOC_E = (E - E_min) / (E_max - E_min) = (I^2 - I_min^2) / (I_max^2 - I_min^2)
        i_range_sq = self.I_max ** 2 - self.I_min ** 2
        self.I_smes = math.sqrt(self.soc * i_range_sq + self.I_min ** 2)

        self.current_history = []

    def get_soc(self):
        """根据线圈电流计算并更新SOC (基于能量)"""
        i_range_sq = self.I_max ** 2 - self.I_min ** 2
        if i_range_sq <= 0: return 0
        self.soc = (self.I_smes ** 2 - self.I_min ** 2) / i_range_sq
        return self.soc

    def _get_pcs_voltage(self, power_elec_net, is_charging):
        """根据净电功率指令计算PCS施加的电压"""
        # 防止除以零
        current_I = self.I_smes if self.I_smes > 1e-3 else 1e-3

        if is_charging:
            # V_pcs = (P_elec * eta) / I
            voltage = (power_elec_net * self.eta_pcs) / current_I
        else:
            # V_pcs = - P_elec / (I * eta)
            voltage = - (power_elec_net / (current_I * self.eta_pcs))

        # 应用电压约束
        return max(-self.V_pcs_max, min(self.V_pcs_max, voltage))

    def _update_current(self, V_pcs, time_s):
        """根据PCS电压更新线圈电流"""
        # dI = (V / L) * dt
        self.I_smes += (V_pcs / self.L_smes) * time_s
        # 应用电流约束
        self.I_smes = max(self.I_min, min(self.I_smes, self.I_max))

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (W)"""
        if self.I_smes >= self.I_max: return 0
        return self.rated_power_w

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (W)"""
        if self.I_smes <= self.I_min: return 0
        return self.rated_power_w

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (制冷功耗由EMS在系统层面考虑)"""
        power_elec_net = min(power_elec, self.get_available_charge_power())
        if power_elec_net <= 0: return
        self.state = 'charging'

        V_pcs = self._get_pcs_voltage(power_elec_net, is_charging=True)
        self._update_current(V_pcs, time_s)

        # 记录的功率是PCS的净功率
        self._record_history_smes(time_s, power_elec_net)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (制冷功耗由EMS在系统层面考虑)"""
        power_elec_net = min(power_elec, self.get_available_discharge_power())
        if power_elec_net <= 0: return
        self.state = 'discharging'

        V_pcs = self._get_pcs_voltage(power_elec_net, is_charging=False)
        self._update_current(V_pcs, time_s)

        self._record_history_smes(time_s, -power_elec_net)

    def idle_loss(self, time_s):
        """闲置时，线圈电流无损耗，但系统仍有制冷功耗"""
        self.state = 'idle'
        # 理想超导体，电流不变，V_pcs = 0
        self._update_current(0, time_s)
        self._record_history_smes(time_s, 0)

    def get_total_power(self, net_power):
        """供EMS调用，计算包含制冷损耗的总功率"""
        # 充电时，总功率 = PCS功率 + 制冷功率
        # 放电时，总功率 = PCS功率 - 制冷功率 (负数)
        return net_power + self.P_cryo if net_power >= 0 else net_power - self.P_cryo

    def _record_history_smes(self, time_delta, power_net):
        """记录SMES特有的历史数据"""
        current_soc = self.get_soc()
        # 总功率 = 净功率 + 制冷功率
        total_power = self.get_total_power(power_net)
        # 调用父类方法记录总功率和SOC
        super()._record_history(time_delta, total_power, current_soc)
        # 记录自身特有历史
        self.current_history.append(self.I_smes)


# --- 单元测试用的示例函数 (使用合理化参数) ---
def simulate_smes_test():
    smes = SuperconductingMagneticEnergyStorage(initial_soc=0.5)

    max_energy_mj = 0.5 * smes.L_smes * smes.I_max ** 2 / 1e6
    print(f"SMES Initialized. Max Energy: {max_energy_mj:.2f} MJ")
    print(f"Initial SOC: {smes.get_soc():.3f}, Initial Current: {smes.I_smes:.2f} A\n")

    charge_power = 4e6  # 4 MW
    charge_time = 0.5  # 0.5 s
    print(f"--- Charging with {charge_power / 1e6} MW for {charge_time}s ---")
    smes.charge(charge_power, charge_time)
    print(f"After charging, SOC: {smes.get_soc():.3f}, Current: {smes.I_smes:.2f} A")
    total_power = smes.get_total_power(charge_power)
    print(f"Net power: {charge_power / 1e6:.3f} MW, Total power (with cryo): {total_power / 1e6:.3f} MW\n")

    idle_time = 1.0  # 1 s
    print(f"--- Idling for {idle_time}s ---")
    smes.idle_loss(idle_time)
    print(f"After idling, SOC: {smes.get_soc():.3f}, Current: {smes.I_smes:.2f} A")
    total_power = smes.get_total_power(0)
    print(f"Net power: 0.000 MW, Total power (with cryo): {total_power / 1e6:.3f} MW\n")

    discharge_power = 5e6  # 5 MW
    discharge_time = 0.4  # 0.4 s
    print(f"--- Discharging with {discharge_power / 1e6} MW for {discharge_time}s ---")
    smes.discharge(discharge_power, discharge_time)
    print(f"After discharging, SOC: {smes.get_soc():.3f}, Current: {smes.I_smes:.2f} A")
    total_power = smes.get_total_power(-discharge_power)
    print(f"Net power: {-discharge_power / 1e6:.3f} MW, Total power (with cryo): {total_power / 1e6:.3f} MW\n")


if __name__ == "__main__":
    simulate_smes_test()