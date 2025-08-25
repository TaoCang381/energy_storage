# file: PythonProject/high_power_density_group/flywheel_simulation.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit


class FlywheelModel(EnergyStorageUnit):
    """
    飞轮储能系统模型 (动力学升级版 - 严格对应论文公式)
    模型严格按照旋转动力学方程组进行构建。
    """

    def __init__(self,
                 ess_id="flywheel_01",
                 initial_soc=0.5,
                 initial_soh=1.0,
                 moment_of_inertia_J=1000,
                 max_angular_vel=1500,
                 min_angular_vel=300,
                 rated_power=2e6,
                 rated_torque=2000,
                 charge_efficiency=0.95,
                 discharge_efficiency=0.95,
                 friction_coeff_kf=0.1,
                 cost_per_kwh=0.05
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)
        self.soc_min = 0.0
        self.soc_max = 1.0
        self.J = moment_of_inertia_J
        self.omega_max = max_angular_vel
        self.omega_min = min_angular_vel
        self.rated_power_w = rated_power
        self.rated_torque_mg = rated_torque
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency
        self.kf = friction_coeff_kf
        self.cost_per_kwh = cost_per_kwh

        # 核心状态变量：角速度 omega
        self.omega = math.sqrt(self.soc * (self.omega_max ** 2 - self.omega_min ** 2) + self.omega_min ** 2)

        self.angular_vel_history = []

    # ==============================================================================
    # --- 模型核心方法：严格对应Word文档中的“方程组” ---
    # ==============================================================================

    def _get_electromagnetic_torque(self, power_elec, is_charging):
        """
        对应Word公式 (3): 功率-扭矩转换关系
        """
        # 防止除以零
        current_omega = self.omega if self.omega > 1e-3 else 1e-3

        if is_charging:
            # 充电时，扭矩为正（加速）
            tau_mg = (power_elec * self.eta_ch) / current_omega
        else:
            # 放电时，扭矩为负（减速），因为它是阻力矩
            tau_mg = - (power_elec / (current_omega * self.eta_dis))

        # 应用扭矩约束
        return max(-self.rated_torque_mg, min(self.rated_torque_mg, tau_mg))

    def _get_loss_torque(self):
        """
        对应Word公式 (4): 损耗扭矩模型
        """
        return self.kf * self.omega

    def _get_net_torque(self, tau_mg):
        """
        对应Word公式 (2): 净扭矩构成
        """
        return tau_mg - self._get_loss_torque()

    def _update_angular_velocity(self, tau_net, time_s):
        """
        对应Word公式 (1): 角速度动态方程
        """
        # dw = (tau_net / J) * dt
        self.omega += (tau_net / self.J) * time_s
        # 应用角速度约束
        self.omega = max(self.omega_min, min(self.omega, self.omega_max))

    # ==============================================================================
    # --- HESS标准接口实现 ---
    # ==============================================================================

    def get_soc(self):
        """根据角速度计算并更新SOC"""
        omega_range_sq = self.omega_max ** 2 - self.omega_min ** 2
        if omega_range_sq <= 0: return 0
        self.soc = (self.omega ** 2 - self.omega_min ** 2) / omega_range_sq
        return self.soc

    def get_available_charge_power(self):
        if self.omega >= self.omega_max:
            return 0
        power_limit_by_rated_p = self.rated_power_w
        power_limit_by_rated_t = (self.rated_torque_mg * self.omega) / self.eta_ch
        return min(power_limit_by_rated_p, power_limit_by_rated_t)

    def get_available_discharge_power(self):
        if self.omega <= self.omega_min:
            return 0
        power_limit_by_rated_p = self.rated_power_w
        power_limit_by_rated_t = (self.rated_torque_mg * self.omega) * self.eta_dis
        return min(power_limit_by_rated_p, power_limit_by_rated_t)

    def charge(self, power_elec, time_s):
        """按指定电功率充电"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return

        self.state = 'charging'

        # === 按照方程组顺序，清晰地执行计算 ===
        # 1. 计算电磁扭矩
        tau_mg = self._get_electromagnetic_torque(power_elec, is_charging=True)
        # 2. 计算净扭矩
        tau_net = self._get_net_torque(tau_mg)
        # 3. 更新角速度
        self._update_angular_velocity(tau_net, time_s)

        self._record_history_flywheel(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return

        self.state = 'discharging'

        # === 按照方程组顺序，清晰地执行计算 ===
        # 1. 计算电磁扭矩 (放电时为负)
        tau_mg = self._get_electromagnetic_torque(power_elec, is_charging=False)
        # 2. 计算净扭矩
        tau_net = self._get_net_torque(tau_mg)
        # 3. 更新角速度
        self._update_angular_velocity(tau_net, time_s)

        self._record_history_flywheel(time_s, -power_elec)

    def idle_loss(self, time_s):
        """计算闲置时的自放电损耗"""
        self.state = 'idle'

        # 闲置时，电磁扭矩为0
        tau_mg = 0
        # 净扭矩只有损耗扭矩
        tau_net = self._get_net_torque(tau_mg)
        # 更新角速度
        self._update_angular_velocity(tau_net, time_s)

        self._record_history_flywheel(time_s, 0)

    def _record_history_flywheel(self, time_delta, power):
        """记录飞轮特有的历史数据"""
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.angular_vel_history.append(self.omega)


# --- 单元测试用的示例函数 (可选) ---
def simulate_flywheel_test():
    flywheel = FlywheelModel(initial_soc=0.5)
    print(f"Initial SOC: {flywheel.get_soc():.2f}, Initial Omega: {flywheel.omega:.2f} rad/s")
    flywheel.charge(1e6, 60)
    print(f"After charging, SOC: {flywheel.get_soc():.2f}, Omega: {flywheel.omega:.2f} rad/s")
    flywheel.idle_loss(30)
    print(f"After idling, SOC: {flywheel.get_soc():.2f}, Omega: {flywheel.omega:.2f} rad/s")
    flywheel.discharge(1.5e6, 40)
    print(f"After discharging, SOC: {flywheel.get_soc():.2f}, Omega: {flywheel.omega:.2f} rad/s")


if __name__ == "__main__":
    simulate_flywheel_test()