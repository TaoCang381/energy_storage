# file: high_power_density_group/flywheel_simulation.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_storage_model import BaseStorageModel


class FlywheelModel(BaseStorageModel):
    """
    飞轮储能系统模型 (动力学升级版 - 严格对应论文公式)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,  # <--- 标准接口参数
                 dt_s,  # <--- 标准接口参数
                 initial_soc=0.5,
                 moment_of_inertia_J=1000,
                 max_angular_vel=1500,
                 min_angular_vel=300,
                 rated_power_mw=4.0,  # 额定功率，单位改为MW
                 rated_torque=2000,
                 charge_efficiency=0.95,
                 discharge_efficiency=0.95,
                 friction_coeff_kf=0.1,
                 om_cost_per_mwh=20
                 ):

        # --- 关键改动 1: 调用父类的构造函数 ---
        # 它会处理 id, dt_s 的赋值, 并初始化通用的 self.soc 等
        super().__init__(id, dt_s)

        # --- 关键改动 2: 将参数赋值给父类中的标准属性 ---
        # 这使得所有储能单元的参数都可以通过统一的名称被访问
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        # 飞轮的容量是动态的，但为了统一接口，我们基于最大能量差计算一个额定容量
        self.capacity_mwh = 0.5 * moment_of_inertia_J * (max_angular_vel ** 2 - min_angular_vel ** 2) / (
            3.6e9)  # 转换为 MWh
        self.efficiency = np.sqrt(charge_efficiency * discharge_efficiency)  # 取几何平均作为综合效率
        self.soc_min = 0.1  # 通常飞轮SOC有一定限制
        self.soc_max = 0.9
        self.om_cost_per_mwh = om_cost_per_mwh

        # --- 保留飞轮特有的物理参数 ---
        self.J = moment_of_inertia_J
        self.omega_max = max_angular_vel
        self.omega_min = min_angular_vel
        self.rated_power_w = self.power_m_w * 1e6  # 内部计算仍使用瓦特
        self.rated_torque_mg = rated_torque
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency
        self.kf = friction_coeff_kf

        # --- 核心状态变量：角速度 omega ---
        # 根据初始SOC计算初始角速度
        self.omega = math.sqrt(self.soc * (self.omega_max ** 2 - self.omega_min ** 2) + self.omega_min ** 2)

        # 飞轮特有的历史记录
        self.angular_vel_history = []
        self.state = 'idle'

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

    def _get_electromagnetic_torque(self, power_elec, is_charging):
        current_omega = self.omega if self.omega > 1e-3 else 1e-3
        if is_charging:
            tau_mg = (power_elec * self.eta_ch) / current_omega
        else:
            tau_mg = - (power_elec / (current_omega * self.eta_dis))
        return np.clip(tau_mg, -self.rated_torque_mg, self.rated_torque_mg)

    def _get_loss_torque(self):
        return self.kf * self.omega

    def _get_net_torque(self, tau_mg):
        return tau_mg - self._get_loss_torque()

    def _update_angular_velocity(self, tau_net, time_s):
        self.omega += (tau_net / self.J) * time_s
        self.omega = np.clip(self.omega, self.omega_min, self.omega_max)

    # ==============================================================================
    # --- HESS标准接口实现 (charge/discharge等现在作为内部方法) ---
    # ==============================================================================

    def get_soc(self):
        omega_range_sq = self.omega_max ** 2 - self.omega_min ** 2
        if omega_range_sq <= 1e-6: return self.soc_min
        self.soc = (self.omega ** 2 - self.omega_min ** 2) / omega_range_sq
        return self.soc

    def get_available_charge_power(self):
        if self.omega >= self.omega_max: return 0
        power_limit_by_rated_p = self.rated_power_w
        power_limit_by_rated_t = (self.rated_torque_mg * self.omega) / self.eta_ch
        return min(power_limit_by_rated_p, power_limit_by_rated_t)

    def get_available_discharge_power(self):
        if self.omega <= self.omega_min: return 0
        power_limit_by_rated_p = self.rated_power_w
        power_limit_by_rated_t = (self.rated_torque_mg * self.omega) * self.eta_dis
        return min(power_limit_by_rated_p, power_limit_by_rated_t)

    def charge(self, power_elec, time_s):
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'
        tau_mg = self._get_electromagnetic_torque(power_elec, is_charging=True)
        tau_net = self._get_net_torque(tau_mg)
        self._update_angular_velocity(tau_net, time_s)

    def discharge(self, power_elec, time_s):
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'
        tau_mg = self._get_electromagnetic_torque(power_elec, is_charging=False)
        tau_net = self._get_net_torque(tau_mg)
        self._update_angular_velocity(tau_net, time_s)

    def idle_loss(self, time_s):
        self.state = 'idle'
        tau_mg = 0
        tau_net = self._get_net_torque(tau_mg)
        self._update_angular_velocity(tau_net, time_s)


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    # 测试需要提供id和dt_s
    flywheel = FlywheelModel(id='fw_test', dt_s=1, initial_soc=0.5)
    print(f"Initial SOC: {flywheel.get_soc():.3f}, Initial Omega: {flywheel.omega:.2f} rad/s")

    flywheel.update_state(-1e6)  # 充电 1MW for 1s
    print(f"After charging 1MW for 1s, SOC: {flywheel.get_soc():.3f}, Omega: {flywheel.omega:.2f} rad/s")

    flywheel.update_state(0)  # 闲置 1s
    print(f"After idling for 1s, SOC: {flywheel.get_soc():.3f}, Omega: {flywheel.omega:.2f} rad/s")

    flywheel.update_state(1.5e6)  # 放电 1.5MW for 1s
    print(f"After discharging 1.5MW for 1s, SOC: {flywheel.get_soc():.3f}, Omega: {flywheel.omega:.2f} rad/s")