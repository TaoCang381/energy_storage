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

    # 请将此函数完整复制并替换掉 flywheel_simulation.py 中旧的 __init__ 函数

    def __init__(self,
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 rated_power_mw=5.0,
                 rated_capacity_mwh=0.25,
                 charge_efficiency=0.98,
                 discharge_efficiency=0.98,
                 om_cost_per_mwh=200,

                 # --- 物理特性参数（可选择性提供，或使用默认值） ---
                 # 设定一个典型的最高转速，其他物理参数将由此推算
                 max_angular_vel_rpm=15000,
                 # 设定一个典型的最低与最高转速比
                 min_to_max_vel_ratio=0.3
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        self.efficiency = np.sqrt(charge_efficiency * discharge_efficiency)
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.om_cost_per_mwh = om_cost_per_mwh

        # 3. 根据顶层参数，反向推算内部物理参数
        # 这样做的好处是，我们的模型严格匹配了我们想要的功率和容量
        self.rated_power_w = self.power_m_w * 1e6
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency

        # 将转速从 RPM (转/分钟) 转换为 rad/s
        self.omega_max = max_angular_vel_rpm * (2 * math.pi) / 60
        self.omega_min = self.omega_max * min_to_max_vel_ratio

        # 核心推算：根据能量公式 E = 0.5 * J * (w_max^2 - w_min^2)，反算转动惯量 J
        # E的单位是焦耳, 1 MWh = 3.6e9 J
        energy_joules = self.capacity_mwh * 3.6e9
        omega_range_sq = self.omega_max ** 2 - self.omega_min ** 2
        if omega_range_sq <= 1e-6:
            self.J = 0
        else:
            self.J = 2 * energy_joules / omega_range_sq

        # 核心推算：根据功率公式 P = T * w, 反算额定扭矩 T
        # 额定扭矩必须足以在最低转速下也能提供额定功率
        if self.omega_min > 1e-3:
            self.rated_torque_mg = self.rated_power_w / self.omega_min
        else:
            self.rated_torque_mg = 0

        # 设定一个合理的待机损耗，例如占额定功率的0.1%
        # 损耗转矩 T_loss = kf * w, 损耗功率 P_loss = kf * w^2
        # 假设在平均转速下，损耗为额定功率的0.1%
        avg_omega = (self.omega_max + self.omega_min) / 2
        power_loss_w = self.rated_power_w * 0.001
        if avg_omega ** 2 > 1e-3:
            self.kf = power_loss_w / (avg_omega ** 2)
        else:
            self.kf = 0

        # 4. 初始化状态变量
        # 根据初始SOC和新的速度范围，精确计算初始角速度
        self.omega = math.sqrt(self.soc * (self.omega_max ** 2 - self.omega_min ** 2) + self.omega_min ** 2)

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