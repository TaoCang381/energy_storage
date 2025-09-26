# file: Medium_power_density_group/electrochemical_energy_storage.py (SOH简化版 V2.0)

import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_storage_model import BaseStorageModel


class ElectrochemicalEnergyStorage(BaseStorageModel):
    """
    通用电化学储能 (EES) 模型 (简化版)
    - 移除了SOH衰减模型，严格对应您截图中的简化公式。
    - 参数已按基准表进行校准。
    """

    def __init__(self,
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心参数，与基准表和简化理论对齐 ---
                 rated_power_mw=20.0,
                 rated_capacity_mwh=40.0,
                 charge_efficiency=0.95,
                 discharge_efficiency=0.95,
                 om_cost_per_mwh=150,
                 soc_upper_limit=0.9,
                 soc_lower_limit=0.1
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        self.efficiency = np.sqrt(charge_efficiency * discharge_efficiency)
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh

        # 3. 保留EES特有的参数
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency

        # 4. 初始化核心状态变量：储存的能量 E_ees (单位: MWh)
        # 能量 = 容量 * SOC
        self.E_ees_mwh = self.capacity_mwh * self.soc
        self.state = 'idle'


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
            # 零功率表示闲置 (无损耗)
            self.idle_loss(self.dt_s)


    def get_soc(self):
        """根据储存的能量计算并更新SOC"""
        if self.capacity_mwh > 1e-6:
            self.soc = self.E_ees_mwh / self.capacity_mwh
        else:
            self.soc = 0
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (单位: W)"""
        if self.get_soc() >= self.soc_max:
            return 0
        return self.power_m_w * 1e6

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (单位: W)"""
        if self.get_soc() <= self.soc_min:
            return 0
        return self.power_m_w * 1e6

    def charge(self, power_elec_w, time_s):
        """按指定电功率充电，对应动态方程"""
        # 确认充电功率不超过限制
        power_elec_w = min(power_elec_w, self.get_available_charge_power())
        if power_elec_w <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'

        # E(t) = E(t-dt) + P_ch * eta_ch * dt
        # 单位转换: 功率(W) -> (MW), 时间(s) -> (h)
        power_elec_mw = power_elec_w / 1e6
        time_h = time_s / 3600.0
        delta_energy_mwh = power_elec_mw * self.eta_ch * time_h

        # 更新储能，并确保不超过SOC上限对应的能量
        self.E_ees_mwh += delta_energy_mwh
        self.E_ees_mwh = min(self.E_ees_mwh, self.capacity_mwh * self.soc_max)

    def discharge(self, power_elec_w, time_s):
        """按指定电功率放电，对应动态方程"""
        # 确认放电功率不超过限制
        power_elec_w = min(power_elec_w, self.get_available_discharge_power())
        if power_elec_w <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'

        # E(t) = E(t-dt) - (P_dis / eta_dis) * dt
        # 单位转换: 功率(W) -> (MW), 时间(s) -> (h)
        power_elec_mw = power_elec_w / 1e6
        time_h = time_s / 3600.0
        delta_energy_mwh = (power_elec_mw / self.eta_dis) * time_h

        # 更新储能，并确保不低于SOC下限对应的能量
        self.E_ees_mwh -= delta_energy_mwh
        self.E_ees_mwh = max(self.E_ees_mwh, self.capacity_mwh * self.soc_min)

    def idle_loss(self, time_s):
        """简化模型，暂不考虑自放电"""
        self.state = 'idle'
        pass


# --- 单元测试代码 (已简化) ---
if __name__ == "__main__":
    ees = ElectrochemicalEnergyStorage(id='ees_test', dt_s=3600, initial_soc=0.5)

    print(f"EES Initialized. Rated Power: {ees.power_m_w} MW, Rated Capacity: {ees.capacity_mwh} MWh")
    print(f"Initial SOC: {ees.get_soc():.3f}, Initial Energy: {ees.E_ees_mwh:.2f} MWh\n")

    # 模拟充电: 以10MW功率充电1小时
    charge_power = 10e6
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    ees.update_state(-charge_power)  # 充电
    print(f"After charging, SOC: {ees.get_soc():.3f}, Energy: {ees.E_ees_mwh:.2f} MWh\n")

    # 模拟放电: 以15MW功率放电1小时
    discharge_power = 15e6
    print(f"--- Discharging with {discharge_power / 1e6} MW for 1h ---")
    ees.update_state(discharge_power)  # 放电
    print(f"After discharging, SOC: {ees.get_soc():.3f}, Energy: {ees.E_ees_mwh:.2f} MWh\n")