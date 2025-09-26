# file: low_power_density_group/pumped_storage_simulation.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel

# --- 物理常数 ---
WATER_DENSITY_KG_M3 = 1000
GRAVITY_G = 9.81


# --- 修改区域 2: 让 PHS 继承 BaseStorageModel ---
class PumpedHydroStorage(BaseStorageModel):
    """
    抽水蓄能 (PHS) 模型 (HESS集成版 - 简化版)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 rated_power_mw=100.0,
                 rated_capacity_mwh=800.0,
                 turbine_efficiency=0.92,  # 发电效率
                 pump_efficiency=0.90,  # 抽水效率
                 om_cost_per_mwh=10,

                 # --- 其他关键参数 ---
                 soc_upper_limit=0.95,
                 soc_lower_limit=0.1,
                 # 设定一个典型的有效水头高度
                 effective_head_m=400
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = rated_power_mw  # 以发电功率作为额定功率
        self.capacity_mwh = rated_capacity_mwh
        self.efficiency = np.sqrt(turbine_efficiency * pump_efficiency)
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh

        # 3. 根据顶层参数，反向推算内部物理参数
        self.eta_gen = turbine_efficiency
        self.eta_pump = pump_efficiency
        self.h_eff = effective_head_m
        self.P_gen_rated_w = self.power_m_w * 1e6
        # 假设抽水功率与发电功率相同
        self.P_pump_rated_w = self.power_m_w * 1e6

        # 核心推算：根据能量公式 E = V * rho * g * h, 反算上水库的总可用容积
        # E的单位是焦耳, 1 MWh = 3.6e9 J
        energy_joules = self.capacity_mwh * 3.6e9
        denominator = WATER_DENSITY_KG_M3 * GRAVITY_G * self.h_eff
        if denominator > 1e-6:
            # 这是水库在SOC_min和SOC_max之间的有效容积
            usable_volume_m3 = energy_joules / denominator
            # 从可用容积反推总容积
            self.V_ur_max = usable_volume_m3 / (self.soc_max - self.soc_min)
        else:
            self.V_ur_max = 0

        self.V_ur_min = self.V_ur_max * self.soc_min

        # 4. 初始化核心状态变量：上水库水量 V_ur
        self.V_ur_m3 = self.V_ur_min + self.soc * (self.V_ur_max - self.V_ur_min)

        self.volume_history = []
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
            # 正功率表示放电 (发电)
            self.discharge(dispatch_power_w, self.dt_s)
        elif dispatch_power_w < 0:
            # 负功率表示充电 (抽水)
            self.charge(abs(dispatch_power_w), self.dt_s)
        else:
            # 零功率表示闲置
            self.idle_loss(self.dt_s)

    # ==============================================================================
    # --- 模型核心物理方法 (完全保留您原有的代码) ---
    # ==============================================================================

    def get_soc(self):
        """根据可用水量计算并更新SOC"""
        usable_volume_range = self.V_ur_max - self.V_ur_min
        if usable_volume_range > 1e-6:
            self.soc = (self.V_ur_m3 - self.V_ur_min) / usable_volume_range
        else:
            self.soc = self.soc_min
        return self.soc

    def _power_to_flow(self, power_w, is_charging):
        """根据功率计算流量"""
        denominator = WATER_DENSITY_KG_M3 * GRAVITY_G * self.h_eff
        if denominator < 1e-6: return 0
        if is_charging:
            return (power_w * self.eta_pump) / denominator
        else:
            return power_w / (denominator * self.eta_gen)

    def _update_volume(self, flow_rate_m3s, time_s, is_charging):
        """根据流量更新水量"""
        delta_volume = flow_rate_m3s * time_s
        if is_charging:
            self.V_ur_m3 += delta_volume
        else:
            self.V_ur_m3 -= delta_volume
        # 应用水量约束
        self.V_ur_m3 = np.clip(self.V_ur_m3, self.V_ur_min, self.V_ur_max)

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (W)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_pump_rated_w

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (W)"""
        if self.get_soc() <= self.soc_min: return 0
        return self.P_gen_rated_w

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (抽水)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'
        flow_rate = self._power_to_flow(power_elec, is_charging=True)
        self._update_volume(flow_rate, time_s, is_charging=True)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'
        flow_rate = self._power_to_flow(power_elec, is_charging=False)
        self._update_volume(flow_rate, time_s, is_charging=False)

    def idle_loss(self, time_s):
        """简化模型，抽水蓄能闲置时无损耗"""
        self.state = 'idle'
        # 水量不发生变化


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    phs = PumpedHydroStorage(id='phs_test', dt_s=3600, initial_soc=0.5)

    print(f"PHS Initialized. Rated Power: {phs.power_m_w} MW, Usable Capacity: {phs.capacity_mwh:.2f} MWh")
    print(f"Initial SOC: {phs.get_soc():.3f}\n")

    charge_power = 200e6  # 200 MW
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    phs.update_state(-charge_power)  # 充电
    print(f"After charging, SOC: {phs.get_soc():.3f}\n")

    discharge_power = 300e6  # 300 MW
    print(f"--- Discharging with {discharge_power / 1e6} MW for 1h ---")
    phs.update_state(discharge_power)  # 放电
    print(f"After discharging, SOC: {phs.get_soc():.3f}\n")