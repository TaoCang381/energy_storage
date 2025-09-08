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
                 id,  # <--- 标准接口参数
                 dt_s,  # <--- 标准接口参数
                 initial_soc=0.5,
                 upper_reservoir_volume_m3=1.835e7,
                 effective_head_m=400,
                 turbine_rated_power_mw=300.0,  # 额定功率, 单位 MW
                 pump_rated_power_mw=300.0,  # 额定功率, 单位 MW
                 turbine_efficiency=0.9,
                 pump_efficiency=0.9,
                 soc_upper_limit=0.98,
                 soc_lower_limit=0.02,
                 om_cost_per_mwh=5  # 元/MWh
                 ):

        # --- 关键改动 3: 调用父类的构造函数 ---
        super().__init__(id, dt_s)

        # --- 关键改动 4: 将参数赋值给父类中的标准属性 ---
        self.soc = initial_soc
        self.power_m_w = turbine_rated_power_mw  # 以发电功率作为额定功率
        # 额定容量 MWh = V * rho * g * h / 3.6e9
        self.capacity_mwh = upper_reservoir_volume_m3 * WATER_DENSITY_KG_M3 * GRAVITY_G * effective_head_m / 3.6e9
        self.efficiency = np.sqrt(turbine_efficiency * pump_efficiency)
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh

        # --- 保留PHS特有的物理参数 ---
        self.V_ur_max = upper_reservoir_volume_m3
        self.V_ur_min = self.V_ur_max * self.soc_min
        self.h_eff = effective_head_m
        self.P_gen_rated_w = self.power_m_w * 1e6  # 内部计算仍使用瓦特
        self.P_pump_rated_w = pump_rated_power_mw * 1e6
        self.eta_gen = turbine_efficiency
        self.eta_pump = pump_efficiency

        # --- 核心状态变量：上水库水量 V_ur ---
        # 基于可用水量范围计算当前水量
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