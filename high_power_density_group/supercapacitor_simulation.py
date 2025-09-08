# file: high_power_density_group/supercapacitor_simulation.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel


# --- 修改区域 2: 让 Supercapacitor 继承 BaseStorageModel ---
class Supercapacitor(BaseStorageModel):
    """
    超级电容器模型 (HESS集成版 - 依据Word文档简化模型)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,  # <--- 标准接口参数
                 dt_s,  # <--- 标准接口参数
                 initial_soc=0.5,
                 capacitance_F=120000,
                 max_voltage=48,
                 min_voltage=24,
                 esr_ohm=0.005,
                 rated_power_mw=0.05,  # 额定功率, 50kW = 0.05MW
                 rated_current=1000,
                 self_discharge_rate_sigma=1e-7,
                 om_cost_per_mwh=80  # 元/MWh
                 ):

        # --- 关键改动 3: 调用父类的构造函数 ---
        super().__init__(id, dt_s)

        # --- 关键改动 4: 将参数赋值给父类中的标准属性 ---
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        # 超级电容器的额定容量 MWh = 0.5 * C * (Vmax^2 - Vmin^2) / 3.6e9
        self.capacity_mwh = 0.5 * capacitance_F * (max_voltage ** 2 - min_voltage ** 2) / (3.6e9)
        self.efficiency = 0.98  # 超级电容器效率很高，这里给一个典型值
        self.soc_min = 0.05  # 通常有电压下限保护
        self.soc_max = 0.95
        self.om_cost_per_mwh = om_cost_per_mwh

        # --- 保留超级电容器特有的物理参数 ---
        self.C_sc = capacitance_F
        self.V_max = max_voltage
        self.V_min = min_voltage
        self.R_esr = esr_ohm
        self.rated_power_w = self.power_m_w * 1e6  # 内部计算仍使用瓦特
        self.rated_current_sc = rated_current
        self.sigma = self_discharge_rate_sigma

        # --- 核心状态变量：电压 V_sc ---
        v_range_sq = self.V_max ** 2 - self.V_min ** 2
        self.V_sc = math.sqrt(self.soc * v_range_sq + self.V_min ** 2)

        self.voltage_history = []
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

    def get_soc(self):
        """根据电压计算并更新SOC (基于能量)"""
        v_range_sq = self.V_max ** 2 - self.V_min ** 2
        if v_range_sq <= 1e-6: return self.soc_min
        self.soc = (self.V_sc ** 2 - self.V_min ** 2) / v_range_sq
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (W)，依据Word文档约束"""
        if self.V_sc >= self.V_max: return 0
        power_limit_by_p = self.rated_power_w
        # 简化版：功率约束主要由额定功率决定
        return power_limit_by_p

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (W)，依据Word文档约束"""
        if self.V_sc <= self.V_min: return 0
        power_limit_by_p = self.rated_power_w
        # 简化版：功率约束主要由额定功率决定
        return power_limit_by_p

    def charge(self, power_elec, time_s):
        """按指定电功率充电"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'

        # I = P / V
        current = power_elec / self.V_sc if self.V_sc > 1e-3 else self.rated_current_sc
        current = min(current, self.rated_current_sc)

        # 考虑ESR的电压损耗，实际用于充电的电压 V_c = V_terminal - I * R
        # 简化处理：我们直接更新电容电压，认为P_elec是端子功率
        delta_v = (current * time_s) / self.C_sc
        self.V_sc += delta_v

        # 应用电压约束
        self.V_sc = min(self.V_sc, self.V_max)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'

        current = power_elec / self.V_sc if self.V_sc > 1e-3 else 0
        current = min(current, self.rated_current_sc)

        delta_v = (current * time_s) / self.C_sc
        self.V_sc -= delta_v

        self.V_sc = max(self.V_sc, self.V_min)

    def idle_loss(self, time_s):
        """计算闲置时的自放电损耗，对应公式中的 sigma 项"""
        self.state = 'idle'

        # V(t) = V(0) * e^(-sigma*t) ~= V(0) * (1 - sigma*t)
        self.V_sc *= (1 - self.sigma * time_s)


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    sc = Supercapacitor(id='sc_test', dt_s=1, initial_soc=0.5)

    total_energy_kj = 0.5 * sc.C_sc * (sc.V_max ** 2 - sc.V_min ** 2) / 1000
    print(f"Supercapacitor Initialized. Usable Energy: {total_energy_kj:.2f} kJ")
    print(f"Initial SOC: {sc.get_soc():.3f}, Initial Voltage: {sc.V_sc:.3f} V\n")

    charge_power = 40000  # 40 kW
    sc.update_state(-charge_power)  # 充电
    print(f"--- Charging with {charge_power / 1000} kW for 1s ---")
    print(f"After charging, SOC: {sc.get_soc():.3f}, Voltage: {sc.V_sc:.3f} V\n")

    sc.update_state(0)  # 闲置
    print(f"--- Idling for 1s ---")
    print(f"After idling, SOC: {sc.get_soc():.3f}, Voltage: {sc.V_sc:.3f} V\n")

    discharge_power = 50000  # 50 kW
    sc.update_state(discharge_power)  # 放电
    print(f"--- Discharging with {discharge_power / 1000} kW for 1s ---")
    print(f"After discharging, SOC: {sc.get_soc():.3f}, Voltage: {sc.V_sc:.3f} V\n")