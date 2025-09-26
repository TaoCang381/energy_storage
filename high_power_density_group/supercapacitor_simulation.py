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
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 rated_power_mw=10.0,
                 rated_capacity_mwh=0.1,
                 charge_efficiency=0.99,
                 discharge_efficiency=0.99,
                 om_cost_per_mwh=250,

                 # --- 物理特性参数（可选择性提供，或使用默认值）---
                 # 设定一个典型的最高工作电压
                 max_voltage=500,
                 # 设定一个典型的最低与最高电压比
                 min_to_max_voltage_ratio=0.5
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        # 对于超级电容，其充放电库仑效率接近1，总效率主要受ESR损耗影响
        # P_loss = I^2 * R. P_out = V*I - I^2*R. eta = P_out / (V*I) = 1 - I*R/V
        # 这是一个动态值，我们这里用一个较高的典型值
        self.efficiency = np.sqrt(charge_efficiency * discharge_efficiency)
        self.soc_min = 0.05
        self.soc_max = 0.95
        self.om_cost_per_mwh = om_cost_per_mwh

        # 3. 根据顶层参数，反向推算内部物理参数
        self.rated_power_w = self.power_m_w * 1e6
        self.V_max = max_voltage
        self.V_min = self.V_max * min_to_max_voltage_ratio

        # 核心推算：根据能量公式 E = 0.5 * C * (V_max^2 - V_min^2)，反算电容值 C
        # E的单位是焦耳, 1 MWh = 3.6e9 J
        energy_joules = self.capacity_mwh * 3.6e9
        voltage_range_sq = self.V_max ** 2 - self.V_min ** 2
        if voltage_range_sq <= 1e-6:
            self.C_sc = 0
        else:
            self.C_sc = 2 * energy_joules / voltage_range_sq

        # 核心推算：根据功率损耗 P_loss = (P_rated/V)^2 * R_esr 来估算ESR
        # 假设在额定功率、平均电压下，损耗为 (1 - 效率) * P_rated
        avg_voltage = (self.V_max + self.V_min) / 2
        power_loss_w = self.rated_power_w * (1 - self.efficiency)
        if avg_voltage > 1e-3:
            # P_loss = I^2 * R = (P_rated/V_avg)^2 * R
            self.R_esr = power_loss_w * (avg_voltage / self.rated_power_w) ** 2
        else:
            self.R_esr = 0.01  # 给一个默认的小值

        # 核心推算：根据 P = I * V 反算额定电流
        # 额定电流应足以在最低电压下也能提供额定功率
        if self.V_min > 1e-3:
            self.rated_current_sc = self.rated_power_w / self.V_min
        else:
            self.rated_current_sc = 0

        # 设定一个合理的自放电率，例如每天损失5%的能量
        # 简化模型 V(t) ~= V(0) * (1 - sigma*t)
        # 假设一天后电压下降5%， t = 86400s
        daily_loss_ratio = 0.05
        self.sigma = daily_loss_ratio / 86400

        # 4. 初始化状态变量
        # 根据初始SOC和新的电压范围，精确计算初始电压
        self.V_sc = math.sqrt(self.soc * (self.V_max ** 2 - self.V_min ** 2) + self.V_min ** 2)

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