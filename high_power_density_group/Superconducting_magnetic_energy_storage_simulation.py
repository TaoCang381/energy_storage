# file: high_power_density_group/Superconducting_magnetic_energy_storage_simulation.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel


# --- 修改区域 2: 让 SMES 继承 BaseStorageModel ---
class SuperconductingMagneticEnergyStorage(BaseStorageModel):
    """
    超导磁储能 (SMES) 模型 (物理升级版 - 严格对应论文公式)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 rated_power_mw=8.0,
                 rated_capacity_mwh=0.15,
                 # SMES的PCS效率非常高
                 pcs_efficiency=0.98,
                 om_cost_per_mwh=300,

                 # --- 物理特性参数（可选择性提供，或使用默认值）---
                 # 设定一个典型的最大工作电流
                 max_current_A=2500.0,
                 # 设定一个典型的最低与最高电流比
                 min_to_max_current_ratio=0.1
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        self.efficiency = pcs_efficiency  # PCS效率作为综合效率
        self.soc_min = 0.05
        self.soc_max = 0.95
        self.om_cost_per_mwh = om_cost_per_mwh

        # 3. 根据顶层参数，反向推算内部物理参数
        self.rated_power_w = self.power_m_w * 1e6
        self.eta_pcs = pcs_efficiency
        self.I_max = max_current_A
        self.I_min = self.I_max * min_to_max_current_ratio

        # 核心推算：根据能量公式 E = 0.5 * L * (I_max^2 - I_min^2)，反算电感L
        # E的单位是焦耳, 1 MWh = 3.6e9 J
        energy_joules = self.capacity_mwh * 3.6e9
        current_range_sq = self.I_max ** 2 - self.I_min ** 2
        if current_range_sq <= 1e-6:
            self.L_smes = 0
        else:
            self.L_smes = 2 * energy_joules / current_range_sq

        # 核心推算：制冷功率通常是额定功率的一个很小的比例，例如0.5%
        self.P_cryo_w = self.rated_power_w * 0.005

        # 设定一个典型的PCS最大电压
        self.V_pcs_max = 3000

        # 4. 初始化状态变量
        # 根据初始SOC和新的电流范围，精确计算初始电流
        self.I_smes = math.sqrt(self.soc * (self.I_max ** 2 - self.I_min ** 2) + self.I_min ** 2)

        self.current_history = []
        self.state = 'idle'

    # ==============================================================================
    # --- 新增：核心标准接口 update_state ---
    # ==============================================================================
    def update_state(self, dispatch_power_w):
        """
        根据调度指令（单位：W）更新储能状态。
        注意：传入的功率是PCS的净功率，不包含制冷损耗。
        """
        # SMES的总功率消耗是 PCS功率 + 制冷功率，调度指令应减去制冷功率
        net_dispatch_power_w = dispatch_power_w - self.P_cryo_w

        if net_dispatch_power_w > 0:
            # 正功率表示放电
            self.discharge(net_dispatch_power_w, self.dt_s)
        elif net_dispatch_power_w < 0:
            # 负功率表示充电
            self.charge(abs(net_dispatch_power_w), self.dt_s)
        else:
            # 仅有制冷损耗
            self.idle_loss(self.dt_s)

    # ==============================================================================
    # --- 模型核心物理方法 (完全保留您原有的代码) ---
    # ==============================================================================

    def get_soc(self):
        """根据线圈电流计算并更新SOC (基于能量)"""
        i_range_sq = self.I_max ** 2 - self.I_min ** 2
        if i_range_sq <= 1e-6: return self.soc_min
        self.soc = (self.I_smes ** 2 - self.I_min ** 2) / i_range_sq
        return self.soc

    def _get_pcs_voltage(self, power_elec_net, is_charging):
        """根据净电功率指令计算PCS施加的电压"""
        current_I = self.I_smes if self.I_smes > 1e-3 else 1e-3
        if is_charging:
            voltage = (power_elec_net * self.eta_pcs) / current_I
        else:
            voltage = - (power_elec_net / (current_I * self.eta_pcs))
        return np.clip(voltage, -self.V_pcs_max, self.V_pcs_max)

    def _update_current(self, V_pcs, time_s):
        """根据PCS电压更新线圈电流"""
        self.I_smes += (V_pcs / self.L_smes) * time_s
        self.I_smes = np.clip(self.I_smes, self.I_min, self.I_max)

    def get_available_charge_power(self):
        """获取当前可用的充电功率 (W), 这是PCS的净功率"""
        if self.I_smes >= self.I_max: return 0
        return self.rated_power_w

    def get_available_discharge_power(self):
        """获取当前可用的放电功率 (W), 这是PCS的净功率"""
        if self.I_smes <= self.I_min: return 0
        return self.rated_power_w

    def charge(self, power_elec, time_s):
        """按指定净电功率充电"""
        power_elec_net = min(power_elec, self.get_available_charge_power())
        if power_elec_net <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'
        V_pcs = self._get_pcs_voltage(power_elec_net, is_charging=True)
        self._update_current(V_pcs, time_s)

    def discharge(self, power_elec, time_s):
        """按指定净电功率放电"""
        power_elec_net = min(power_elec, self.get_available_discharge_power())
        if power_elec_net <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'
        V_pcs = self._get_pcs_voltage(power_elec_net, is_charging=False)
        self._update_current(V_pcs, time_s)

    def idle_loss(self, time_s):
        """闲置时，线圈电流无损耗"""
        self.state = 'idle'
        self._update_current(0, time_s)


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    smes = SuperconductingMagneticEnergyStorage(id='smes_test', dt_s=0.1, initial_soc=0.5)

    max_energy_mj = 0.5 * smes.L_smes * smes.I_max ** 2 / 1e6
    print(f"SMES Initialized. Max Energy: {max_energy_mj:.2f} MJ")
    print(f"Initial SOC: {smes.get_soc():.3f}, Initial Current: {smes.I_smes:.2f} A\n")

    charge_power_grid = 4e6  # 从电网吸收 4 MW
    smes.update_state(-charge_power_grid)  # 充电指令为负
    print(f"--- Commanding charge with {charge_power_grid / 1e6} MW from grid for {smes.dt_s}s ---")
    print(f"After charging, SOC: {smes.get_soc():.3f}, Current: {smes.I_smes:.2f} A\n")

    smes.update_state(0)  # 闲置指令
    print(f"--- Commanding idle for {smes.dt_s}s ---")
    print(f"After idling, SOC: {smes.get_soc():.3f}, Current: {smes.I_smes:.2f} A\n")

    discharge_power_grid = 5e6  # 向电网注入 5 MW
    smes.update_state(discharge_power_grid)  # 放电指令为正
    print(f"--- Commanding discharge with {discharge_power_grid / 1e6} MW to grid for {smes.dt_s}s ---")
    print(f"After discharging, SOC: {smes.get_soc():.3f}, Current: {smes.I_smes:.2f} A\n")