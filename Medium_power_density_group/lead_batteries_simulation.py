import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示，确保图表中的中文标签能正确显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


class LeadAcidBattery(EnergyStorageUnit):
    """
    铅酸电池模型 (HESS集成版)
    特点：
    1. 引入了与放电深度(DoD)相关的非线性SOH衰减模型。
    2. 实现了与SOC相关的可变充电效率。
    3. 参数反映了铅酸电池低成本、低能量密度的特性。
    """

    def __init__(self,
                 # --- 基础电化学规格 ---
                 nominal_capacity_ah=100,  # 额定安时容量 (Ah)
                 nominal_voltage=12.6,  # 额定电压 (V)
                 initial_internal_resistance=0.01,  # 初始内阻 (Ohm)

                 # --- 运行限制 ---
                 max_c_rate_charge=0.2,  # 最大充电倍率 (C)
                 max_c_rate_discharge=0.5,  # 最大放电倍率 (C)
                 soc_upper_limit=0.95,  # SOC运行上限
                 soc_lower_limit=0.2,  # SOC运行下限

                 # --- HESS集成 & 其他参数 ---
                 ess_id="lead_acid_battery_01",  # 唯一标识符: 在混合储能系统中区分不同的设备。
                 initial_soh=1.0,  # 初始健康状态: 1.0代表全新出厂状态。
                 initial_soc=0.8,  # 初始荷电状态: 铅酸电池通常保持在较高SOC以备不时之需。
                 cost_per_kwh=0.01,  # 度电成本: 铅酸电池的核心优势之一，运行成本非常低廉。

                 # OCV-SOC模型参数: 用于拟合铅酸电池开路电压与SOC关系的经验公式参数
                 ocv_params={'P1': 12.0, 'P2': 0.8, 'P3': 0.1},

                 # DoD-Cycle Life 衰减模型 (核心): 定义了在不同放电深度(DoD)下的总循环寿命
                 # 这是铅酸电池最关键的特性，深放会急剧减少寿命。
                 # Key: DoD (0-1), Value: 在该DoD下的总循环寿命
                 cycle_life_model={
                     0.1: 5000,  # 10% DoD: 可循环5000次
                     0.3: 1500,  # 30% DoD: 可循环1500次
                     0.5: 600,  # 50% DoD: 可循环600次
                     0.8: 300,  # 80% DoD: 可循环300次
                     1.0: 200  # 100% DoD: 可循环200次
                 }
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh
        self.state_of_charge = initial_soc
        self.cost_per_kwh = cost_per_kwh

        # --- 规格与模型参数 ---
        self.nominal_capacity_ah = nominal_capacity_ah
        self.nominal_voltage = nominal_voltage
        self.initial_internal_resistance = initial_internal_resistance
        self.max_c_rate_charge = max_c_rate_charge
        self.max_c_rate_discharge = max_c_rate_discharge
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit
        self.ocv_params = ocv_params
        self.cycle_life_model = cycle_life_model

        # --- 实时工作参数 (会随SOH变化) ---
        self.capacity_ah = self.nominal_capacity_ah * self.state_of_health
        self.internal_resistance = self.initial_internal_resistance / self.state_of_health

        # --- 状态与历史记录 ---
        self.state = 'idle'
        self.last_cycle_soc_min = self.state_of_charge
        self.last_cycle_soc_max = self.state_of_charge
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.soh_history = []
        self.voltage_history = []

    def _get_ocv(self, soc):
        """简化OCV模型: V_ocv = P1 + P2*soc + P3*ln(soc)"""
        soc = max(min(soc, 0.999), 0.001)
        p = self.ocv_params
        return p['P1'] + p['P2'] * soc + p['P3'] * math.log(soc)

    def _get_charge_efficiency(self, soc):
        """计算与SOC相关的充电效率，SOC越高，副反应越多，效率越低"""
        return 0.98 - 0.15 * soc

    def get_soc(self):
        return self.state_of_charge

    def get_available_charge_power(self):
        soc = self.state_of_charge
        if soc >= self.soc_upper_limit: return 0
        max_current_by_c_rate = self.max_c_rate_charge * self.capacity_ah
        ocv = self._get_ocv(soc)
        return min(max_current_by_c_rate * self.nominal_voltage,
                   self.max_c_rate_charge * self.capacity_ah * ocv)

    def get_available_discharge_power(self):
        soc = self.state_of_charge
        if soc <= self.soc_lower_limit: return 0
        max_current_by_c_rate = self.max_c_rate_discharge * self.capacity_ah
        ocv = self._get_ocv(soc)
        return min(max_current_by_c_rate * self.nominal_voltage,
                   self.max_c_rate_discharge * self.capacity_ah * ocv)

    def charge(self, power, time_s):
        available_power = self.get_available_charge_power()
        power = min(power, available_power)
        if power <= 0: return
        if self.state == 'discharging':
            self._update_soh()
        self.state = 'charging'
        ocv = self._get_ocv(self.state_of_charge)
        current = power / ocv
        time_h = time_s / 3600.0
        delta_ah = current * time_h
        effective_delta_ah = delta_ah * self._get_charge_efficiency(self.state_of_charge)
        delta_soc = effective_delta_ah / self.capacity_ah
        self.state_of_charge += delta_soc
        self.state_of_charge = min(self.state_of_charge, self.soc_upper_limit)
        self.last_cycle_soc_max = max(self.last_cycle_soc_max, self.state_of_charge)
        self._record_history(time_s, power, ocv + current * self.internal_resistance)

    def discharge(self, power, time_s):
        available_power = self.get_available_discharge_power()
        power = min(power, available_power)
        if power <= 0: return
        self.state = 'discharging'
        ocv = self._get_ocv(self.state_of_charge)
        current = power / ocv
        time_h = time_s / 3600.0
        delta_ah = current * time_h
        delta_soc = delta_ah / self.capacity_ah
        self.state_of_charge -= delta_soc
        self.state_of_charge = max(self.state_of_charge, self.soc_lower_limit)
        self.last_cycle_soc_min = min(self.last_cycle_soc_min, self.state_of_charge)
        self._record_history(time_s, -power, ocv - current * self.internal_resistance)

    def idle_loss(self, time_s):
        self.state = 'idle'
        daily_loss = 0.003
        self.state_of_charge -= (daily_loss / (24 * 3600)) * time_s
        self._record_history(time_s, 0, self._get_ocv(self.state_of_charge))

    def _update_soh(self):
        """核心方法：根据上一个循环的DoD来更新SOH"""
        dod = self.last_cycle_soc_max - self.last_cycle_soc_min
        if dod < 0.01: return
        life_key = min([k for k in self.cycle_life_model.keys() if k >= dod], default=1.0)
        total_cycles_at_this_dod = self.cycle_life_model[life_key]
        soh_loss_per_cycle = 1.0 / total_cycles_at_this_dod
        self.state_of_health -= soh_loss_per_cycle
        self.state_of_health = max(0, self.state_of_health)
        self.capacity_ah = self.nominal_capacity_ah * self.state_of_health
        self.internal_resistance = self.initial_internal_resistance / self.state_of_health
        self.last_cycle_soc_min = self.state_of_charge
        self.last_cycle_soc_max = self.state_of_charge

    def _record_history(self, time_delta, power, voltage):
        """内部方法：记录历史数据"""
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(self.state_of_charge)
        self.soh_history.append(self.state_of_health)
        self.voltage_history.append(voltage)

    def plot_performance(self):
        """绘制所有性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'铅酸电池 ({self.id}) 性能曲线', fontsize=16)

        time_h = [t / 3600.0 for t in self.time_history]

        axes[0].plot(time_h, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].plot(time_h, self.soh_history, 'r-', lw=2, label='SOH')
        axes[1].set_title('健康状态 (SOH) 变化');
        axes[1].set_ylabel('SOH');
        axes[1].grid(True);
        axes[1].legend()
        if len(self.soh_history) > 1 and min(self.soh_history) < 0.999:
            axes[1].set_ylim(min(self.soh_history) * 0.99, 1.001)
        else:
            axes[1].set_ylim(0.9, 1.001)

        axes[2].plot(time_h, self.voltage_history, 'c-', lw=2, label='终端电压')
        axes[2].set_title('终端电压变化');
        axes[2].set_ylabel('电压 (V)');
        axes[2].grid(True);
        axes[2].legend()

        axes[3].plot(time_h, [p / 1000 for p in self.power_history], 'g-', lw=2, label='实时功率')
        axes[3].set_title('功率变化');
        axes[3].set_ylabel('功率 (kW)');
        axes[3].grid(True);
        axes[3].legend()
        axes[3].set_xlabel('时间 (小时)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


def simulate_hess_with_lead_acid():
    """一个简化的示例，演示铅酸电池用于后备电源或小型离网系统"""
    battery = LeadAcidBattery(initial_soc=0.9)

    time_steps_h = np.arange(0, 24 * 3, 0.5)
    base_load = 200
    peak_load = np.zeros_like(time_steps_h)
    for day in range(3):
        start_hour = day * 24 + 18
        peak_load[(time_steps_h >= start_hour) & (time_steps_h < start_hour + 2)] = 800
    load = base_load + peak_load
    pv_generation = np.maximum(0, 1000 * np.sin((time_steps_h % 24 - 6) * np.pi / 12))
    net_power = load - pv_generation

    print(f"--- 开始模拟，铅酸电池初始SOC: {battery.get_soc():.2f} ---")
    battery._record_history(0, 0, battery._get_ocv(battery.state_of_charge))

    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        demand = net_power[i]

        if demand > 0:
            power = min(demand, battery.get_available_discharge_power())
            battery.discharge(power, dt_s)
        elif demand < 0:
            power = min(abs(demand), battery.get_available_charge_power())
            battery.charge(power, dt_s)
        else:
            battery.idle_loss(dt_s)

    print(f"--- 模拟结束，最终SOH: {battery.state_of_health:.4f} ---")
    battery.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_lead_acid()