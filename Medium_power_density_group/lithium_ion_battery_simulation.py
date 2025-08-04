import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class LithiumIonBattery(EnergyStorageUnit):
    """
    锂离子电池模型 (HESS集成版)
    包含非线性OCV-SOC关系、内阻、SOH衰减等关键电化学特性。
    """

    def __init__(self,
                 # --- 基础电化学规格 ---
                 nominal_capacity_ah=100,  # 额定安时容量 (Ah)
                 nominal_voltage=3.7,  # 额定电压 (V)
                 initial_internal_resistance=0.002,  # 初始内阻 (Ohm)

                 # --- 运行限制 ---
                 max_c_rate_charge=1.0,  # 最大充电倍率 (C)
                 max_c_rate_discharge=2.0,  # 最大放电倍率 (C)
                 soc_upper_limit=0.9,  # SOC运行上限
                 soc_lower_limit=0.1,  # SOC运行下限
                 max_voltage_limit=4.2,  # 终端电压上限
                 min_voltage_limit=2.8,  # 终端电压下限

                 # --- HESS集成 & 其他参数 ---
                 ess_id="li_ion_battery_01",
                 initial_soh=1.0,
                 initial_soc=0.5,
                 coulombic_efficiency=0.99,  # 库伦效率
                 # 简化的OCV-SOC曲线模型参数 (示例，针对某款NCM电池)
                 # V(soc) = P1 - P2/soc - P3*soc + P4*ln(soc)
                 ocv_params={'P1': 3.5, 'P2': 0.05, 'P3': 0.1, 'P4': 0.15},
                 # 简化的SOH衰减模型: 每当总吞吐量达到全容量的2000倍时，SOH下降1%
                 cycles_for_1pct_soh_loss=2000
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh
        self.state_of_charge = initial_soc
        self.coulombic_efficiency = coulombic_efficiency

        # --- 规格参数 ---
        self.nominal_capacity_ah = nominal_capacity_ah
        self.initial_internal_resistance = initial_internal_resistance
        self.max_c_rate_charge = max_c_rate_charge
        self.max_c_rate_discharge = max_c_rate_discharge
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit
        self.max_voltage_limit = max_voltage_limit
        self.min_voltage_limit = min_voltage_limit
        self.ocv_params = ocv_params

        # --- SOH 衰减模型参数 ---
        self.total_ah_throughput_for_soh_loss = self.nominal_capacity_ah * cycles_for_1pct_soh_loss
        self.cumulative_ah_throughput = 0

        # --- 实时工作参数 (受SOH影响) ---
        self.capacity_ah = self.nominal_capacity_ah * self.state_of_health
        self.internal_resistance = self.initial_internal_resistance / self.state_of_health

        # --- 状态与历史记录 ---
        self.state = 'idle'
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.soh_history = []
        self.voltage_history = []

    def _get_ocv(self, soc):
        """内部方法：根据SOC计算开路电压 (OCV)"""
        soc = max(min(soc, 0.999), 0.001)
        p = self.ocv_params
        return p['P1'] - p['P2'] / soc - p['P3'] * soc + p['P4'] * math.log(soc)

    def get_soc(self):
        return self.state_of_charge

    def get_available_charge_power(self):
        soc = self.state_of_charge
        if soc >= self.soc_upper_limit: return 0
        max_current_by_c_rate = self.max_c_rate_charge * self.capacity_ah
        ocv = self._get_ocv(soc)
        max_current_by_voltage = (self.max_voltage_limit - ocv) / self.internal_resistance
        charge_current = max(0, min(max_current_by_c_rate, max_current_by_voltage))
        charge_voltage = ocv + charge_current * self.internal_resistance
        return charge_current * charge_voltage

    def get_available_discharge_power(self):
        soc = self.state_of_charge
        if soc <= self.soc_lower_limit: return 0
        max_current_by_c_rate = self.max_c_rate_discharge * self.capacity_ah
        ocv = self._get_ocv(soc)
        max_current_by_voltage = (ocv - self.min_voltage_limit) / self.internal_resistance
        discharge_current = max(0, min(max_current_by_c_rate, max_current_by_voltage))
        discharge_voltage = ocv - discharge_current * self.internal_resistance
        return discharge_current * discharge_voltage

    def charge(self, power, time_s):
        available_power = self.get_available_charge_power()
        power = min(power, available_power)
        if power <= 0: return
        self.state = 'charging'
        ocv = self._get_ocv(self.state_of_charge)
        a = self.internal_resistance;
        b = ocv;
        c = -power
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0: return
        current = (-b + math.sqrt(discriminant)) / (2 * a)
        time_h = time_s / 3600.0
        delta_ah = current * time_h
        effective_delta_ah = delta_ah * self.coulombic_efficiency
        delta_soc = effective_delta_ah / self.capacity_ah
        self.state_of_charge += delta_soc
        self.state_of_charge = min(self.state_of_charge, self.soc_upper_limit)
        self._update_soh(delta_ah)
        self._record_history(time_s, power, ocv + current * a)

    def discharge(self, power, time_s):
        available_power = self.get_available_discharge_power()
        power = min(power, available_power)
        if power <= 0: return
        self.state = 'discharging'
        ocv = self._get_ocv(self.state_of_charge)
        a = self.internal_resistance;
        b = -ocv;
        c = power
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0: return
        current = (-b - math.sqrt(discriminant)) / (2 * a)
        time_h = time_s / 3600.0
        delta_ah = current * time_h
        delta_soc = delta_ah / self.capacity_ah
        self.state_of_charge -= delta_soc
        self.state_of_charge = max(self.state_of_charge, self.soc_lower_limit)
        self._update_soh(delta_ah)
        self._record_history(time_s, -power, ocv - current * a)

    def idle_loss(self, time_s):
        self.state = 'idle'
        daily_loss = 0.001
        loss_per_second = daily_loss / (24 * 3600)
        self.state_of_charge -= loss_per_second * time_s
        self._record_history(time_s, 0, self._get_ocv(self.state_of_charge))

    def _update_soh(self, ah_throughput):
        self.cumulative_ah_throughput += ah_throughput
        soh_loss = (self.cumulative_ah_throughput / self.total_ah_throughput_for_soh_loss) * 0.01
        self.state_of_health = 1.0 - soh_loss
        self.state_of_health = max(0, self.state_of_health)
        self.capacity_ah = self.nominal_capacity_ah * self.state_of_health
        self.internal_resistance = self.initial_internal_resistance / self.state_of_health

    def _record_history(self, time_delta, power, voltage):
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
        fig.suptitle(f'锂离子电池 ({self.id}) 性能曲线', fontsize=16)

        # 将时间从秒转换为小时，更适合长时间模拟的可视化
        time_in_hours = [t / 3600.0 for t in self.time_history]

        # 1. SOC 曲线
        axes[0].plot(time_in_hours, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化')
        axes[0].set_ylabel('SOC')
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_ylim(-0.05, 1.05)

        # 2. SOH 曲线 (对电池模型特别重要)
        axes[1].plot(time_in_hours, self.soh_history, 'r-', lw=2, label='SOH')
        axes[1].set_title('健康状态 (SOH) 变化')
        axes[1].set_ylabel('SOH')
        axes[1].grid(True)
        axes[1].legend()
        # 动态调整Y轴范围以便更清晰地观察SOH的微小变化
        if len(self.soh_history) > 1 and min(self.soh_history) < 0.99:
            axes[1].set_ylim(min(self.soh_history) * 0.99, 1.01)
        else:
            axes[1].set_ylim(0.9, 1.01)

        # 3. 终端电压曲线
        axes[2].plot(time_in_hours, self.voltage_history, 'c-', lw=2, label='终端电压')
        axes[2].set_title('终端电压变化')
        axes[2].set_ylabel('电压 (V)')
        axes[2].grid(True)
        axes[2].legend()

        # 4. 功率曲线
        axes[3].plot(time_in_hours, [p / 1000 for p in self.power_history], 'g-', lw=2, label='实时功率')
        axes[3].set_title('功率变化')
        axes[3].set_ylabel('功率 (kW)')
        axes[3].grid(True)
        axes[3].legend()
        axes[3].set_xlabel('时间 (小时)')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# --- HESS中的EMS调用示例 ---
def simulate_hess_with_battery():
    """一个简化的示例，演示EMS如何与电池模型交互，用于能量时移（削峰填谷）"""
    battery = LithiumIonBattery(initial_soc=0.5)

    time_steps_h = np.arange(0, 24, 0.5)
    price = 10 + 8 * np.sin((time_steps_h - 8) * np.pi / 12)

    print(f"--- 开始模拟，电池初始SOC: {battery.get_soc():.2f} ---")
    battery._record_history(0, 0, battery._get_ocv(battery.state_of_charge))

    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        current_price = price[i]

        if current_price < 8:
            power = battery.get_available_charge_power()
            battery.charge(power, dt_s)
        elif current_price > 15:
            power = battery.get_available_discharge_power()
            battery.discharge(power, dt_s)
        else:
            battery.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    battery.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_battery()