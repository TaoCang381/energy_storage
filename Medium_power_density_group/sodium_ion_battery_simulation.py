import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SodiumIonBattery(EnergyStorageUnit):
    """
    钠离子电池模型 (HESS集成版)
    模型结构与锂离子电池相同，但参数反映了钠离子电池的独有特性
    （如更低的电压平台、更低的成本等）。
    """

    def __init__(self,
                 # --- 基础电化学规格 (注意与锂电池的差异) ---
                 nominal_capacity_ah=100,  # 额定安时容量 (Ah) - 假设与锂电池相同以便比较
                 nominal_voltage=3.1,  # 额定电压 (V) - 明显低于锂电池
                 initial_internal_resistance=0.003,  # 初始内阻 (Ohm) - 可能略高于锂电池

                 # --- 运行限制 ---
                 max_c_rate_charge=1.0,
                 max_c_rate_discharge=1.5,  # 放电倍率可能略低于高性能锂电池
                 soc_upper_limit=0.9,
                 soc_lower_limit=0.1,
                 max_voltage_limit=3.8,  # 终端电压上限 (低于锂电池)
                 min_voltage_limit=2.0,  # 终端电压下限 (低于锂电池)

                 # --- HESS集成 & 其他参数 ---
                 ess_id="sodium_ion_battery_01",
                 initial_soh=1.0,
                 initial_soc=0.5,
                 coulombic_efficiency=0.99,
                 cost_per_kwh=0.02,  # 度电成本 - 显著低于锂电池，是其核心优势
                 # 简化的OCV-SOC曲线模型参数 (示例，针对某款钠离子电池)
                 ocv_params={'P1': 3.2, 'P2': 0.06, 'P3': 0.12, 'P4': 0.1},
                 # 循环寿命 - 假设与LFP电池类似
                 cycles_for_1pct_soh_loss=3000
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh
        self.state_of_charge = initial_soc
        self.coulombic_efficiency = coulombic_efficiency
        self.cost_per_kwh = cost_per_kwh

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
        # V(soc) = P1 - P2/soc - P3*soc + P4*ln(soc)
        return p['P1'] - p['P2'] / soc - p['P3'] * soc + p['P4'] * math.log(soc)

    def get_soc(self):
        return self.state_of_charge

    # --- HESS接口核心方法 ---
    def get_available_charge_power(self):
        """EMS查询：获取当前可用的充电功率 (W)"""
        soc = self.state_of_charge
        if soc >= self.soc_upper_limit: return 0
        max_current_by_c_rate = self.max_c_rate_charge * self.capacity_ah
        ocv = self._get_ocv(soc)
        max_current_by_voltage = (self.max_voltage_limit - ocv) / self.internal_resistance
        charge_current = max(0, min(max_current_by_c_rate, max_current_by_voltage))
        charge_voltage = ocv + charge_current * self.internal_resistance
        return charge_current * charge_voltage

    def get_available_discharge_power(self):
        """EMS查询：获取当前可用的放电功率 (W)"""
        soc = self.state_of_charge
        if soc <= self.soc_lower_limit: return 0
        max_current_by_c_rate = self.max_c_rate_discharge * self.capacity_ah
        ocv = self._get_ocv(soc)
        max_current_by_voltage = (ocv - self.min_voltage_limit) / self.internal_resistance
        discharge_current = max(0, min(max_current_by_c_rate, max_current_by_voltage))
        discharge_voltage = ocv - discharge_current * self.internal_resistance
        return discharge_current * discharge_voltage

    # --- 充放电与损耗控制方法 ---
    def charge(self, power, time_s):
        """按指定功率和时间充电"""
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
        """按指定功率和时间放电"""
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
        """模拟自放电 (简化模型)"""
        self.state = 'idle'
        daily_loss = 0.001
        loss_per_second = daily_loss / (24 * 3600)
        self.state_of_charge -= loss_per_second * time_s
        self._record_history(time_s, 0, self._get_ocv(self.state_of_charge))

    def _update_soh(self, ah_throughput):
        """内部方法：根据安时吞吐量更新SOH"""
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
        fig.suptitle(f'钠离子电池 ({self.id}) 性能曲线', fontsize=16)
        time_in_hours = [t / 3600.0 for t in self.time_history]
        axes[0].plot(time_in_hours, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()
        axes[1].plot(time_in_hours, self.soh_history, 'r-', lw=2, label='SOH')
        axes[1].set_title('健康状态 (SOH) 变化');
        axes[1].set_ylabel('SOH');
        axes[1].grid(True);
        axes[1].legend()
        axes[2].plot(time_in_hours, self.voltage_history, 'c-', lw=2, label='终端电压')
        axes[2].set_title('终端电压变化');
        axes[2].set_ylabel('电压 (V)');
        axes[2].grid(True);
        axes[2].legend()
        axes[3].plot(time_in_hours, [p / 1000 for p in self.power_history], 'g-', lw=2, label='实时功率')
        axes[3].set_title('功率变化');
        axes[3].set_ylabel('功率 (kW)');
        axes[3].grid(True);
        axes[3].legend()
        axes[3].set_xlabel('时间 (小时)')
        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


# --- HESS中的EMS调用示例 ---
def simulate_hess_with_sib():
    """一个简化的示例，演示EMS如何与钠离子电池模型交互，用于成本敏感的能量时移"""
    # 注意，我们使用的是SodiumIonBattery类
    sib = SodiumIonBattery(initial_soc=0.5)

    # 使用与锂电池相同的场景，以便进行对比
    time_steps_h = np.arange(0, 24 * 3, 0.5)  # 模拟3天
    price = 10 + 8 * np.sin((time_steps_h % 24 - 8) * np.pi / 12)

    print(f"--- 开始模拟，钠离子电池初始SOC: {sib.get_soc():.2f} ---")
    sib._record_history(0, 0, sib._get_ocv(sib.state_of_charge))

    # EMS决策循环：与锂电池相同的电价套利策略
    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        current_price = price[i]

        if current_price < 8:
            power = sib.get_available_charge_power()
            sib.charge(power, dt_s)
        elif current_price > 15:
            power = sib.get_available_discharge_power()
            sib.discharge(power, dt_s)
        else:
            sib.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    sib.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_sib()