import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Supercapacitor(EnergyStorageUnit):
    """
    超级电容器模型 (HESS集成版)
    增加了动态可用功率、SOH、SOC等接口，适用于混合储能系统能源管理策略(EMS)调用。
    """

    def __init__(self,
                 # --- 基础物理参数 (设计值) ---
                 initial_capacitance=3000,  # 初始电容量 (F)
                 initial_esr=0.01,  # 初始等效串联电阻 (Ω)
                 max_voltage=2.7,  # 最大工作电压 (V)
                 min_voltage=1.0,  # 最小工作电压 (V)
                 rated_current=200,  # 额定电流 (A) - 关键运行参数

                 # --- HESS集成新增参数 ---
                 ess_id="supercap_01",  # 唯一标识符
                 initial_soh=1.0,  # 初始健康状态 (1.0代表全新)
                 initial_soc=0.0,  # 初始荷电状态
                 self_discharge_rate=1e-4  # 自放电率 (简化为等效漏电流 A)
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh
        self.self_discharge_rate = self_discharge_rate

        # --- 基础物理和性能参数 ---
        self.initial_capacitance = initial_capacitance
        self.initial_esr = initial_esr
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.rated_current = rated_current

        # --- 实时工作参数 (受SOH影响) ---
        # SOH降低，电容量减小，内阻增大
        self.capacitance = self.initial_capacitance * self.state_of_health
        self.esr = self.initial_esr / self.state_of_health  # 内阻通常与SOH成反比

        # --- 状态变量 ---
        self.current_voltage = self.min_voltage + (self.max_voltage - self.min_voltage) * math.sqrt(initial_soc)
        self.state = 'idle'  # 'idle', 'charging', 'discharging'

        # --- 历史记录 ---
        self.time_history = []
        self.voltage_history = []
        self.power_history = []
        self.soc_history = []

    def get_soc(self):
        """计算SOC，基于能量的定义，公式: SOC = (V_curr^2 - V_min^2) / (V_max^2 - V_min^2)"""
        v_curr_sq = self.current_voltage ** 2
        v_min_sq = self.min_voltage ** 2
        v_max_sq = self.max_voltage ** 2

        if v_max_sq <= v_min_sq: return 0
        soc = (v_curr_sq - v_min_sq) / (v_max_sq - v_min_sq)
        return max(0, min(1, soc))  # 保证SOC在0-1之间

    # --- HESS接口核心方法 (新增) ---
    def get_available_charge_power(self):
        """EMS查询：获取当前可用的充电功率 (W)"""
        if self.current_voltage >= self.max_voltage:
            return 0
        # 充电功率受限于1)额定电流 2)不超过最大电压
        # I_max_charge = (V_max - V_current) / R_esr (理论值，可能极大)
        # 实际受额定电流限制
        max_power_by_current = self.rated_current * self.current_voltage
        # 终端电压 V_terminal = V_current + I * R_esr <= V_max
        # => I <= (V_max - V_current) / R_esr
        # => Power = I * V_current <= (V_max - V_current) * V_current / R_esr
        max_power_by_voltage = (self.max_voltage - self.current_voltage) * self.current_voltage / self.esr
        return max(0, min(max_power_by_current, max_power_by_voltage))

    def get_available_discharge_power(self):
        """EMS查询：获取当前可用的放电功率 (W)"""
        if self.current_voltage <= self.min_voltage:
            return 0
        # 放电功率受限于1)额定电流 2)不低于最小电压
        # V_terminal = V_current - I * R_esr >= V_min
        # => I <= (V_current - V_min) / R_esr
        # => Power = I * V_current <= (V_current - V_min) * V_current / R_esr
        max_power_by_voltage = (self.current_voltage - self.min_voltage) * self.current_voltage / self.esr
        max_power_by_current = self.rated_current * self.current_voltage
        return max(0, min(max_power_by_current, max_power_by_voltage))

    # --- 充放电与损耗控制方法 (重构) ---
    #!!!问题存在P大于0和小于0分别表示充电和放电状态，没有区分
    def charge(self, power, time):
        """按指定功率和时间充电"""
        available_power = self.get_available_charge_power()
        power = min(power, available_power)
        if power <= 0 or self.current_voltage <= 0: return

        self.state = 'charging'
        current = power / self.current_voltage

        # 计算电荷量变化 dQ = I * dt
        delta_q = current * time
        # 计算电压变化 dV = dQ / C
        delta_v = delta_q / self.capacitance

        self.current_voltage += delta_v
        # 确保电压不超过上限
        self.current_voltage = min(self.current_voltage, self.max_voltage)
        self._record_history(time, power)

    def discharge(self, power, time):
        """按指定功率和时间放电"""
        available_power = self.get_available_discharge_power()
        power = min(power, available_power)
        if power <= 0 or self.current_voltage <= 0: return

        self.state = 'discharging'
        current = power / self.current_voltage

        delta_q = current * time
        delta_v = delta_q / self.capacitance

        self.current_voltage -= delta_v
        # 确保电压不低于下限
        self.current_voltage = max(self.current_voltage, self.min_voltage)
        self._record_history(time, -power)

    def idle_loss(self, time):
        """计算自放电损耗"""
        self.state = 'idle'
        delta_q = self.self_discharge_rate * time
        delta_v = delta_q / self.capacitance
        self.current_voltage -= delta_v
        self.current_voltage = max(self.current_voltage, self.min_voltage)
        self._record_history(time, 0)

    def _record_history(self, time_delta, power):
        """记录历史数据"""
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.voltage_history.append(self.current_voltage)
        self.power_history.append(power)
        self.soc_history.append(self.get_soc())

    def plot_performance(self):
        """绘制性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(f'超级电容器 ({self.id}) 性能曲线', fontsize=16)

        axes[0].plot(self.time_history, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].plot(self.time_history, self.voltage_history, 'c-', lw=2, label='电压')
        axes[1].axhline(self.max_voltage, color='r', ls='--', label='最大电压')
        axes[1].axhline(self.min_voltage, color='orange', ls='--', label='最小电压')
        axes[1].set_title('电压变化');
        axes[1].set_ylabel('电压 (V)');
        axes[1].grid(True);
        axes[1].legend()

        axes[2].plot(self.time_history, [p / 1000 for p in self.power_history], 'g-', lw=2, label='实时功率')
        axes[2].set_title('功率变化');
        axes[2].set_ylabel('功率 (kW)');
        axes[2].grid(True);
        axes[2].legend()
        axes[2].set_xlabel('时间 (s)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


# --- HESS中的EMS调用示例 ---
def simulate_hess_with_supercap():
    """一个简化的示例，演示EMS如何与超级电容器模型交互，用于平滑高频功率波动"""
    supercap = Supercapacitor(initial_soc=0.5, rated_current=300)

    # 模拟一个高频波动的功率信号 (e.g., 风电输出的毛刺)
    time_steps = np.arange(0, 20, 0.1)  # 20秒，每0.1秒一个决策点
    # 一个基准功率 + 高频噪声
    power_fluctuation = 20000 * np.sin(time_steps * 50) * np.exp(-0.1 * time_steps)

    print(f"--- 开始模拟，超容初始SOC: {supercap.get_soc():.2f} ---")
    supercap._record_history(0, 0)

    # EMS决策循环
    for i in range(len(time_steps) - 1):
        dt = time_steps[i + 1] - time_steps[i]
        demand = power_fluctuation[i]

        if demand > 0:  # 功率过剩，需要充电吸收
            available_power = supercap.get_available_charge_power()
            power_to_dispatch = min(demand, available_power)
            supercap.charge(power_to_dispatch, dt)
        elif demand < 0:  # 功率不足，需要放电支撑
            available_power = supercap.get_available_discharge_power()
            power_to_dispatch = min(abs(demand), available_power)
            supercap.discharge(power_to_dispatch, dt)
        else:
            supercap.idle_loss(dt)

    print("--- 模拟结束 ---")
    supercap.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_supercap()