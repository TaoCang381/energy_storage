import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SuperconductingMagneticEnergyStorage(EnergyStorageUnit):
    """
    超导磁储能 (SMES) 模型 (HESS集成版)
    引入了功率调节系统(PCS)和低温制冷系统模型，适用于HESS的能源管理策略(EMS)调用。
    """

    def __init__(self,
                 # --- 核心物理参数 ---
                 inductance=100,  # 电感 (H)
                 critical_current=1000,  # 临界电流 (A)

                 # --- PCS 和制冷系统参数 ---
                 pcs_rated_power=100000,  # PCS额定功率 (W)，SMES功率非常大
                 pcs_efficiency=0.97,  # PCS转换效率
                 cryogenic_power=5000,  # 低温系统维持功率 (W)，固有损耗

                 # --- HESS集成新增参数 ---
                 ess_id="smes_01",
                 initial_soc=0.0):

        # --- HESS 集成参数 ---
        self.id = ess_id

        # --- 核心物理参数 ---
        self.inductance = inductance
        self.critical_current = critical_current

        # --- PCS 和制冷系统 ---
        self.pcs_rated_power = pcs_rated_power
        self.pcs_efficiency = pcs_efficiency
        self.cryogenic_power = cryogenic_power

        # --- 状态变量 ---
        # SOC = (I / I_crit)^2 => I = I_crit * sqrt(SOC)
        self.current = self.critical_current * math.sqrt(initial_soc)
        self.state = 'idle'  # 'idle', 'charging', 'discharging'

        # --- 历史记录 ---
        self.time_history = []
        self.current_history = []
        self.power_history = []
        self.soc_history = []

    def get_soc(self):
        """计算SOC，基于电流的平方比，公式: SOC = (I_curr / I_crit)^2"""
        return (self.current / self.critical_current) ** 2

    def calculate_energy(self):
        """计算当前储能, E = 0.5 * L * I^2"""
        return 0.5 * self.inductance * self.current ** 2

    # --- HESS接口核心方法 ---
    def get_available_charge_power(self):
        """EMS查询：获取当前可用的充电功率 (W)，主要受限于PCS功率"""
        # 如果电流已达临界值，则不能再充电
        if self.current >= self.critical_current:
            return 0
        return self.pcs_rated_power

    def get_available_discharge_power(self):
        """EMS查询：获取当前可用的放电功率 (W)，主要受限于PCS功率"""
        # 如果电流为0，则不能再放电
        if self.current <= 0:
            return 0
        return self.pcs_rated_power

    # --- 充放电与损耗控制方法 (重构) ---
    def charge(self, power, time):
        """
        按指定功率和时间充电 (功率指从电网吸收的功率)
        """
        available_power = self.get_available_charge_power()
        power = min(power, available_power)
        if power <= 0: return

        self.state = 'charging'
        # 实际注入到线圈的功率需考虑PCS效率
        power_to_coil = power * self.pcs_efficiency

        # P = V*I = (L*dI/dt)*I => dI/dt = P / (L*I)
        # 为避免I=0时奇异，在电流很小时，我们假设PCS以恒定dI/dt启动
        if self.current < 1e-3 * self.critical_current:
            # 简化处理：以额定功率的1%对应的dI/dt启动
            delta_i = (self.pcs_rated_power * 0.01 / (self.inductance * self.critical_current)) * time
        else:
            di_dt = power_to_coil / (self.inductance * self.current)
            delta_i = di_dt * time

        self.current += delta_i
        self.current = min(self.current, self.critical_current)
        self._record_history(time, power)

    def discharge(self, power, time):
        """
        按指定功率和时间放电 (功率指注入到电网的功率)
        """
        available_power = self.get_available_discharge_power()
        power = min(power, available_power)
        if power <= 0: return

        self.state = 'discharging'
        # 从线圈提取的功率需考虑PCS效率
        power_from_coil = power / self.pcs_efficiency

        if self.current <= 0: return

        di_dt = power_from_coil / (self.inductance * self.current)
        delta_i = di_dt * time

        self.current -= delta_i
        self.current = max(0, self.current)
        self._record_history(time, -power)

    def apply_cryogenic_load(self, time):
        """
        在HESS层面，此方法用于让EMS知晓SMES的持续寄生损耗。
        在模型内部，我们仅记录状态和时间。
        注意：这个功率损耗是持续的，无论充放电还是闲置。
        """
        self.state = 'idle'
        # SMES的电流在超导状态下几乎无损耗，因此不像其他储能有自放电。
        # 这里仅记录状态，真正的能量消耗是cryogenic_power，由EMS在系统层面核算。
        self._record_history(time, 0)  # 线圈本身功率交换为0

    def _record_history(self, time_delta, power):
        """记录历史数据"""
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.current_history.append(self.current)
        self.power_history.append(power)
        self.soc_history.append(self.get_soc())

    def plot_performance(self):
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(f'SMES ({self.id}) 性能曲线', fontsize=16)

        axes[0].plot(self.time_history, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].plot(self.time_history, self.current_history, 'c-', lw=2, label='电流')
        axes[1].axhline(self.critical_current, color='r', ls='--', label='临界电流')
        axes[1].set_title('线圈电流变化');
        axes[1].set_ylabel('电流 (A)');
        axes[1].grid(True);
        axes[1].legend()

        # 绘制净功率曲线，包含制冷损耗
        net_power = np.array(self.power_history) - self.cryogenic_power
        axes[2].plot(self.time_history, [p / 1000 for p in net_power], 'g-', lw=2, label='净输出功率')
        axes[2].axhline(-self.cryogenic_power / 1000, color='grey', ls=':',
                        label=f'制冷损耗 ({-self.cryogenic_power / 1000} kW)')
        axes[2].set_title('净功率变化 (包含制冷损耗)');
        axes[2].set_ylabel('功率 (kW)');
        axes[2].grid(True);
        axes[2].legend()
        axes[2].set_xlabel('时间 (s)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


# --- HESS中的EMS调用示例 ---
def simulate_hess_with_smes():
    """一个简化的示例，演示EMS如何与SMES模型交互，用于暂态功率支撑"""
    smes = SuperconductingMagneticEnergyStorage(initial_soc=0.5, pcs_rated_power=500000)

    # 模拟一个突然的、短暂的大功率缺口 (e.g., 大型电机启动)
    time_steps = np.arange(0, 10, 0.05)  # 10秒，每50ms一个决策点
    power_demand = np.zeros_like(time_steps)
    # 在 t=2s 时出现一个持续0.5秒的400kW功率缺口
    power_demand[(time_steps >= 2) & (time_steps < 2.5)] = 400000
    # 在 t=6s 时有一个反向的充电需求
    power_demand[(time_steps >= 6) & (time_steps < 6.8)] = -300000

    print(f"--- 开始模拟，SMES初始SOC: {smes.get_soc():.2f} ---")
    smes._record_history(0, 0)

    # EMS决策循环
    for i in range(len(time_steps) - 1):
        dt = time_steps[i + 1] - time_steps[i]
        demand = power_demand[i]

        if demand > 0:  # 需要放电
            available_power = smes.get_available_discharge_power()
            power_to_dispatch = min(demand, available_power)
            smes.discharge(power_to_dispatch, dt)
        elif demand < 0:  # 需要充电
            available_power = smes.get_available_charge_power()
            power_to_dispatch = min(abs(demand), available_power)
            smes.charge(power_to_dispatch, dt)
        else:  # 闲置
            smes.apply_cryogenic_load(dt)

    print("--- 模拟结束 ---")
    # 注意：绘图时会显示净功率，即从电网看，即使闲置SMES也是一个负荷。
    smes.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_smes()