import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 物理常数 ---
WATER_DENSITY_KG_M3 = 1000  # 水的密度 (kg/m^3)
GRAVITY_G = 9.81  # 重力加速度 (m/s^2)


class PumpedHydroStorage(EnergyStorageUnit):
    """
    抽水蓄能 (PHS) 模型 (HESS集成版)
    特点：
    1. 基于重力势能和流体动力学的基础物理公式。
    2. 能量和功率容量巨大，适用于长时程、大尺度应用。
    3. SOH几乎无衰减，寿命极长。
    """

    def __init__(self,
                 # --- 核心物理参数 ---
                 upper_reservoir_volume_m3=1.5e7,  # 上水库有效库容 (m^3), e.g., 1500万立方米
                 effective_head_m=400,  # 有效水头差 (m), e.g., 400米

                 # --- 机组性能参数 ---
                 turbine_rated_power_w=300e6,  # 水轮机组额定功率 (W), e.g., 300 MW
                 pump_rated_power_w=320e6,  # 水泵机组额定功率 (W), e.g., 320 MW (通常略大于水轮机)
                 turbine_efficiency=0.9,  # 水轮机效率
                 pump_efficiency=0.88,  # 水泵效率

                 # --- HESS集成参数 ---
                 ess_id="pumped_hydro_01",
                 initial_soc=0.5,
                 soc_upper_limit=0.98,  # 考虑蒸发和死水位，不完全充满放空
                 soc_lower_limit=0.02
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = 1.0  # PHS的SOH几乎不衰减，恒定为1.0

        # --- 规格参数 ---
        self.upper_reservoir_volume_m3 = upper_reservoir_volume_m3
        self.effective_head_m = effective_head_m
        self.turbine_rated_power_w = turbine_rated_power_w
        self.pump_rated_power_w = pump_rated_power_w
        self.turbine_efficiency = turbine_efficiency
        self.pump_efficiency = pump_efficiency
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit

        # --- 状态变量 ---
        self.current_volume_m3 = self.upper_reservoir_volume_m3 * initial_soc
        self.state = 'idle'

        # --- 历史记录 ---
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.volume_history = []

    def calculate_max_energy_j(self):
        """计算最大储能 E = rho * V * g * h"""
        return WATER_DENSITY_KG_M3 * self.upper_reservoir_volume_m3 * GRAVITY_G * self.effective_head_m

    def get_soc(self):
        """SOC = 当前水量 / 总库容"""
        return self.current_volume_m3 / self.upper_reservoir_volume_m3

    # --- HESS接口核心方法 ---
    def get_available_charge_power(self):
        """获取当前可用的充电(抽水)功率 (W)"""
        if self.get_soc() >= self.soc_upper_limit:
            return 0
        return self.pump_rated_power_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_lower_limit:
            return 0
        return self.turbine_rated_power_w

    # --- 充放电与损耗控制方法 ---
    def charge(self, power, time_s):
        """按指定功率充电 (抽水)"""
        power = min(power, self.get_available_charge_power())
        if power <= 0: return
        self.state = 'charging'

        # 根据功率反算水流量 Q = P_pump / (rho*g*h/eta_pump)
        flow_rate_m3s = (power * self.pump_efficiency) / (WATER_DENSITY_KG_M3 * GRAVITY_G * self.effective_head_m)

        # 更新水量
        delta_volume = flow_rate_m3s * time_s
        self.current_volume_m3 += delta_volume
        self.current_volume_m3 = min(self.current_volume_m3, self.upper_reservoir_volume_m3 * self.soc_upper_limit)

        self._record_history(time_s, power)

    def discharge(self, power, time_s):
        """按指定功率放电 (发电)"""
        power = min(power, self.get_available_discharge_power())
        if power <= 0: return
        self.state = 'discharging'

        # 根据功率反算水流量 Q = P_gen / (eta_turbine*rho*g*h)
        flow_rate_m3s = power / (self.turbine_efficiency * WATER_DENSITY_KG_M3 * GRAVITY_G * self.effective_head_m)

        # 更新水量
        delta_volume = flow_rate_m3s * time_s
        self.current_volume_m3 -= delta_volume
        self.current_volume_m3 = max(self.current_volume_m3, self.upper_reservoir_volume_m3 * self.soc_lower_limit)

        self._record_history(time_s, -power)

    def idle_loss(self, time_s):
        """模拟闲置时的蒸发损耗 (非常微小)"""
        self.state = 'idle'
        # 假设一个非常小的蒸发率
        evaporation_rate_m3s = 0.01  # 示例值
        self.current_volume_m3 -= evaporation_rate_m3s * time_s
        self._record_history(time_s, 0)

    def _record_history(self, time_delta, power):
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(self.get_soc())
        self.volume_history.append(self.current_volume_m3)

    def plot_performance(self):
        """绘制性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'抽水蓄能 ({self.id}) 性能曲线', fontsize=16)

        time_h = [t / 3600.0 for t in self.time_history]
        power_mw = [p / 1e6 for p in self.power_history]
        volume_mil_m3 = [v / 1e6 for v in self.volume_history]

        axes[0].plot(time_h, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()

        axes[1].plot(time_h, volume_mil_m3, 'c-', lw=2, label='上水库水量')
        axes[1].set_title('水量变化');
        axes[1].set_ylabel('水量 (百万 m³)');
        axes[1].grid(True);
        axes[1].legend()

        axes[2].plot(time_h, power_mw, 'g-', lw=2, label='净输出功率')
        axes[2].set_title('功率变化');
        axes[2].set_ylabel('功率 (MW)');
        axes[2].grid(True);
        axes[2].legend()
        axes[2].set_xlabel('时间 (小时)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


def simulate_hess_with_phs():
    """一个简化的示例，演示抽水蓄能用于周调节，实现可再生能源的大规模时移"""
    phs = PumpedHydroStorage(initial_soc=0.5)

    # 模拟一周（7天）的电网净负荷
    time_steps_h = np.arange(0, 24 * 7, 1)  # 1小时一个决策点

    # 净负荷 = 负荷 - 可再生能源出力。负值表示能源过剩，需要充电
    # 工作日负荷高，周末负荷低
    weekday_pattern = 150e6 + 100e6 * np.sin((time_steps_h % 24 - 9) * np.pi / 12)
    weekend_pattern = 100e6 + 50e6 * np.sin((time_steps_h % 24 - 9) * np.pi / 12)

    net_load = np.zeros_like(time_steps_h)
    for i, t in enumerate(time_steps_h):
        # 假设周六日是周末（第5、6天）
        if (t // 24) % 7 >= 5:
            net_load[i] = weekend_pattern[i]
        else:
            net_load[i] = weekday_pattern[i]

    print(f"--- 开始模拟，抽水蓄能初始SOC: {phs.get_soc():.2f} ---")
    print(f"最大储能: {phs.calculate_max_energy_j() / 3.6e9:.2f} GWh")
    phs._record_history(0, 0)

    # EMS决策循环：简单的阈值控制（周末或夜间低负荷时抽水，工作日高峰负荷时发电）
    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        demand = net_load[i]

        # 决策逻辑：当负荷低于100MW时，认为是低谷，全力抽水
        if demand < 100e6:
            power = phs.get_available_charge_power()
            phs.charge(power, dt_s)
        # 当负荷高于220MW时，认为是高峰，全力发电
        elif demand > 220e6:
            power = phs.get_available_discharge_power()
            phs.discharge(power, dt_s)
        else:
            phs.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    phs.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_phs()