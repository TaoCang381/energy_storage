import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ThermalEnergyStorage(EnergyStorageUnit):
    """
    热储能 (TES) 模型 (HESS集成版)
    特点：
    1. 基于显热储能物理公式 Q = mcΔT。
    2. 包含非对称的“电-热-电”转换效率。
    3. 模拟储罐的持续散热损失。
    """

    def __init__(self,
                 # --- 储热罐物理参数 ---
                 storage_medium_mass_kg=1e7,  # 储热介质质量 (kg), e.g., 1万吨熔盐
                 specific_heat_capacity_j_kgk=1500,  # 介质比热容 (J/(kg·K)), e.g., 熔盐

                 # --- 温度与运行限制 ---
                 max_temp_k=838.15,  # 最高工作温度 (K), e.g., 565°C
                 min_temp_k=563.15,  # 最低工作温度 (K), e.g., 290°C

                 # --- 功率转换系统(PCS)参数 ---
                 heater_rated_power_w=110e6,  # 电加热器额定功率 (W), e.g., 110MW
                 heat_engine_rated_power_w=100e6,  # 热机额定输出功率 (W), e.g., 100MW

                 # --- 效率与损耗 ---
                 elec_to_heat_efficiency=0.98,  # 电->热 转换效率 (非常高)
                 heat_to_elec_efficiency=0.42,  # 热->电 转换效率 (受热力学限制)
                 heat_loss_rate_percent_hr=0.04,  # 每小时散热损失率 (% of stored energy)

                 # --- HESS集成参数 ---
                 ess_id="thermal_storage_01",
                 initial_soc=0.5
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = 1.0  # TES的SOH几乎不衰减

        # --- 规格参数 ---
        self.storage_medium_mass_kg = storage_medium_mass_kg
        self.specific_heat_capacity_j_kgk = specific_heat_capacity_j_kgk
        self.max_temp_k = max_temp_k
        self.min_temp_k = min_temp_k
        self.heater_rated_power_w = heater_rated_power_w
        self.heat_engine_rated_power_w = heat_engine_rated_power_w
        self.elec_to_heat_efficiency = elec_to_heat_efficiency
        self.heat_to_elec_efficiency = heat_to_elec_efficiency
        self.heat_loss_rate_per_sec = (heat_loss_rate_percent_hr / 100) / 3600.0

        # --- 状态变量 ---
        self.current_temp_k = self.min_temp_k + (self.max_temp_k - self.min_temp_k) * initial_soc
        self.state = 'idle'

        # --- 历史记录 ---
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.temp_history = []

    def _temp_to_soc(self, temp_k):
        """内部方法：温度 -> SOC"""
        return (temp_k - self.min_temp_k) / (self.max_temp_k - self.min_temp_k)

    def get_soc(self):
        """获取SOC"""
        return self._temp_to_soc(self.current_temp_k)

    def calculate_max_thermal_energy_j(self):
        """计算最大可储存的热能 E = m*c*dT"""
        delta_t = self.max_temp_k - self.min_temp_k
        return self.storage_medium_mass_kg * self.specific_heat_capacity_j_kgk * delta_t

    # --- HESS接口核心方法 ---
    def get_available_charge_power(self):
        """获取当前可用的充电(加热)功率 (W)"""
        if self.get_soc() >= 1.0:
            return 0
        return self.heater_rated_power_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= 0.0:
            return 0
        return self.heat_engine_rated_power_w

    # --- 充放电与损耗控制方法 ---
    def charge(self, power_elec, time_s):
        """按指定电功率充电 (加热)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return
        self.state = 'charging'

        # 计算注入的热功率
        power_heat = power_elec * self.elec_to_heat_efficiency

        # 计算温升 dT = (P_heat * t) / (m * c)
        delta_temp = (power_heat * time_s) / (self.storage_medium_mass_kg * self.specific_heat_capacity_j_kgk)

        self.current_temp_k += delta_temp
        self.current_temp_k = min(self.current_temp_k, self.max_temp_k)

        self._record_history(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return
        self.state = 'discharging'

        # 计算需要从储罐中提取的热功率
        power_heat = power_elec / self.heat_to_elec_efficiency

        # 计算温降 dT = (P_heat * t) / (m * c)
        delta_temp = (power_heat * time_s) / (self.storage_medium_mass_kg * self.specific_heat_capacity_j_kgk)

        self.current_temp_k -= delta_temp
        self.current_temp_k = max(self.current_temp_k, self.min_temp_k)

        self._record_history(time_s, -power_elec)

    def idle_loss(self, time_s):
        """模拟闲置时的散热损失"""
        self.state = 'idle'

        # 当前储存的热能
        current_thermal_energy = self.storage_medium_mass_kg * self.specific_heat_capacity_j_kgk * (
                    self.current_temp_k - self.min_temp_k)
        # 损失的热能
        lost_heat = current_thermal_energy * self.heat_loss_rate_per_sec * time_s
        # 计算温度下降
        delta_temp = lost_heat / (self.storage_medium_mass_kg * self.specific_heat_capacity_j_kgk)

        self.current_temp_k -= delta_temp
        self._record_history(time_s, 0)

    def _record_history(self, time_delta, power):
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(self.get_soc())
        self.temp_history.append(self.current_temp_k)

    def plot_performance(self):
        """绘制性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'热储能 ({self.id}) 性能曲线', fontsize=16)

        time_h = [t / 3600.0 for t in self.time_history]
        power_mw = [p / 1e6 for p in self.power_history]
        temp_c = [t - 273.15 for t in self.temp_history]

        axes[0].plot(time_h, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()

        axes[1].plot(time_h, temp_c, 'r-', lw=2, label='介质温度')
        axes[1].axhline(self.max_temp_k - 273.15, color='red', ls='--', label='最高温度')
        axes[1].axhline(self.min_temp_k - 273.15, color='blue', ls='--', label='最低温度')
        axes[1].set_title('储热介质温度变化');
        axes[1].set_ylabel('温度 (°C)');
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


def simulate_hess_with_tes():
    """一个简化的示例，演示热储能用于消纳光伏，提供晚高峰电力"""
    tes = ThermalEnergyStorage(initial_soc=0.1)  # 早上开始时，储罐是冷的

    # 模拟一天24小时的净负荷（负荷-光伏）
    time_steps_h = np.arange(0, 24, 0.5)
    base_load = 60e6  # 60MW基础负荷
    pv_generation = np.maximum(0, 150e6 * np.sin((time_steps_h - 6) * np.pi / 12))
    evening_peak = 40e6 * np.exp(-((time_steps_h - 19) ** 2) / 4)
    net_load = base_load + evening_peak - pv_generation

    print(f"--- 开始模拟，热储能初始SOC: {tes.get_soc():.2f} ---")
    print(
        f"最大储能: {tes.calculate_max_thermal_energy_j() / 3.6e9:.2f} GWh(热) / {tes.calculate_max_thermal_energy_j() * tes.heat_to_elec_efficiency / 3.6e9:.2f} GWh(电)")
    tes._record_history(0, 0)

    # EMS决策循环：白天光伏过剩时充电，晚上高峰负荷时放电
    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        demand = -net_load[i]  # 需求 = -净负荷

        # 白天中午(10点到16点)，光伏过剩，全力充电
        if 10 <= time_steps_h[i] <= 16 and demand > 0:
            power = tes.get_available_charge_power()
            tes.charge(power, dt_s)
        # 晚高峰(18点到22点)，需要放电
        elif 18 <= time_steps_h[i] <= 22 and demand < 0:
            power = tes.get_available_discharge_power()
            tes.discharge(power, dt_s)
        else:
            tes.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    tes.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_tes()