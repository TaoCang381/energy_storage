import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 物理常数 ---
LHV_H2_J_PER_KG = 120e6  # 氢气低热值 (J/kg)
LHV_H2_KWH_PER_KG = 33.3  # 氢气低热值 (kWh/kg)


class HydrogenStorage(EnergyStorageUnit):
    """
    氢储能 (H2-ESS) 模型 (HESS集成版)
    特点：
    1. 包含电解槽、储氢罐、燃料电池三个核心部分。
    2. 分别考虑各环节的效率和寄生损耗（如压缩机）。
    3. 适用于大规模、长周期的能量时移场景。
    """

    def __init__(self,
                 # --- 制氢系统 (充电) ---
                 electrolyzer_rated_power_w=50e6,  # 电解槽额定功率 (W), e.g., 50 MW
                 electrolyzer_efficiency_kwh_kg=50,  # 制氢电耗 (kWh/kg), e.g., 50度电制1公斤氢气

                 # --- 储氢系统 ---
                 tank_max_capacity_kg=10000,  # 储氢罐最大容量 (kg), e.g., 10吨氢气
                 compressor_power_ratio=0.1,  # 压缩机功率占电解槽功率的比例, e.g., 10%

                 # --- 发电系统 (放电) ---
                 fuel_cell_rated_power_w=40e6,  # 燃料电池额定功率 (W), e.g., 40 MW
                 fuel_cell_efficiency_percent=0.55,  # 燃料电池发电效率 (%), e.g., 55%

                 # --- HESS集成 & 其他参数 ---
                 ess_id="hydrogen_storage_01",
                 initial_soh=1.0,  # SOH主要影响电解槽和燃料电池的效率
                 initial_soc=0.5,
                 soc_upper_limit=0.95,
                 soc_lower_limit=0.05
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh

        # --- 核心组件参数 ---
        self.electrolyzer_rated_power_w = electrolyzer_rated_power_w
        self.electrolyzer_efficiency_kwh_kg = electrolyzer_efficiency_kwh_kg
        self.tank_max_capacity_kg = tank_max_capacity_kg
        self.compressor_power_ratio = compressor_power_ratio
        self.fuel_cell_rated_power_w = fuel_cell_rated_power_w
        self.fuel_cell_efficiency_percent = fuel_cell_efficiency_percent
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit

        # --- 实时工作参数 (受SOH影响) ---
        # 假设SOH主要影响两端的转换效率
        self.current_fc_efficiency = self.fuel_cell_efficiency_percent * self.state_of_health
        self.current_ely_efficiency = self.electrolyzer_efficiency_kwh_kg / self.state_of_health

        # --- 状态变量 ---
        self.current_hydrogen_kg = self.tank_max_capacity_kg * initial_soc
        self.state = 'idle'
        # --- 历史记录 ---
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.mass_history = []

    def get_soc(self):
        """SOC = 当前氢气质量 / 储罐总容量"""
        return self.current_hydrogen_kg / self.tank_max_capacity_kg

    def calculate_max_energy_j(self):
        """计算最大可存储的化学能"""
        return self.tank_max_capacity_kg * LHV_H2_J_PER_KG

    # --- HESS接口核心方法 ---
    def get_available_charge_power(self):
        """获取当前可用的充电(制氢)功率 (W)"""
        if self.get_soc() >= self.soc_upper_limit:
            return 0
        # 可用功率受限于电解槽的额定功率
        return self.electrolyzer_rated_power_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_lower_limit:
            return 0
        # 可用功率受限于燃料电池的额定功率
        return self.fuel_cell_rated_power_w

    # --- 充放电与损耗控制方法 ---
    def charge(self, power_elec, time_s):
        """按指定电功率充电 (制氢)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return
        self.state = 'charging'

        # 计算辅助设备（压缩机）的功耗
        compressor_power = power_elec * self.compressor_power_ratio
        # 实际用于电解槽的功率
        power_to_electrolyzer = power_elec - compressor_power
        if power_to_electrolyzer <= 0: return

        # 计算生成的氢气质量
        time_h = time_s / 3600.0
        power_to_electrolyzer_kw = power_to_electrolyzer / 1000
        mass_produced_kg = (power_to_electrolyzer_kw * time_h) / self.current_ely_efficiency

        self.current_hydrogen_kg += mass_produced_kg
        self.current_hydrogen_kg = min(self.current_hydrogen_kg, self.tank_max_capacity_kg * self.soc_upper_limit)

        self._record_history(time_s, power_elec)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return
        self.state = 'discharging'

        # 计算消耗的氢气质量
        time_h = time_s / 3600.0
        power_elec_kw = power_elec / 1000
        # mass = (P_elec * t) / (LHV * eta_fc)
        mass_consumed_kg = (power_elec_kw * time_h) / (LHV_H2_KWH_PER_KG * self.current_fc_efficiency)

        # 检查是否有足够的氢气
        if mass_consumed_kg > self.current_hydrogen_kg:
            # 如果不够，则只能释放所有可用的氢气
            mass_consumed_kg = self.current_hydrogen_kg

        self.current_hydrogen_kg -= mass_consumed_kg
        self.current_hydrogen_kg = max(self.current_hydrogen_kg, self.tank_max_capacity_kg * self.soc_lower_limit)

        self._record_history(time_s, -power_elec)

    def idle_loss(self, time_s):
        """模拟闲置时的氢气泄漏 (非常微小)"""
        self.state = 'idle'
        # 假设每天泄漏总容量的0.01%
        daily_loss_ratio = 0.0001
        loss_per_second_kg = (self.tank_max_capacity_kg * daily_loss_ratio) / (24 * 3600)
        self.current_hydrogen_kg -= loss_per_second_kg * time_s
        self._record_history(time_s, 0)

    def _record_history(self, time_delta, power):
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(self.get_soc())
        self.mass_history.append(self.current_hydrogen_kg)

    def plot_performance(self):
        """绘制性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'氢储能 ({self.id}) 性能曲线', fontsize=16)

        time_d = [t / (3600.0 * 24) for t in self.time_history]  # 时间轴单位为天
        power_mw = [p / 1e6 for p in self.power_history]
        mass_ton = [m / 1000 for m in self.mass_history]

        axes[0].plot(time_d, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()

        axes[1].plot(time_d, mass_ton, 'c-', lw=2, label='氢气储存质量')
        axes[1].set_title('储氢量变化');
        axes[1].set_ylabel('质量 (吨)');
        axes[1].grid(True);
        axes[1].legend()

        axes[2].plot(time_d, power_mw, 'g-', lw=2, label='净输出功率')
        axes[2].set_title('功率变化');
        axes[2].set_ylabel('功率 (MW)');
        axes[2].grid(True);
        axes[2].legend()
        axes[2].set_xlabel('时间 (天)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


def simulate_hess_with_hydrogen():
    """一个简化的示例，演示氢储能用于跨季节储能"""
    h2_storage = HydrogenStorage(initial_soc=0.5)

    # 模拟一年中两个季节（丰/枯风期）的净负荷
    time_steps_d = np.arange(0, 60, 0.5)  # 模拟60天，每12小时一个决策点

    # 模拟前30天为风力过剩期（充电），后30天为缺风期（放电）
    net_power_demand = np.zeros_like(time_steps_d)
    # 前30天，平均有20MW的过剩电力用于制氢
    net_power_demand[time_steps_d <= 30] = -20e6
    # 后30天，平均有15MW的电力缺口需要氢气发电来补
    net_power_demand[time_steps_d > 30] = 15e6

    print(f"--- 开始模拟，氢储能初始SOC: {h2_storage.get_soc():.2f} ---")
    h2_storage._record_history(0, 0)

    for i in range(len(time_steps_d) - 1):
        dt_s = (time_steps_d[i + 1] - time_steps_d[i]) * 24 * 3600
        demand = net_power_demand[i]

        if demand < 0:  # 充电
            power = min(abs(demand), h2_storage.get_available_charge_power())
            h2_storage.charge(power, dt_s)
        elif demand > 0:  # 放电
            power = min(demand, h2_storage.get_available_discharge_power())
            h2_storage.discharge(power, dt_s)
        else:
            h2_storage.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    h2_storage.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_hydrogen()