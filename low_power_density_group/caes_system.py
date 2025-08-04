import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class DiabaticCAES(EnergyStorageUnit):
    """
    补燃式压缩空气储能 (D-CAES) 模型 (HESS集成版)
    特点：
    1. 包含压缩机、储气室、燃气轮机三个部分。
    2. 放电过程消耗化石燃料（天然气），是一个混合动力系统。
    3. 性能由热耗率和耗气率等关键参数定义。
    """

    def __init__(self,
                 # --- 储气室 (能量容量) ---
                 cavern_max_air_mass_kg=5e7,  # 储气室最大空气容量 (kg), e.g., 5万吨

                 # --- 压缩机组 (充电) ---
                 compressor_rated_power_w=200e6,  # 压缩机组额定功率 (W), e.g., 200 MW
                 charge_rate_kg_per_j=1e-4,  # 充电速率 (kg/J): 每消耗1焦耳电能存入的空气质量

                 # --- 透平发电机组 (放电) ---
                 turbine_rated_power_w=300e6,  # 透平发电机组额定功率 (W), e.g., 300 MW
                 heat_rate_j_per_j=1.5,  # 热耗率 (J/J): 每输出1J电能，需要消耗1.5J的燃料热值
                 air_usage_rate_kg_per_j=5e-5,  # 耗气率 (kg/J): 每输出1J电能，需要消耗5e-5 kg的压缩空气

                 # --- HESS集成 & 其他参数 ---
                 ess_id="diabatic_caes_01",
                 initial_soc=0.5,
                 soc_upper_limit=0.98,
                 soc_lower_limit=0.2  # 储气室通常保持一定底压
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = 1.0  # CAES的SOH几乎不衰减

        # --- 规格参数 ---
        self.cavern_max_air_mass_kg = cavern_max_air_mass_kg
        self.compressor_rated_power_w = compressor_rated_power_w
        self.charge_rate_kg_per_j = charge_rate_kg_per_j
        self.turbine_rated_power_w = turbine_rated_power_w
        self.heat_rate_j_per_j = heat_rate_j_per_j
        self.air_usage_rate_kg_per_j = air_usage_rate_kg_per_j
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit

        # --- 状态变量 ---
        self.current_air_mass_kg = self.cavern_max_air_mass_kg * initial_soc
        self.state = 'idle'

        # --- 历史记录 ---
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.fuel_consumption_history = []  # 新增：记录燃料消耗

    def get_soc(self):
        """SOC = 当前空气质量 / 最大质量"""
        return self.current_air_mass_kg / self.cavern_max_air_mass_kg

    # --- HESS接口核心方法 ---
    def get_available_charge_power(self):
        """获取当前可用的充电(压缩)功率 (W)"""
        if self.get_soc() >= self.soc_upper_limit:
            return 0
        return self.compressor_rated_power_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_lower_limit:
            return 0
        # 同时受限于额定功率和剩余空气量
        max_power_by_air = self.current_air_mass_kg / self.air_usage_rate_kg_per_j / 3600  # 假设能持续1小时
        return min(self.turbine_rated_power_w, max_power_by_air)

    # --- 充放电与损耗控制方法 ---
    def charge(self, power_elec, time_s):
        """按指定电功率充电 (压缩)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0: return
        self.state = 'charging'

        # 计算存入的空气质量
        energy_consumed_j = power_elec * time_s
        mass_stored_kg = energy_consumed_j * self.charge_rate_kg_per_j

        self.current_air_mass_kg += mass_stored_kg
        self.current_air_mass_kg = min(self.current_air_mass_kg, self.cavern_max_air_mass_kg * self.soc_upper_limit)

        self._record_history(time_s, power_elec, 0)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，同时计算燃料消耗"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0: return
        self.state = 'discharging'

        energy_generated_j = power_elec * time_s

        # 计算消耗的空气质量
        mass_consumed_kg = energy_generated_j * self.air_usage_rate_kg_per_j

        # 检查是否有足够的空气
        if mass_consumed_kg > self.current_air_mass_kg:
            mass_consumed_kg = self.current_air_mass_kg
            energy_generated_j = mass_consumed_kg / self.air_usage_rate_kg_per_j
            power_elec = energy_generated_j / time_s

        self.current_air_mass_kg -= mass_consumed_kg

        # 计算消耗的燃料热量
        fuel_consumed_j = energy_generated_j * self.heat_rate_j_per_j

        self._record_history(time_s, -power_elec, fuel_consumed_j)

    def idle_loss(self, time_s):
        """模拟闲置时的洞穴气体泄漏 (非常微小)"""
        self.state = 'idle'
        leakage_rate_kg_per_s = 1.0  # 示例值: 1 kg/s
        self.current_air_mass_kg -= leakage_rate_kg_per_s * time_s
        self.current_air_mass_kg = max(0, self.current_air_mass_kg)
        self._record_history(time_s, 0, 0)

    def _record_history(self, time_delta, power, fuel_consumed_j):
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(self.get_soc())
        self.fuel_consumption_history.append(fuel_consumed_j)

    def plot_performance(self):
        """绘制性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'补燃式CAES ({self.id}) 性能曲线', fontsize=16)

        time_h = [t / 3600.0 for t in self.time_history]
        power_mw = [p / 1e6 for p in self.power_history]
        fuel_power_mw = [
            (f / (time_h[i + 1] - time_h[i]) / 3600 if (i + 1 < len(time_h) and time_h[i + 1] > time_h[i]) else 0) / 1e6
            for i, f in enumerate(self.fuel_consumption_history)]

        axes[0].plot(time_h, self.soc_history, 'm-', lw=2, label='SOC (储气量)')
        axes[0].set_title('荷电状态 (SOC) 变化');
        axes[0].set_ylabel('SOC');
        axes[0].grid(True);
        axes[0].legend()

        axes[1].plot(time_h, power_mw, 'g-', lw=2, label='净输出电功率')
        axes[1].set_title('电功率变化');
        axes[1].set_ylabel('功率 (MW)');
        axes[1].grid(True);
        axes[1].legend()

        axes[2].plot(time_h, fuel_power_mw, 'r-', lw=2, label='燃料消耗功率')
        axes[2].set_title('燃料消耗功率变化');
        axes[2].set_ylabel('功率 (MW_th)');
        axes[2].grid(True);
        axes[2].legend()
        axes[2].set_xlabel('时间 (小时)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


def simulate_hess_with_caes():
    """一个简化的示例，演示CAES用于电网大规模调峰"""
    caes = DiabaticCAES(initial_soc=0.5)

    # 模拟一周的电网负荷
    time_steps_h = np.arange(0, 24 * 7, 1)

    # 典型负荷曲线，夜间低谷，白天双高峰
    load = 150e6 - 100e6 * np.cos(time_steps_h * np.pi / 12) + 50e6 * np.sin(time_steps_h * np.pi / 6)

    print(f"--- 开始模拟，CAES初始SOC: {caes.get_soc():.2f} ---")
    caes._record_history(0, 0, 0)

    # EMS决策循环：简单的阈值控制，低谷充电，高峰放电
    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        current_load = load[i]

        # 负荷低于100MW时，认为是低谷，全力充电
        if current_load < 100e6:
            power = caes.get_available_charge_power()
            caes.charge(power, dt_s)
        # 负荷高于250MW时，认为是高峰，全力放电
        elif current_load > 250e6:
            power = caes.get_available_discharge_power()
            caes.discharge(power, dt_s)
        else:
            caes.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    caes.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_caes()