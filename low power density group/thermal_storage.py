import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class ThermalEnergyStorage:
    def __init__(self, mass, specific_heat, phase_change_temp, latent_heat,
                 surface_area, heat_transfer_coeff, initial_temp=20, env_temp=25):
        """
        初始化热储能系统参数
        :param mass: 储能介质质量 (kg)
        :param specific_heat: 比热容 (J/(kg·K))
        :param phase_change_temp: 相变温度 (℃)
        :param latent_heat: 相变潜热 (J/kg)
        :param surface_area: 传热面积 (m²)
        :param heat_transfer_coeff: 总传热系数 (W/(m²·K))
        :param initial_temp: 初始温度 (℃)
        :param env_temp: 环境温度 (℃)
        """
        self.mass = mass
        self.specific_heat = specific_heat
        self.phase_change_temp = phase_change_temp
        self.latent_heat = latent_heat
        self.surface_area = surface_area
        self.heat_transfer_coeff = heat_transfer_coeff

        # 状态变量
        self.current_temp = initial_temp  # 当前温度 (℃)
        self.phase_change_complete = False  # 相变是否完成
        self.stored_energy = 0  # 累计储存能量 (kJ)
        self.released_energy = 0  # 累计释放能量 (kJ)
        self.energy_loss = 0  # 累计热损失 (kJ)

        # 环境参数
        self.env_temp = env_temp

        # 历史记录
        self.time_history = []  # 时间记录 (s)
        self.energy_history = []  # 能量记录 (kJ)
        self.power_history = []  # 功率记录 (kW)

    def calculate_sensible_heat(self, delta_T):
        """显热计算公式: Q = m·c·ΔT"""
        heat_joule = self.mass * self.specific_heat * delta_T  # J
        return heat_joule / 1000  # 转换为kJ

    def calculate_latent_heat(self, mass_fraction=1.0):
        """潜热计算公式: Q = m·L"""
        heat_joule = self.mass * mass_fraction * self.latent_heat  # J
        return heat_joule / 1000  # 转换为kJ

    def calculate_total_heat(self, delta_T, in_phase_change=False):
        """总储热量公式: Q总 = Q显 + Q潜"""
        sensible = self.calculate_sensible_heat(delta_T)
        if in_phase_change:
            latent = self.calculate_latent_heat()
            return sensible + latent
        return sensible

    def calculate_heat_loss(self, time):
        """热损失公式: Q损 = U·A·ΔT·Δt"""
        delta_T = abs(self.current_temp - self.env_temp)
        heat_loss_joule = self.heat_transfer_coeff * self.surface_area * delta_T * time  # J
        return heat_loss_joule / 1000  # 转换为kJ

    def calculate_power(self, energy, time):
        """功率计算公式: P = Q / Δt"""
        time_hours = time / 3600  # 转换为小时
        return energy / time_hours  # kW

    def calculate_thermal_efficiency(self):
        """热效率公式: η = Q输出 / Q输入 × 100%"""
        if self.stored_energy == 0:
            return 0
        return (self.released_energy / self.stored_energy) * 100  # %

    def calculate_heat_density(self, volume):
        """储热密度公式: ρq = Q / V"""
        total_heat = self.stored_energy - self.energy_loss  # kJ
        return (total_heat * 1000) / volume  # J/m³ (转换为J)

    def charge(self, target_temp, time):
        """储热过程: 加热介质储存热量"""
        # 计算温度变化
        initial_temp = self.current_temp
        delta_T = target_temp - initial_temp

        if delta_T <= 0:
            return {"energy_stored": 0, "power": 0, "final_temp": self.current_temp}

        # 计算相变影响
        in_phase_change = False
        if (initial_temp < self.phase_change_temp < target_temp):
            in_phase_change = True
            self.phase_change_complete = True

        # 计算总储热量
        total_heat = self.calculate_total_heat(delta_T, in_phase_change)

        # 计算热损失
        heat_loss = self.calculate_heat_loss(time)
        self.energy_loss += heat_loss

        # 实际储存的能量（扣除损失）
        energy_stored = total_heat - heat_loss
        self.stored_energy += energy_stored

        # 更新温度
        self.current_temp = min(target_temp, self.current_temp + delta_T)

        # 计算功率
        power = self.calculate_power(total_heat, time)

        # 记录历史
        self._record_history(time, energy_stored, power)

        return {
            "energy_stored": energy_stored,
            "power": power,
            "heat_loss": heat_loss,
            "final_temp": self.current_temp,
            "phase_change_complete": self.phase_change_complete
        }

    def discharge(self, target_temp, time):
        """释热过程: 释放储存的热量"""
        # 计算温度变化
        initial_temp = self.current_temp
        delta_T = initial_temp - target_temp

        if delta_T <= 0:
            return {"energy_released": 0, "power": 0, "final_temp": self.current_temp}

        # 计算相变影响
        in_phase_change = False
        if (target_temp < self.phase_change_temp < initial_temp):
            in_phase_change = True

        # 计算可释放的热量
        releasable_heat = self.calculate_total_heat(delta_T, in_phase_change)

        # 计算热损失
        heat_loss = self.calculate_heat_loss(time)
        self.energy_loss += heat_loss

        # 实际释放的能量（扣除损失）
        energy_released = releasable_heat - heat_loss
        self.released_energy += energy_released

        # 更新温度
        self.current_temp = max(target_temp, self.current_temp - delta_T)

        # 计算功率
        power = self.calculate_power(releasable_heat, time)

        # 记录历史
        self._record_history(time, -energy_released, power)  # 释放能量为负

        return {
            "energy_released": energy_released,
            "power": power,
            "heat_loss": heat_loss,
            "final_temp": self.current_temp
        }

    def _record_history(self, time, energy, power):
        """记录历史数据"""
        current_time = self.time_history[-1] + time if self.time_history else time
        self.time_history.append(current_time)

        # 累计能量（储热为正，释热为负）
        total_energy = self.energy_history[-1] + energy if self.energy_history else energy
        self.energy_history.append(total_energy)

        self.power_history.append(power)

    def plot_performance(self):
        """绘制能量和功率随时间变化的曲线"""
        # 转换时间为小时
        time_hours = [t / 3600 for t in self.time_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 能量曲线
        ax1.plot(time_hours, self.energy_history, 'r-', linewidth=2)
        ax1.set_title('热储能系统累计能量变化')
        ax1.set_ylabel('累计能量 (kJ)')
        ax1.grid(True)

        # 功率曲线
        ax2.plot(time_hours, self.power_history, 'orange', linewidth=2)
        ax2.set_title('热储能系统功率变化')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('功率 (kW)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# 模拟热储能系统运行
def simulate_thermal_storage():
    # 创建热储能系统实例（以相变材料为例）
    thermal_storage = ThermalEnergyStorage(
        mass=1000,  # 介质质量1000kg
        specific_heat=1800,  # 比热容1800 J/(kg·K)
        phase_change_temp=80,  # 相变温度80℃
        latent_heat=200000,  # 相变潜热200,000 J/kg
        surface_area=10,  # 传热面积10m²
        heat_transfer_coeff=5,  # 传热系数5 W/(m²·K)
        initial_temp=20,  # 初始温度20℃
        env_temp=25  # 环境温度25℃
    )

    print("热储能系统参数:")
    print(f"介质质量: {thermal_storage.mass}kg")
    print(f"相变温度: {thermal_storage.phase_change_temp}℃")
    print(f"相变潜热: {thermal_storage.latent_heat / 1000}kJ/kg")
    print(f"初始温度: {thermal_storage.current_temp}℃\n")

    # 模拟储热过程: 加热到100℃，持续5小时
    charge_result = thermal_storage.charge(target_temp=100, time=5 * 3600)
    print(f"储热后:")
    print(f"储存能量: {charge_result['energy_stored']:.2f}kJ")
    print(f"储热功率: {charge_result['power']:.2f}kW")
    print(f"热损失: {charge_result['heat_loss']:.2f}kJ")
    print(f"最终温度: {charge_result['final_temp']}℃")
    print(f"相变完成: {charge_result['phase_change_complete']}\n")

    # 模拟释热过程: 冷却到30℃，持续6小时
    discharge_result = thermal_storage.discharge(target_temp=30, time=6 * 3600)
    print(f"释热后:")
    print(f"释放能量: {discharge_result['energy_released']:.2f}kJ")
    print(f"释热功率: {discharge_result['power']:.2f}kW")
    print(f"热损失: {discharge_result['heat_loss']:.2f}kJ")
    print(f"最终温度: {discharge_result['final_temp']}℃\n")

    # 计算热效率
    print(f"热效率: {thermal_storage.calculate_thermal_efficiency():.2f}%")

    # 绘制性能曲线
    thermal_storage.plot_performance()


if __name__ == "__main__":
    simulate_thermal_storage()
