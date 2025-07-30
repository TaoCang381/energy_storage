import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class PumpedStorageSystem:
    def __init__(self, head, max_volume, pump_efficiency, turbine_efficiency, density=1000, gravity=9.81):
        """
        初始化抽水蓄能系统参数
        :param head: 水位落差 (m)
        :param max_volume: 最大蓄水量 (m³)
        :param pump_efficiency: 抽水效率 (0-1)
        :param turbine_efficiency: 发电效率 (0-1)
        :param density: 水的密度 (kg/m³)，默认1000
        :param gravity: 重力加速度 (m/s²)，默认9.81
        """
        self.head = head
        self.max_volume = max_volume
        self.pump_efficiency = pump_efficiency
        self.turbine_efficiency = turbine_efficiency
        self.density = density
        self.gravity = gravity

        # 状态变量
        self.current_volume = 0  # 当前蓄水量 (m³)
        self.pumping_energy = 0  # 累计抽水消耗能量 (kWh)
        self.generating_energy = 0  # 累计发电输出能量 (kWh)

        # 历史记录
        self.time_history = []  # 时间记录 (s)
        self.energy_history = []  # 能量记录 (kWh)
        self.power_history = []  # 功率记录 (kW)

    def calculate_water_mass(self, volume=None):
        """水体质量公式: m = ρ·V"""
        vol = volume if volume is not None else self.current_volume
        return self.density * vol  # kg

    def calculate_potential_energy(self, volume=None):
        """重力势能公式: E势 = m·g·h (转换为kWh)"""
        mass = self.calculate_water_mass(volume)
        energy_joule = mass * self.gravity * self.head  # J
        return energy_joule / 3.6e6  # 转换为kWh

    def calculate_pumping_energy(self, volume):
        """抽水能量公式: E抽 = E势 / η抽"""
        potential_energy = self.calculate_potential_energy(volume)
        return potential_energy / self.pump_efficiency  # kWh

    def calculate_generating_energy(self, volume):
        """发电能量公式: E发 = E势 · η发"""
        potential_energy = self.calculate_potential_energy(volume)
        return potential_energy * self.turbine_efficiency  # kWh

    def calculate_overall_efficiency(self):
        """综合效率公式: η总 = η抽 · η发"""
        return self.pump_efficiency * self.turbine_efficiency

    def calculate_power(self, energy, time):
        """功率公式: P = E / t"""
        time_hours = time / 3600  # 转换为小时
        return energy / time_hours  # kW

    def calculate_flow_rate_power(self, flow_rate, is_generating=True):
        """流量与功率关系: P发 = ρ·g·Q·h·η发 或 P抽 = ρ·g·Q·h / η抽"""
        # 转换流量为m³/s
        flow_rate_per_sec = flow_rate / 3600  # m³/h -> m³/s

        # 计算功率 (W)
        if is_generating:
            power_watt = self.density * self.gravity * flow_rate_per_sec * self.head * self.turbine_efficiency
        else:
            power_watt = self.density * self.gravity * flow_rate_per_sec * self.head / self.pump_efficiency

        return power_watt / 1000  # 转换为kW

    def pump_water(self, flow_rate, time):
        """抽水过程: 将水从低位抽到高位"""
        # 计算抽水量 (m³)
        volume_pumped = (flow_rate / 3600) * time  # 流量(m³/h) * 时间(h)
        new_volume = self.current_volume + volume_pumped

        # 限制在最大蓄水量内
        if new_volume > self.max_volume:
            volume_pumped = self.max_volume - self.current_volume
            new_volume = self.max_volume

        if volume_pumped <= 0:
            return {"energy_used": 0, "power": 0, "final_volume": self.current_volume}

        # 计算消耗能量和功率
        energy_used = self.calculate_pumping_energy(volume_pumped)
        power = self.calculate_power(energy_used, time)

        # 更新状态
        self.current_volume = new_volume
        self.pumping_energy += energy_used

        # 记录历史
        self._record_history(time, energy_used, power)

        return {
            "energy_used": energy_used,
            "power": power,
            "final_volume": new_volume,
            "potential_energy": self.calculate_potential_energy()
        }

    def generate_power(self, flow_rate, time):
        """发电过程: 高位水流下驱动涡轮发电"""
        # 计算放水量 (m³)
        volume_released = (flow_rate / 3600) * time
        new_volume = self.current_volume - volume_released

        # 限制在最小蓄水量内
        if new_volume < 0:
            volume_released = self.current_volume
            new_volume = 0

        if volume_released <= 0:
            return {"energy_produced": 0, "power": 0, "final_volume": self.current_volume}

        # 计算产生能量和功率
        energy_produced = self.calculate_generating_energy(volume_released)
        power = self.calculate_power(energy_produced, time)

        # 更新状态
        self.current_volume = new_volume
        self.generating_energy += energy_produced

        # 记录历史
        self._record_history(time, energy_produced, power)

        return {
            "energy_produced": energy_produced,
            "power": power,
            "final_volume": new_volume,
            "potential_energy": self.calculate_potential_energy()
        }

    def _record_history(self, time, energy, power):
        """记录历史数据"""
        current_time = self.time_history[-1] + time if self.time_history else time
        self.time_history.append(current_time)

        # 累计能量（抽水为负，发电为正）
        total_energy = self.energy_history[-1] + energy if self.energy_history else energy
        self.energy_history.append(total_energy)

        self.power_history.append(power)

    def plot_performance(self):
        """绘制能量和功率随时间变化的曲线"""
        # 转换时间为小时
        time_hours = [t / 3600 for t in self.time_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 能量曲线
        ax1.plot(time_hours, self.energy_history, 'b-', linewidth=2)
        ax1.set_title('抽水蓄能系统累计能量变化')
        ax1.set_ylabel('累计能量 (kWh)')
        ax1.grid(True)

        # 功率曲线
        ax2.plot(time_hours, self.power_history, 'g-', linewidth=2)
        ax2.set_title('抽水蓄能系统功率变化')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('功率 (kW)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# 模拟抽水蓄能系统运行
def simulate_pumped_storage():
    # 创建抽水蓄能系统实例（典型参数）
    storage = PumpedStorageSystem(
        head=300,  # 水位落差300米
        max_volume=100000,  # 最大蓄水量100,000立方米
        pump_efficiency=0.85,  # 抽水效率85%
        turbine_efficiency=0.9  # 发电效率90%
    )

    print("抽水蓄能系统参数:")
    print(f"水位落差: {storage.head}m")
    print(f"最大蓄水量: {storage.max_volume}m³")
    print(f"抽水效率: {storage.pump_efficiency:.0%}")
    print(f"发电效率: {storage.turbine_efficiency:.0%}")
    print(f"综合效率: {storage.calculate_overall_efficiency():.0%}\n")

    # 模拟抽水过程: 流量5000m³/h，持续10小时
    pump_result = storage.pump_water(flow_rate=5000, time=10 * 3600)
    print(f"抽水后:")
    print(f"消耗能量: {pump_result['energy_used']:.2f}kWh")
    print(f"抽水功率: {pump_result['power']:.2f}kW")
    print(f"蓄水量: {pump_result['final_volume']:.0f}m³")
    print(f"储存势能: {pump_result['potential_energy']:.2f}kWh\n")

    # 模拟发电过程: 流量4000m³/h，持续12小时
    generate_result = storage.generate_power(flow_rate=4000, time=12 * 3600)
    print(f"发电后:")
    print(f"产生能量: {generate_result['energy_produced']:.2f}kWh")
    print(f"发电功率: {generate_result['power']:.2f}kW")
    print(f"剩余蓄水量: {generate_result['final_volume']:.0f}m³")
    print(f"剩余势能: {generate_result['potential_energy']:.2f}kWh\n")

    # 绘制性能曲线
    storage.plot_performance()


if __name__ == "__main__":
    simulate_pumped_storage()
