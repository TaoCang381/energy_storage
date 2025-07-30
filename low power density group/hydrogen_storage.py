import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class HydrogenEnergyStorage:
    def __init__(self, electrolyzer_efficiency, storage_efficiency, fuel_cell_efficiency,
                 hhv=33.3, lhv=28.9, initial_hydrogen=0):
        """
        初始化氢储能系统参数
        :param electrolyzer_efficiency: 电解槽效率 (0-1)
        :param storage_efficiency: 储氢效率 (0-1)
        :param fuel_cell_efficiency: 燃料电池效率 (0-1)
        :param hhv: 氢气高热值 (kWh/kg)，默认33.3
        :param lhv: 氢气低热值 (kWh/kg)，默认28.9
        :param initial_hydrogen: 初始氢气质量 (kg)
        """
        self.electrolyzer_efficiency = electrolyzer_efficiency
        self.storage_efficiency = storage_efficiency
        self.fuel_cell_efficiency = fuel_cell_efficiency
        self.hhv = hhv  # 高热值
        self.lhv = lhv  # 低热值

        # 状态变量
        self.current_hydrogen = initial_hydrogen  # 当前氢气质量 (kg)
        self.electrolysis_energy = 0  # 累计电解消耗能量 (kWh)
        self.fuel_cell_energy = 0  # 累计燃料电池输出能量 (kWh)
        self.combustion_heat = 0  # 累计燃烧放热 (kWh)

        # 历史记录
        self.time_history = []  # 时间记录 (s)
        self.energy_history = []  # 能量记录 (kWh)
        self.power_history = []  # 功率记录 (kW)

    def calculate_electrolysis_energy(self, hydrogen_mass):
        """电解制氢能耗公式: E电解 = (m·HHV) / η电解"""
        return (hydrogen_mass * self.hhv) / self.electrolyzer_efficiency  # kWh

    def calculate_fuel_cell_energy(self, hydrogen_mass):
        """燃料电池发电能量公式: E发电 = m·LHV·η燃料电池"""
        # 考虑储氢效率损失
        usable_hydrogen = hydrogen_mass * self.storage_efficiency
        return usable_hydrogen * self.lhv * self.fuel_cell_efficiency  # kWh

    def calculate_system_efficiency(self):
        """系统总效率公式: η总 = η电解·η储存·η燃料电池"""
        return self.electrolyzer_efficiency * self.storage_efficiency * self.fuel_cell_efficiency

    def calculate_combustion_heat(self, hydrogen_mass):
        """氢燃烧放热公式: Q = m·HHV"""
        return hydrogen_mass * self.hhv  # kWh

    def calculate_power(self, energy, time):
        """功率计算公式: P = E / t"""
        time_hours = time / 3600  # 转换为小时
        return energy / time_hours  # kW

    def produce_hydrogen(self, power, time):
        """制氢过程: 电解水产生氢气"""
        # 计算消耗的能量
        energy_used = power * (time / 3600)  # kWh

        # 计算产生的氢气质量
        hydrogen_produced = (energy_used * self.electrolyzer_efficiency) / self.hhv

        # 更新状态
        self.current_hydrogen += hydrogen_produced
        self.electrolysis_energy += energy_used

        # 记录历史
        self._record_history(time, -energy_used, power)  # 消耗能量为负

        return {
            "hydrogen_produced": hydrogen_produced,
            "energy_used": energy_used,
            "power": power,
            "total_hydrogen": self.current_hydrogen
        }

    def generate_electricity(self, power, time):
        """发电过程: 燃料电池发电"""
        # 计算需要的能量
        energy_needed = power * (time / 3600)  # kWh

        # 计算需要的氢气质量
        required_hydrogen = energy_needed / (self.lhv * self.fuel_cell_efficiency * self.storage_efficiency)

        # 限制在可用氢气范围内
        if required_hydrogen > self.current_hydrogen:
            required_hydrogen = self.current_hydrogen
            energy_produced = self.calculate_fuel_cell_energy(required_hydrogen)
            power = self.calculate_power(energy_produced, time)
        else:
            energy_produced = energy_needed

        # 更新状态
        self.current_hydrogen -= required_hydrogen
        self.fuel_cell_energy += energy_produced

        # 记录历史
        self._record_history(time, energy_produced, power)

        return {
            "energy_produced": energy_produced,
            "hydrogen_used": required_hydrogen,
            "power": power,
            "remaining_hydrogen": self.current_hydrogen
        }

    def combust_hydrogen(self, mass):
        """燃烧过程: 氢气燃烧放热"""
        # 限制燃烧量
        mass_burned = min(mass, self.current_hydrogen)

        # 计算释放的热量
        heat_released = self.calculate_combustion_heat(mass_burned)
        self.combustion_heat += heat_released

        # 更新状态
        self.current_hydrogen -= mass_burned

        return {
            "heat_released": heat_released,
            "hydrogen_burned": mass_burned,
            "remaining_hydrogen": self.current_hydrogen
        }

    def _record_history(self, time, energy, power):
        """记录历史数据"""
        current_time = self.time_history[-1] + time if self.time_history else time
        self.time_history.append(current_time)

        # 累计能量（消耗为负，产生为正）
        total_energy = self.energy_history[-1] + energy if self.energy_history else energy
        self.energy_history.append(total_energy)

        self.power_history.append(power)

    def plot_performance(self):
        """绘制能量和功率随时间变化的曲线"""
        # 转换时间为小时
        time_hours = [t / 3600 for t in self.time_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 能量曲线
        ax1.plot(time_hours, self.energy_history, 'g-', linewidth=2)
        ax1.set_title('氢储能系统累计能量变化')
        ax1.set_ylabel('累计能量 (kWh)')
        ax1.grid(True)

        # 功率曲线
        ax2.plot(time_hours, self.power_history, 'b-', linewidth=2)
        ax2.set_title('氢储能系统功率变化')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('功率 (kW)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# 模拟氢储能系统运行
def simulate_hydrogen_storage():
    # 创建氢储能系统实例（典型参数）
    hydrogen_storage = HydrogenEnergyStorage(
        electrolyzer_efficiency=0.7,  # 电解槽效率70%
        storage_efficiency=0.95,  # 储氢效率95%
        fuel_cell_efficiency=0.6  # 燃料电池效率60%
    )

    print("氢储能系统参数:")
    print(f"电解槽效率: {hydrogen_storage.electrolyzer_efficiency:.0%}")
    print(f"储氢效率: {hydrogen_storage.storage_efficiency:.0%}")
    print(f"燃料电池效率: {hydrogen_storage.fuel_cell_efficiency:.0%}")
    print(f"系统总效率: {hydrogen_storage.calculate_system_efficiency():.0%}\n")

    # 模拟制氢过程: 50kW功率运行8小时
    production_result = hydrogen_storage.produce_hydrogen(power=50, time=8 * 3600)
    print(f"制氢后:")
    print(f"消耗电能: {production_result['energy_used']:.2f}kWh")
    print(f"产生氢气: {production_result['hydrogen_produced']:.2f}kg")
    print(f"总储氢量: {production_result['total_hydrogen']:.2f}kg\n")

    # 模拟发电过程: 30kW功率运行10小时
    generation_result = hydrogen_storage.generate_electricity(power=30, time=10 * 3600)
    print(f"发电后:")
    print(f"输出电能: {generation_result['energy_produced']:.2f}kWh")
    print(f"消耗氢气: {generation_result['hydrogen_used']:.2f}kg")
    print(f"剩余氢气: {generation_result['remaining_hydrogen']:.2f}kg\n")

    # 模拟燃烧过程: 燃烧剩余氢气
    combustion_result = hydrogen_storage.combust_hydrogen(generation_result['remaining_hydrogen'])
    print(f"燃烧后:")
    print(f"释放热量: {combustion_result['heat_released']:.2f}kWh")
    print(f"燃烧氢气: {combustion_result['hydrogen_burned']:.2f}kg\n")

    # 绘制性能曲线
    hydrogen_storage.plot_performance()


if __name__ == "__main__":
    simulate_hydrogen_storage()
