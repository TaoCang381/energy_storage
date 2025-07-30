import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class CompressedAirEnergyStorage:
    def __init__(self,
                 storage_volume,
                 compressor_efficiency=0.85,
                 turbine_efficiency=0.88,
                 generator_efficiency=0.95,
                 ambient_temp=298,  # 环境温度 (K)
                 ambient_pressure=101325,  # 环境压力 (Pa)
                 max_pressure=7e6,  # 最大储气压力 (Pa)
                 k=1.4,  # 绝热指数
                 gas_constant=287.058,  # 气体常数 (J/(kg·K))
                 specific_heat=1005):  # 定压比热容 (J/(kg·K))
        """
        初始化压缩空气储能系统参数
        """
        self.storage_volume = storage_volume  # 储气室体积 (m³)
        self.compressor_efficiency = compressor_efficiency
        self.turbine_efficiency = turbine_efficiency
        self.generator_efficiency = generator_efficiency

        # 热力学参数
        self.ambient_temp = ambient_temp
        self.ambient_pressure = ambient_pressure
        self.max_pressure = max_pressure
        self.k = k  # 绝热指数 (空气约1.4)
        self.R = gas_constant  # 空气气体常数
        self.cp = specific_heat  # 定压比热容

        # 状态变量 - 初始温度与环境温度有微小差异，避免除以零
        self.current_pressure = ambient_pressure  # 当前储气压力 (Pa)
        self.current_temp = ambient_temp + 0.1  # 当前储气温度 (K)，增加微小温差
        self.compression_energy = 0  # 累计压缩消耗能量 (kWh)
        self.generation_energy = 0  # 累计发电输出能量 (kWh)

        # 历史记录
        self.time_history = []  # 时间记录 (s)
        self.energy_history = []  # 能量记录 (kWh)
        self.power_history = []  # 功率记录 (kW)

    def adiabatic_compression_temp(self, p1, p2, t1):
        """绝热压缩温度变化公式: T2 = T1*(P2/P1)^((k-1)/k)"""
        return t1 * (p2 / p1) ** ((self.k - 1) / self.k)

    def compression_work(self, mass_flow, p1, p2, t1):
        """压缩功耗公式 (kJ)"""
        # 等熵压缩功
        isentropic_work = (self.k / (self.k - 1)) * self.R * t1 * (
                (p2 / p1) ** ((self.k - 1) / self.k) - 1
        ) * mass_flow

        # 实际压缩功 (考虑效率)
        actual_work = isentropic_work / self.compressor_efficiency
        return actual_work / 1000  # 转换为kJ

    def adiabatic_expansion_temp(self, p1, p2, t1):
        """膨胀过程温度变化公式: T2 = T1*(P2/P1)^((k-1)/k)"""
        return t1 * (p2 / p1) ** ((self.k - 1) / self.k)

    def expansion_work(self, mass_flow, p1, p2, t1):
        """膨胀输出功公式 (kJ)"""
        # 等熵膨胀功
        isentropic_work = (self.k / (self.k - 1)) * self.R * t1 * (
                1 - (p2 / p1) ** ((self.k - 1) / self.k)
        ) * mass_flow

        # 实际输出功 (考虑效率)
        actual_work = isentropic_work * self.turbine_efficiency
        return actual_work / 1000  # 转换为kJ

    def calculate_system_efficiency(self):
        """系统总效率公式: η = 输出能量 / 输入能量"""
        if self.compression_energy == 0:
            return 0
        return self.generation_energy / self.compression_energy

    def calculate_exergy(self, pressure, temp):
        """单位质量㶲公式: e = Cp(T-T0) - T0*R*ln(P/P0)"""
        return self.cp * (temp - self.ambient_temp) - self.ambient_temp * self.R * np.log(
            pressure / self.ambient_pressure)

    def calculate_air_mass(self, pressure, temp):
        """储气室内空气质量 (kg)，基于理想气体状态方程"""
        return (pressure * self.storage_volume) / (self.R * temp)

    def compress_air(self, power, time):
        """压缩过程: 消耗电能压缩空气到储气室"""
        # 计算压缩时间 (小时)
        time_hours = time / 3600

        # 计算消耗的总能量 (kWh)
        energy_used = power * time_hours

        # 计算空气质量流量 (kg/s)，增加安全检查防止温差为零
        temp_diff = self.current_temp - self.ambient_temp
        if abs(temp_diff) < 1e-6:
            temp_diff = 1e-6 if temp_diff >= 0 else -1e-6
        mass_flow = (power * 1000) / (self.cp * temp_diff)

        # 计算压缩后的压力
        new_pressure = min(
            self.current_pressure + (energy_used * 3.6e6 * self.compressor_efficiency) / (
                        self.storage_volume * self.k / (self.k - 1)),
            self.max_pressure
        )

        # 计算压缩后的温度
        new_temp = self.adiabatic_compression_temp(self.current_pressure, new_pressure, self.current_temp)

        # 计算实际压缩功和空气质量
        mass = self.calculate_air_mass(new_pressure - self.current_pressure, (self.current_temp + new_temp) / 2)
        compression_work = self.compression_work(mass, self.current_pressure, new_pressure, self.current_temp)

        # 更新状态
        self.current_pressure = new_pressure
        self.current_temp = new_temp
        self.compression_energy += energy_used

        # 记录历史
        self._record_history(time, -energy_used, power)  # 消耗能量为负

        return {
            "energy_used": energy_used,
            "power": power,
            "final_pressure": new_pressure / 1e6,  # 转换为MPa
            "final_temp": new_temp,
            "air_mass_added": mass
        }

    def generate_power(self, power, time):
        """发电过程: 压缩空气膨胀驱动涡轮发电"""
        # 计算发电时间 (小时)
        time_hours = time / 3600

        # 计算期望输出的总能量 (kWh)
        energy_desired = power * time_hours

        # 计算所需空气质量 (考虑涡轮和发电机效率)
        required_energy = energy_desired / (self.turbine_efficiency * self.generator_efficiency)

        # 计算可释放的压力
        pressure_drop = (required_energy * 3.6e6) / (self.storage_volume * self.k / (self.k - 1))
        new_pressure = max(self.current_pressure - pressure_drop, self.ambient_pressure)

        # 计算膨胀后的温度
        new_temp = self.adiabatic_expansion_temp(self.current_pressure, new_pressure, self.current_temp)

        # 计算实际发电量
        mass = self.calculate_air_mass(self.current_pressure - new_pressure, (self.current_temp + new_temp) / 2)
        expansion_work = self.expansion_work(mass, self.current_pressure, new_pressure, self.current_temp)
        energy_produced = expansion_work * self.generator_efficiency / 1000  # 转换为kWh

        # 调整实际功率
        actual_power = self.calculate_power(energy_produced, time)

        # 更新状态
        self.current_pressure = new_pressure
        self.current_temp = new_temp
        self.generation_energy += energy_produced

        # 记录历史
        self._record_history(time, energy_produced, actual_power)

        return {
            "energy_produced": energy_produced,
            "power": actual_power,
            "final_pressure": new_pressure / 1e6,  # 转换为MPa
            "final_temp": new_temp,
            "air_mass_used": mass
        }

    def calculate_power(self, energy, time):
        """功率计算公式: P = E / t"""
        time_hours = time / 3600
        return energy / time_hours

    def _record_history(self, time, energy, power):
        """记录历史数据"""
        current_time = self.time_history[-1] + time if self.time_history else time
        self.time_history.append(current_time)

        # 累计能量 (消耗为负，产生为正)
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
        ax1.set_title('压缩空气储能系统累计能量变化')
        ax1.set_ylabel('累计能量 (kWh)')
        ax1.grid(True)

        # 功率曲线
        ax2.plot(time_hours, self.power_history, 'g-', linewidth=2)
        ax2.set_title('压缩空气储能系统功率变化')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('功率 (kW)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# 模拟压缩空气储能系统运行
def simulate_caes():
    # 创建压缩空气储能系统实例 (典型参数)
    caes = CompressedAirEnergyStorage(
        storage_volume=5000,  # 储气室体积5000m³
        compressor_efficiency=0.85,  # 压缩机效率85%
        turbine_efficiency=0.88,  # 涡轮效率88%
        generator_efficiency=0.95,  # 发电机效率95%
        max_pressure=7e6  # 最大压力7MPa
    )

    print("压缩空气储能系统参数:")
    print(f"储气室体积: {caes.storage_volume}m³")
    print(f"压缩机效率: {caes.compressor_efficiency:.0%}")
    print(f"涡轮效率: {caes.turbine_efficiency:.0%}")
    print(f"发电机效率: {caes.generator_efficiency:.0%}\n")

    # 模拟压缩过程: 1000kW功率运行6小时
    compression_result = caes.compress_air(power=1000, time=6 * 3600)
    print(f"压缩后:")
    print(f"消耗电能: {compression_result['energy_used']:.2f}kWh")
    print(f"储气压力: {compression_result['final_pressure']:.2f}MPa")
    print(f"空气温度: {compression_result['final_temp']:.2f}K")
    print(f"新增空气质量: {compression_result['air_mass_added']:.2f}kg\n")

    # 模拟发电过程: 800kW功率运行5小时
    generation_result = caes.generate_power(power=800, time=5 * 3600)
    print(f"发电后:")
    print(f"输出电能: {generation_result['energy_produced']:.2f}kWh")
    print(f"剩余压力: {generation_result['final_pressure']:.2f}MPa")
    print(f"消耗空气质量: {generation_result['air_mass_used']:.2f}kg\n")

    # 系统效率
    print(f"系统总效率: {caes.calculate_system_efficiency():.2%}")

    # 绘制性能曲线
    caes.plot_performance()


if __name__ == "__main__":
    simulate_caes()
