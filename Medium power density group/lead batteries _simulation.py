import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class LeadAcidBattery:
    def __init__(self, nominal_voltage, rated_capacity, internal_resistance, max_soc=1.0, min_soc=0.05):
        """
        初始化铅酸电池参数
        :param nominal_voltage: 标称电压 (V)
        :param rated_capacity: 额定容量 (Ah)
        :param internal_resistance: 内阻 (Ω)
        :param max_soc: 最大荷电状态 (0-1)
        :param min_soc: 最小荷电状态 (0-1)
        """
        self.nominal_voltage = nominal_voltage
        self.rated_capacity = rated_capacity
        self.internal_resistance = internal_resistance
        self.max_soc = max_soc
        self.min_soc = min_soc

        # 状态变量
        self.current_soc = max_soc  # 初始荷电状态
        self.charge_energy = 0  # 累计充电能量 (Wh)
        self.discharge_energy = 0  # 累计放电能量 (Wh)

        # 历史记录
        self.time_history = []  # 时间记录 (s)
        self.soc_history = []  # SOC记录
        self.power_history = []  # 功率记录 (W)
        self.energy_history = []  # 能量记录 (Wh)

    def calculate_capacity(self, current, time):
        """容量计算公式: Q = I × t"""
        time_hours = time / 3600  # 转换为小时
        return current * time_hours  # Ah

    def calculate_soc(self, current, time):
        """荷电状态计算公式: SOC(t) = SOC₀ - (1/C_N)∫I(t)dt"""
        capacity_change = self.calculate_capacity(current, time)
        new_soc = self.current_soc - capacity_change / self.rated_capacity

        # 限制SOC在安全范围内
        if new_soc > self.max_soc:
            new_soc = self.max_soc
        elif new_soc < self.min_soc:
            new_soc = self.min_soc

        self.current_soc = new_soc
        return new_soc

    def calculate_power(self, voltage, current):
        """功率计算公式: P = V × I"""
        return voltage * current  # W

    def calculate_energy(self, power, time):
        """能量计算公式: E = P × t"""
        time_hours = time / 3600  # 转换为小时
        return power * time_hours  # Wh

    def calculate_c_rate(self, current):
        """C-Rate计算公式: C-Rate = I / C_N"""
        return current / self.rated_capacity

    def calculate_energy_efficiency(self):
        """能量效率计算公式: EE = (E_out / E_in) × 100%"""
        if self.charge_energy == 0:
            return 0
        return (self.discharge_energy / self.charge_energy) * 100  # %

    def charge(self, current, time):
        """充电过程"""
        # 计算充电前的状态
        initial_soc = self.current_soc

        # 计算充电电压（内部使用，不对外暴露公式）
        charge_voltage = self.nominal_voltage + current * self.internal_resistance

        # 计算充电功率
        charge_power = self.calculate_power(charge_voltage, current)

        # 计算SOC变化（充电时电流为负）
        self.calculate_soc(-current, time)

        # 计算充电能量
        charge_energy = self.calculate_energy(charge_power, time)
        self.charge_energy += charge_energy

        # 记录历史数据
        self._record_history(time, charge_power, charge_energy)

        return {
            "energy_charged": charge_energy,
            "final_soc": self.current_soc,
            "power": charge_power
        }

    def discharge(self, current, time):
        """放电过程"""
        # 计算放电前的状态
        initial_soc = self.current_soc

        # 计算放电电压（内部使用，不对外暴露公式）
        discharge_voltage = self.nominal_voltage - current * self.internal_resistance

        # 计算放电功率
        discharge_power = self.calculate_power(discharge_voltage, current)

        # 计算SOC变化
        self.calculate_soc(current, time)

        # 计算放电能量
        discharge_energy = self.calculate_energy(discharge_power, time)
        self.discharge_energy += discharge_energy

        # 记录历史数据
        self._record_history(time, discharge_power, discharge_energy)

        return {
            "energy_discharged": discharge_energy,
            "final_soc": self.current_soc,
            "power": discharge_power
        }

    def _record_history(self, time, power, energy):
        """记录历史数据"""
        current_time = self.time_history[-1] + time if self.time_history else time
        self.time_history.append(current_time)
        self.soc_history.append(self.current_soc)
        self.power_history.append(power)

        # 计算累计能量
        total_energy = self.energy_history[-1] + energy if self.energy_history else energy
        self.energy_history.append(total_energy)

    def plot_performance(self):
        """绘制SOC、功率、能量随时间变化的曲线"""
        # 转换时间为小时
        time_hours = [t / 3600 for t in self.time_history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # SOC曲线
        ax1.plot(time_hours, self.soc_history, 'b-', linewidth=2)
        ax1.set_title('铅酸电池SOC变化')
        ax1.set_ylabel('荷电状态 (SOC)')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True)

        # 功率曲线
        ax2.plot(time_hours, self.power_history, 'g-', linewidth=2)
        ax2.set_title('铅酸电池功率变化')
        ax2.set_ylabel('功率 (W)')
        ax2.grid(True)

        # 能量曲线
        ax3.plot(time_hours, self.energy_history, 'r-', linewidth=2)
        ax3.set_title('铅酸电池累计能量变化')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('累计能量 (Wh)')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


# 模拟铅酸电池运行
def simulate_lead_acid_battery():
    # 创建铅酸电池实例（典型参数）
    battery = LeadAcidBattery(
        nominal_voltage=12,  # 12V铅酸电池
        rated_capacity=100,  # 额定容量100Ah
        internal_resistance=0.03  # 内阻0.03Ω
    )

    print("铅酸电池参数:")
    print(f"标称电压: {battery.nominal_voltage}V")
    print(f"额定容量: {battery.rated_capacity}Ah")
    print(f"额定能量: {battery.nominal_voltage * battery.rated_capacity:.2f}Wh\n")

    # 模拟充电过程: 20A电流充电3小时
    charge_result = battery.charge(current=20, time=3 * 3600)
    print(f"充电后:")
    print(f"充入能量: {charge_result['energy_charged']:.2f}Wh")
    print(f"当前SOC: {charge_result['final_soc']:.2%}")
    print(f"充电功率: {charge_result['power']:.2f}W")
    print(f"充电C-Rate: {battery.calculate_c_rate(20):.2f}C\n")

    # 模拟放电过程: 15A电流放电4小时
    discharge_result = battery.discharge(current=15, time=4 * 3600)
    print(f"放电后:")
    print(f"放出能量: {discharge_result['energy_discharged']:.2f}Wh")
    print(f"当前SOC: {discharge_result['final_soc']:.2%}")
    print(f"放电功率: {discharge_result['power']:.2f}W")
    print(f"放电C-Rate: {battery.calculate_c_rate(15):.2f}C\n")

    # 计算能量效率
    print(f"能量效率: {battery.calculate_energy_efficiency():.2f}%")

    # 绘制性能曲线
    battery.plot_performance()


if __name__ == "__main__":
    simulate_lead_acid_battery()
