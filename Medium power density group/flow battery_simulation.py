import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class FlowBattery:
    def __init__(self, nominal_voltage, rated_capacity, max_flow_rate, internal_resistance):
        """
        初始化液流电池参数
        :param nominal_voltage: 标称电压 (V)
        :param rated_capacity: 额定容量 (Ah)
        :param max_flow_rate: 最大流量 (L/min)
        :param internal_resistance: 内阻 (Ω)
        """
        self.nominal_voltage = nominal_voltage
        self.rated_capacity = rated_capacity
        self.max_flow_rate = max_flow_rate
        self.internal_resistance = internal_resistance

        # 状态变量
        self.current_soc = 1.0  # 初始荷电状态(0-1)
        self.current_voltage = nominal_voltage
        self.flow_rate = 0  # 当前流量
        self.charge_energy = 0  # 累计充电能量
        self.discharge_energy = 0  # 累计放电能量

        # 历史记录
        self.time_history = []
        self.soc_history = []
        self.voltage_history = []

    def calculate_energy(self, voltage=None, capacity=None):
        """能量计算公式: E = V_avg × Q"""
        v = voltage if voltage is not None else self.nominal_voltage
        q = capacity if capacity is not None else self.rated_capacity
        return v * q  # Wh

    def calculate_capacity(self, current, time):
        """容量计算公式: Q = I × t"""
        # 转换时间为小时
        time_hours = time / 3600
        return current * time_hours  # Ah

    def calculate_soc(self, current, time):
        """荷电状态计算公式: SOC_n = SOC_{n-1} - (I×Δt)/Q_N"""
        capacity_change = self.calculate_capacity(current, time)
        self.current_soc -= capacity_change / self.rated_capacity

        # 限制SOC在0-1范围内
        if self.current_soc < 0:
            self.current_soc = 0
        elif self.current_soc > 1:
            self.current_soc = 1

        return self.current_soc

    def calculate_power(self, voltage=None, current=None):
        """功率计算公式: P = V × I"""
        v = voltage if voltage is not None else self.current_voltage
        i = current if current is not None else 0
        return v * i  # W

    def calculate_c_rate(self, current):
        """C-Rate计算公式: I = C-Rate × C_N"""
        return current / self.rated_capacity

    def calculate_energy_efficiency(self):
        """能量效率计算公式: EE = (E_out / E_in) × 100%"""
        if self.charge_energy == 0:
            return 0
        return (self.discharge_energy / self.charge_energy) * 100  # %

    def set_flow_rate(self, rate):
        """设置电解液流量，影响功率输出能力"""
        self.flow_rate = min(rate, self.max_flow_rate)
        return self.flow_rate

    def charge(self, current, time, flow_rate=None):
        """充电过程"""
        if flow_rate:
            self.set_flow_rate(flow_rate)

        # 计算充电前的SOC
        initial_soc = self.current_soc

        # 计算充电容量和SOC变化（充电时电流为负）
        self.calculate_soc(-current, time)

        # 计算充电电压（考虑内阻影响）
        self.current_voltage = self.nominal_voltage + current * self.internal_resistance

        # 计算充电能量
        charge_power = self.calculate_power(self.current_voltage, current)
        charge_energy = charge_power * (time / 3600)  # 转换为Wh
        self.charge_energy += charge_energy

        # 记录历史数据
        self._record_history(time, charge=True)

        return {
            "energy_charged": charge_energy,
            "final_soc": self.current_soc,
            "voltage": self.current_voltage
        }

    def discharge(self, current, time, flow_rate=None):
        """放电过程"""
        if flow_rate:
            self.set_flow_rate(flow_rate)

        # 计算放电前的SOC
        initial_soc = self.current_soc

        # 计算放电容量和SOC变化
        self.calculate_soc(current, time)

        # 计算放电电压（考虑内阻影响）
        self.current_voltage = self.nominal_voltage - current * self.internal_resistance

        # 计算放电能量
        discharge_power = self.calculate_power(self.current_voltage, current)
        discharge_energy = discharge_power * (time / 3600)  # 转换为Wh
        self.discharge_energy += discharge_energy

        # 记录历史数据
        self._record_history(time, charge=False)

        return {
            "energy_discharged": discharge_energy,
            "final_soc": self.current_soc,
            "voltage": self.current_voltage
        }

    def _record_history(self, time, charge):
        """记录历史数据"""
        current_time = self.time_history[-1] + time if self.time_history else time
        self.time_history.append(current_time)
        self.soc_history.append(self.current_soc)
        self.voltage_history.append(self.current_voltage)

    def plot_performance(self):
        """绘制电池性能曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # 转换时间为小时
        time_hours = [t / 3600 for t in self.time_history]

        # SOC曲线
        ax1.plot(time_hours, self.soc_history, 'b-', linewidth=2)
        ax1.set_title('液流电池SOC变化')
        ax1.set_ylabel('荷电状态 (SOC)')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True)

        # 电压曲线
        ax2.plot(time_hours, self.voltage_history, 'r-', linewidth=2)
        ax2.set_title('液流电池电压变化')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('电压 (V)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# 模拟液流电池运行
def simulate_flow_battery():
    # 创建液流电池实例（全钒液流电池典型参数）
    battery = FlowBattery(
        nominal_voltage=1.4,  # 标称电压1.4V
        rated_capacity=500,  # 额定容量500Ah
        max_flow_rate=50,  # 最大流量50L/min
        internal_resistance=0.02  # 内阻0.02Ω
    )

    print("液流电池参数:")
    print(f"标称电压: {battery.nominal_voltage}V")
    print(f"额定容量: {battery.rated_capacity}Ah")
    print(f"额定能量: {battery.calculate_energy():.2f}Wh")
    print(f"最大理论功率: {battery.calculate_power(current=100):.2f}W\n")

    # 模拟充电过程: 100A电流充电3小时，流量30L/min
    charge_result = battery.charge(current=100, time=3 * 3600, flow_rate=30)
    print(f"充电后:")
    print(f"充入能量: {charge_result['energy_charged']:.2f}Wh")
    print(f"当前SOC: {charge_result['final_soc']:.2%}")
    print(f"充电电压: {charge_result['voltage']:.2f}V\n")

    # 模拟放电过程: 80A电流放电3小时，流量40L/min
    discharge_result = battery.discharge(current=80, time=3 * 3600, flow_rate=40)
    print(f"放电后:")
    print(f"放出能量: {discharge_result['energy_discharged']:.2f}Wh")
    print(f"当前SOC: {discharge_result['final_soc']:.2%}")
    print(f"放电电压: {discharge_result['voltage']:.2f}V\n")

    # 计算能量效率
    print(f"能量效率: {battery.calculate_energy_efficiency():.2f}%")
    print(f"当前C-Rate: {battery.calculate_c_rate(80):.2f}C")

    # 绘制性能曲线
    battery.plot_performance()


if __name__ == "__main__":
    simulate_flow_battery()
