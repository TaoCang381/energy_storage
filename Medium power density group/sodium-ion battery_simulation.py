import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class SodiumIonBattery:
    def __init__(self, nominal_voltage, capacity, internal_resistance):
        # 初始化钠离子电池参数
        self.nominal_voltage = nominal_voltage  # 标称电压 (V)
        self.capacity = capacity  # 容量 (Ah)
        self.internal_resistance = internal_resistance  # 内阻 (Ω)
        self.initial_soc = 1.0  # 初始荷电状态 (0-1)
        self.current_soc = self.initial_soc  # 当前荷电状态 (0-1)
        self.current = 0  # 当前电流 (A)
        self.time = 0  # 当前时间 (s)

    def calculate_energy(self):
        # 计算能量公式: E = V_nom * Q
        return self.nominal_voltage * self.capacity

    def calculate_capacity(self, current, time):
        # 计算容量公式: Q = I * t
        return current * time

    def calculate_soc(self, current, time):
        # 计算荷电状态公式: SOC(t) = SOC₀ - (1/C_N) ∫I(t)dt
        charge_change = (current * time) / 3600  # 转换为 Ah
        self.current_soc = self.current_soc - charge_change / self.capacity
        if self.current_soc < 0:
            self.current_soc = 0
        return self.current_soc

    def calculate_c_rate(self, current):
        # 计算 C-Rate 公式: I = C-Rate * C_N
        return current / self.capacity

    def calculate_terminal_voltage(self, ocv):
        # 计算端电压公式: V_t = OCV - I * R_int
        return ocv - self.current * self.internal_resistance

    def charge(self, current, time):
        # 充电方法
        self.current = current
        self.time = time
        self.calculate_soc(-current, time)  # 负电流表示充电

    def discharge(self, current, time):
        # 放电方法
        self.current = current
        self.time = time
        self.calculate_soc(current, time)  # 正电流表示放电


# 创建钠离子电池实例
battery = SodiumIonBattery(
    nominal_voltage=3.2,  # 标称电压 (V)
    capacity=2.0,  # 容量 (Ah)
    internal_resistance=0.02  # 内阻 (Ω)
)

# 模拟充电过程
battery.charge(current=1.0, time=3600)  # 1A 充电 1 小时
print(f"充电后 SOC: {battery.current_soc:.2f}")
print(f"充电后容量: {battery.calculate_capacity(battery.current, battery.time / 3600):.2f} Ah")

# 模拟放电过程
battery.discharge(current=1.0, time=3600)  # 1A 放电 1 小时
print(f"放电后 SOC: {battery.current_soc:.2f}")
print(f"放电后容量: {battery.calculate_capacity(battery.current, battery.time / 3600):.2f} Ah")

# 计算 C-Rate
c_rate = battery.calculate_c_rate(battery.current)
print(f"C-Rate: {c_rate:.2f} C")

# 计算端电压
ocv = 3.4  # 开路电压 (V)
terminal_voltage = battery.calculate_terminal_voltage(ocv)
print(f"端电压: {terminal_voltage:.2f} V")

# 绘制 SOC 变化曲线
times = np.linspace(0, 3600, 100)  # 0 到 3600 秒
soc_values = []

for t in times:
    battery.calculate_soc(battery.current, t)
    soc_values.append(battery.current_soc)

plt.figure(figsize=(10, 6))
plt.plot(times / 3600, soc_values)
plt.title('钠离子电池 SOC 变化')
plt.xlabel('时间 (小时)')
plt.ylabel('SOC')
plt.grid(True)
plt.show()