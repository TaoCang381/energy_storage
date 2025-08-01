import math
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class FlywheelModel:
    """
    飞轮储能系统模型 (HESS集成版)
    增加了动态可用功率、SOH、成本等接口，适用于混合储能系统能源管理策略(EMS)调用。
    """

    def __init__(self,
                 # --- 基础物理参数 ---
                 radius=0.5,  # 飞轮半径 (m)
                 mass=500,  # 飞轮质量 (kg)
                 max_angular_vel=1000,  # 设计最大角速度 (rad/s)

                 # --- 性能与效率参数 ---
                 efficiency=0.9,  # 充放电往返效率
                 friction_coeff=0.01,  # 自放电相关的摩擦系数 (单位: 1/s)
                 rated_power=5000,  # 额定功率 (W)
                 max_power=7500,  # 最大瞬时功率 (W)
                 response_time=0.01,  # 响应时间 (s)

                 # --- HESS集成新增参数 ---
                 ess_id="flywheel_01",  # 唯一标识符
                 initial_soh=1.0,  # 初始健康状态 (1.0代表全新)
                 initial_soc=0.0,  # 初始荷电状态
                 cost_per_kwh=0.05,  # 度电成本（用于经济调度，飞轮成本较低）
                 soc_upper_limit=0.95,  # SOC运行上限，保护性限制
                 soc_lower_limit=0.05  # SOC运行下限，保护性限制
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh
        self.cost_per_kwh = cost_per_kwh
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit

        # --- 基础物理和性能参数 ---
        self.radius = radius
        self.mass = mass
        self.initial_max_angular_vel = max_angular_vel  # 初始设计值
        self.efficiency = efficiency
        self.friction_coeff = friction_coeff
        self.rated_power = rated_power
        self.max_power = max_power
        self.response_time = response_time

        # 转动惯量计算 (实心圆柱体)
        self.moment_of_inertia = 0.5 * mass * radius ** 2

        # --- 状态变量 ---
        # SOH会影响最大角速度和最大能量
        self.max_angular_vel = self.initial_max_angular_vel * math.sqrt(self.state_of_health)
        self.current_angular_vel = self.max_angular_vel * math.sqrt(initial_soc)  # 根据初始SOC设置初始角速度
        self.state = 'idle'  # 当前运行状态: 'idle', 'charging', 'discharging'

        # --- 历史记录 ---
        self.time_history = []
        self.energy_history = []
        self.power_history = []
        self.soc_history = []
        self.angular_vel_history = []

    # --- 核心计算方法 (部分有SOH修正) ---
    def calculate_kinetic_energy(self, angular_vel=None):
        if angular_vel is None:
            angular_vel = self.current_angular_vel
        return 0.5 * self.moment_of_inertia * angular_vel ** 2

    def calculate_max_energy(self):
        """计算当前SOH下的最大存储能量 (J)"""
        # 最大能量受SOH影响
        initial_max_energy = 0.5 * self.moment_of_inertia * self.initial_max_angular_vel ** 2
        return initial_max_energy * self.state_of_health

    def get_state_of_charge(self):  #SOC状态
        max_energy = self.calculate_max_energy()
        return self.calculate_kinetic_energy() / max_energy if max_energy > 0 else 0

    # --- HESS接口核心方法 (新增) ---
    def get_available_charge_power(self):
        """
        EMS查询：获取当前可用的充电功率 (W)
        受限于额定功率和SOC上限
        """
        soc = self.get_state_of_charge()
        if soc >= self.soc_upper_limit:
            return 0  # 达到上限，不能再充电

        # 简单线性衰减：越接近上限，可用功率越小
        power_derating_factor = (self.soc_upper_limit - soc) / (self.soc_upper_limit - self.soc_lower_limit)
        available_power = self.rated_power * power_derating_factor

        return min(self.rated_power, max(0, available_power))

    def get_available_discharge_power(self):
        """
        EMS查询：获取当前可用的放电功率 (W)
        受限于额定功率和SOC下限
        """
        soc = self.get_state_of_charge()
        if soc <= self.soc_lower_limit:
            return 0  # 达到下限，不能再放电

        # 简单线性衰减：越接近下限，可用功率越小
        power_derating_factor = (soc - self.soc_lower_limit) / (self.soc_upper_limit - self.soc_lower_limit)
        available_power = self.rated_power * power_derating_factor

        return min(self.rated_power, max(0, available_power))

    # --- 充放电与损耗控制方法 (状态更新) ---
    def charge(self, power, time):
        # 功率限制检查
        available_power = self.get_available_charge_power()
        if power > available_power:
            # print(f"警告: 充电功率 {power/1000:.1f}kW 超过当前可用值 {available_power/1000:.1f}kW，已限制。")
            power = available_power
        if power <= 0: return 0

        self.state = 'charging'

        input_energy = power * time
        effective_energy = input_energy * math.sqrt(self.efficiency)  # 充电效率
        current_energy = self.calculate_kinetic_energy()
        new_energy = current_energy + effective_energy
        max_energy = self.calculate_max_energy()

        if new_energy > max_energy:
            new_energy = max_energy

        self.current_angular_vel = math.sqrt(2 * new_energy / self.moment_of_inertia)
        self._record_history(time, power)
        return effective_energy

    def discharge(self, power, time):
        available_power = self.get_available_discharge_power()
        if power > available_power:
            # print(f"警告: 放电功率 {power/1000:.1f}kW 超过当前可用值 {available_power/1000:.1f}kW，已限制。")
            power = available_power
        if power <= 0: return 0

        self.state = 'discharging'

        required_energy = power * time
        energy_to_consume = required_energy / math.sqrt(self.efficiency)  # 放电效率
        current_energy = self.calculate_kinetic_energy()

        if energy_to_consume > current_energy:
            energy_to_consume = current_energy

        new_energy = current_energy - energy_to_consume
        self.current_angular_vel = math.sqrt(2 * new_energy / self.moment_of_inertia) if new_energy > 0 else 0
        self._record_history(time, -power)

        return energy_to_consume * math.sqrt(self.efficiency)  # 实际输出能量

    def idle_loss(self, time):
        self.state = 'idle'
        # 能量损耗与角速度的平方成正比，这里用简化的指数衰减模型
        loss_factor = math.exp(-self.friction_coeff * time)
        self.current_angular_vel *= loss_factor
        self._record_history(time, 0)

    # --- 内部辅助方法 ---
    def _record_history(self, time_delta, power):
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.energy_history.append(self.calculate_kinetic_energy())
        self.power_history.append(power)
        self.soc_history.append(self.get_state_of_charge())
        self.angular_vel_history.append(self.current_angular_vel)

    def plot_performance(self):
        # 绘图函数保持不变，用于可视化检查
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        # ... (绘图代码与您原版相同，此处省略以保持简洁)
        # ... 建议您将原绘图代码粘贴于此 ...
        axes[0].plot(self.time_history, [e / 1000 for e in self.energy_history], 'b-', lw=2, label='能量')
        axes[0].set_title('能量变化');
        axes[0].set_ylabel('能量 (kJ)');
        axes[0].grid(True);
        axes[0].legend()
        axes[1].plot(self.time_history, [p / 1000 for p in self.power_history], 'g-', lw=2, label='实时功率')
        axes[1].set_title('功率变化');
        axes[1].set_ylabel('功率 (kW)');
        axes[1].grid(True);
        axes[1].legend()
        axes[2].plot(self.time_history, [s * 100 for s in self.soc_history], 'm-', lw=2, label='SOC')
        axes[2].set_title('SOC变化');
        axes[2].set_ylabel('SOC (%)');
        axes[2].grid(True);
        axes[2].legend()
        axes[3].plot(self.time_history, self.angular_vel_history, 'c-', lw=2, label='角速度')
        axes[3].set_title('角速度变化');
        axes[3].set_ylabel('角速度 (rad/s)');
        axes[3].set_xlabel('时间 (s)');
        axes[3].grid(True);
        axes[3].legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        plt.show()


# --- HESS中的EMS调用示例 ---
def simulate_hess_with_flywheel():
    """
    一个简化的示例，演示EMS如何与更新后的飞轮模型交互。
    """
    # 1. 初始化飞轮模型，设置初始状态为50%
    flywheel = FlywheelModel(initial_soc=0.5, rated_power=50000, max_power=75000)

    # 2. 模拟一个电力需求信号 (正值为放电需求，负值为充电需求)
    # 这个信号在实际课题中会来自电网或负荷预测
    time_steps = np.arange(0, 60, 1)  # 模拟60秒，每秒一个决策点
    power_demand = -20000 * np.sin(time_steps * 0.5) + 30000 * np.cos(time_steps * 0.2)
    power_demand[20:25] = 70000  # 模拟一个突然的高功率放电需求

    print(f"--- 开始模拟，飞轮初始SOC: {flywheel.get_state_of_charge() * 100:.1f}% ---")

    # 记录初始状态
    flywheel._record_history(0, 0)

    # 3. EMS决策循环
    for i in range(len(time_steps) - 1):
        dt = time_steps[i + 1] - time_steps[i]
        demand = power_demand[i]

        # --- EMS核心决策逻辑 ---
        if demand > 0:  # 需要放电
            # 查询飞轮此刻能提供多少功率
            available_power = flywheel.get_available_discharge_power()
            power_to_dispatch = min(demand, available_power)
            flywheel.discharge(power_to_dispatch, dt)
            print(
                f"t={time_steps[i + 1]:>2}s: 需求 {demand / 1000:>5.1f}kW, 飞轮放电 {power_to_dispatch / 1000:>5.1f}kW, SOC: {flywheel.get_state_of_charge() * 100:.1f}%")

        elif demand < 0:  # 需要充电
            # 查询飞轮此刻能吸收多少功率
            available_power = flywheel.get_available_charge_power()
            power_to_dispatch = min(abs(demand), available_power)
            flywheel.charge(power_to_dispatch, dt)
            print(
                f"t={time_steps[i + 1]:>2}s: 需求 {demand / 1000:>5.1f}kW, 飞轮充电 {power_to_dispatch / 1000:>5.1f}kW, SOC: {flywheel.get_state_of_charge() * 100:.1f}%")

        else:  # 无需求，计算自放电损耗
            flywheel.idle_loss(dt)
            print(f"t={time_steps[i + 1]:>2}s: 需求 0.0kW, 飞轮闲置, SOC: {flywheel.get_state_of_charge() * 100:.1f}%")

    flywheel.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_flywheel()