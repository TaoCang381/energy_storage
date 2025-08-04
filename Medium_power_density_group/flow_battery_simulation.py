import math
import matplotlib.pyplot as plt
import numpy as np
from base_storage_model import EnergyStorageUnit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 物理常数 ---
R_GAS_CONST = 8.314  # 理想气体常数 (J/(mol·K))
F_FARADAY_CONST = 96485  # 法拉第常数 (C/mol)


class FlowBattery(EnergyStorageUnit):
    """
    全钒液流电池模型 (HESS集成版)
    特点：
    1. 功率和能量解耦。
    2. OCV-SOC关系由能斯特方程精确描述。
    3. 包含泵功等关键辅助系统损耗。
    """

    def __init__(self,
                 # --- 能量相关参数 (由储罐决定) ---
                 electrolyte_volume_liters=5000,  # 电解液体积 (L)
                 vanadium_concentration_mol_l=1.6,  # 钒离子浓度 (mol/L)

                 # --- 功率相关参数 (由电堆决定) ---
                 stack_rated_power=25000,  # 电堆额定功率 (W)
                 stack_internal_resistance=0.05,  # 电堆等效内阻 (Ohm)

                 # --- 辅助系统参数 ---
                 pump_base_power=200,  # 泵的固定基础功耗 (W)
                 pump_flow_coefficient=0.01,  # 泵的流量功耗系数 (W/W)

                 # --- 能斯特方程参数 ---
                 standard_potential_E0=1.26,  # 标准电极电势 (V)
                 temperature_k=298.15,  # 工作温度 (K)

                 # --- HESS集成 & 其他参数 ---
                 ess_id="flow_battery_01",
                 initial_soh=1.0,
                 initial_soc=0.5,
                 soc_upper_limit=0.9,
                 soc_lower_limit=0.1
                 ):

        # --- HESS 集成参数 ---
        self.id = ess_id
        self.state_of_health = initial_soh
        self.state_of_charge = initial_soc

        # --- 规格参数 ---
        self.electrolyte_volume_liters = electrolyte_volume_liters
        self.vanadium_concentration_mol_l = vanadium_concentration_mol_l
        self.stack_rated_power = stack_rated_power
        self.initial_internal_resistance = stack_internal_resistance
        self.pump_base_power = pump_base_power
        self.pump_flow_coefficient = pump_flow_coefficient
        self.standard_potential_E0 = standard_potential_E0
        self.temperature_k = temperature_k
        self.soc_upper_limit = soc_upper_limit
        self.soc_lower_limit = soc_lower_limit

        # --- 实时工作参数 (受SOH影响) ---
        self.internal_resistance = self.initial_internal_resistance / self.state_of_health

        # --- 状态与历史记录 ---
        self.state = 'idle'
        self.time_history = []
        self.power_history = []
        self.soc_history = []
        self.voltage_history = []
        self.pump_power_history = []

    def _calculate_total_capacity_ah(self):
        """计算理论总安时容量 Q = V*C*F / 3600 """
        total_moles = self.electrolyte_volume_liters * self.vanadium_concentration_mol_l
        total_charge_coulombs = total_moles * F_FARADAY_CONST
        return (total_charge_coulombs / 3600.0) * self.state_of_health

    def _get_ocv(self, soc):
        """内部方法：根据能斯特方程计算开路电压 (OCV)"""
        soc = max(min(soc, 0.999), 0.001)
        nernst_term = (2 * R_GAS_CONST * self.temperature_k / F_FARADAY_CONST) * math.log(soc / (1 - soc))
        return self.standard_potential_E0 + nernst_term

    def get_soc(self):
        return self.state_of_charge

    def _get_pump_power(self, stack_power):
        """计算泵的功耗"""
        return self.pump_base_power + self.pump_flow_coefficient * abs(stack_power)

    def get_available_charge_power(self):
        """EMS查询：获取当前可用的净充电功率 (W)"""
        if self.state_of_charge >= self.soc_upper_limit: return 0
        k = self.pump_flow_coefficient
        available_power = self.stack_rated_power * (1 - k) - self.pump_base_power
        return max(0, available_power)

    def get_available_discharge_power(self):
        """EMS查询：获取当前可用的净放电功率 (W)"""
        if self.state_of_charge <= self.soc_lower_limit: return 0
        k = self.pump_flow_coefficient
        available_power = self.stack_rated_power * (1 - k) - self.pump_base_power
        return max(0, available_power)

    def charge(self, power, time_s):
        """按指定净功率和时间充电"""
        net_power = min(power, self.get_available_charge_power())
        if net_power <= 0: return
        self.state = 'charging'
        k = self.pump_flow_coefficient
        stack_power = (net_power + self.pump_base_power) / (1 - k)
        pump_power = self._get_pump_power(stack_power)
        ocv = self._get_ocv(self.state_of_charge)
        a = self.internal_resistance;
        b = ocv;
        c = -stack_power
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0: return
        current = (-b + math.sqrt(discriminant)) / (2 * a)
        delta_ah = (current * time_s) / 3600.0
        delta_soc = delta_ah / self._calculate_total_capacity_ah()
        self.state_of_charge += delta_soc
        self.state_of_charge = min(self.state_of_charge, self.soc_upper_limit)
        self._record_history(time_s, net_power, ocv + current * a, pump_power)

    def discharge(self, power, time_s):
        """按指定净功率和时间放电"""
        net_power = min(power, self.get_available_discharge_power())
        if net_power <= 0: return
        self.state = 'discharging'
        k = self.pump_flow_coefficient
        stack_power = (net_power + self.pump_base_power) / (1 - k)
        pump_power = self._get_pump_power(stack_power)
        ocv = self._get_ocv(self.state_of_charge)
        a = self.internal_resistance;
        b = -ocv;
        c = stack_power
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0: return
        current = (-b - math.sqrt(discriminant)) / (2 * a)
        delta_ah = (current * time_s) / 3600.0
        delta_soc = delta_ah / self._calculate_total_capacity_ah()
        self.state_of_charge -= delta_soc
        self.state_of_charge = max(self.state_of_charge, self.soc_lower_limit)
        self._record_history(time_s, -net_power, ocv - current * a, pump_power)

    def idle_loss(self, time_s):
        """闲置时，仍有泵的基础功耗和自放电"""
        self.state = 'idle'
        daily_loss = 0.002
        self.state_of_charge -= (daily_loss / (24 * 3600)) * time_s
        self._record_history(time_s, -self.pump_base_power, self._get_ocv(self.state_of_charge), self.pump_base_power)

    def _record_history(self, time_delta, power, voltage, pump_power):
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(self.state_of_charge)
        self.voltage_history.append(voltage)
        self.pump_power_history.append(pump_power)

    def plot_performance(self):
        """绘制所有性能曲线"""
        if not self.time_history:
            print("没有历史数据可供绘图。")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'液流电池 ({self.id}) 性能曲线', fontsize=16)

        # 将时间从秒转换为小时，以匹配长时间尺度模拟
        time_h = [t / 3600.0 for t in self.time_history]

        # 功率转换为kW
        net_power_kw = [p / 1000 for p in self.power_history]
        pump_power_kw = [p / 1000 for p in self.pump_power_history]

        # --- 1. SOC 曲线 ---
        axes[0].plot(time_h, self.soc_history, 'm-', lw=2, label='SOC')
        axes[0].set_title('荷电状态 (SOC) 变化')
        axes[0].set_ylabel('SOC')
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_ylim(-0.05, 1.05)

        # --- 2. 功率曲线 (包含净功率和泵功) ---
        axes[1].plot(time_h, net_power_kw, 'g-', lw=2, label='净输出功率 (电网侧)')
        axes[1].plot(time_h, pump_power_kw, 'r:', lw=2, label='泵消耗功率')
        axes[1].set_title('功率变化')
        axes[1].set_ylabel('功率 (kW)')
        axes[1].grid(True)
        axes[1].legend()

        # --- 3. 终端电压曲线 ---
        axes[2].plot(time_h, self.voltage_history, 'c-', lw=2, label='电堆终端电压')
        axes[2].set_title('电堆终端电压变化')
        axes[2].set_ylabel('电压 (V)')
        axes[2].grid(True)
        axes[2].legend()
        axes[2].set_xlabel('时间 (小时)')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# --- HESS中的EMS调用示例 ---
def simulate_hess_with_flow_battery():
    """一个简化的示例，演示液流电池用于长时能量时移（如可再生能源日内消纳）"""
    # 25kW/100kWh的液流电池系统，能量容量为4小时
    # 根据额定容量计算等效的电解液体积
    capacity_kwh = 100
    # 假设电堆由80个单体电池串联
    nominal_voltage = 1.26 * 80
    total_ah = capacity_kwh * 1000 / nominal_voltage
    volume_l = total_ah * 3600 / (1.6 * F_FARADAY_CONST)  # 约1830 L

    flow_battery = FlowBattery(
        electrolyte_volume_liters=volume_l,
        stack_rated_power=25000,
        initial_soc=0.5
    )

    # 模拟一个典型的光伏日内出力曲线
    time_steps_h = np.arange(0, 24, 0.25)  # 15分钟一个决策点
    # 净负荷 = 基础负荷 - 光伏出力。负值表示光伏过剩，需要充电
    base_load = 15000
    pv_generation = np.maximum(0, 40000 * np.sin((time_steps_h - 6) * np.pi / 12))
    net_load = base_load - pv_generation

    print(f"--- 开始模拟，液流电池初始SOC: {flow_battery.state_of_charge:.2f} ---")
    flow_battery._record_history(0, 0, flow_battery._get_ocv(flow_battery.state_of_charge),
                                 flow_battery.pump_base_power)

    # EMS决策循环
    for i in range(len(time_steps_h) - 1):
        dt_s = (time_steps_h[i + 1] - time_steps_h[i]) * 3600
        # 需求 = -净负荷。净负荷为负（光伏过剩）时，需求为正（充电）
        demand = -net_load[i]

        if demand > 0:  # 充电
            power = min(demand, flow_battery.get_available_charge_power())
            flow_battery.charge(power, dt_s)
        elif demand < 0:  # 放电
            power = min(abs(demand), flow_battery.get_available_discharge_power())
            flow_battery.discharge(power, dt_s)
        else:
            flow_battery.idle_loss(dt_s)

    print("--- 模拟结束 ---")
    flow_battery.plot_performance()


if __name__ == "__main__":
    simulate_hess_with_flow_battery()