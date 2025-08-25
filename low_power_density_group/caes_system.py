# file: PythonProject/low power density group/caes_system.py

import math
import matplotlib.pyplot as plt
import numpy as np

# 解决导入错误的路径问题
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_storage_model import EnergyStorageUnit


class DiabaticCAES(EnergyStorageUnit):
    """
    补燃式压缩空气储能 (D-CAES) 模型 (HESS集成版 - 严格对应论文公式)
    包含压缩机、储气室、燃气轮机三个部分，并考虑燃料消耗。
    """

    def __init__(self,
                 ess_id="diabatic_caes_01",
                 initial_soc=0.5,
                 initial_soh=1.0,  # CAES的SOH几乎不衰减
                 # ========================== 合理化参数配置 (开始) ==========================
                 # --- 储气室 (能量容量) ---
                 cavern_max_air_mass_kg=5e8,  # 储气室最大空气容量 (kg), e.g., 50万吨

                 # --- 压缩机组 (充电) ---
                 compressor_rated_power_w=200e6,  # 压缩机组额定功率 (W), e.g., 200 MW
                 charge_rate_kg_per_kwh=0.2,  # 充电效率 (kg/kWh): 每消耗1度电存入的空气质量

                 # --- 透平发电机组 (放电) ---
                 turbine_rated_power_w=300e6,  # 透平发电机组额定功率 (W), e.g., 300 MW
                 heat_rate_kj_per_kwh=5000,  # 热耗率 (kJ/kWh): 每发1度电，需要消耗5000kJ的燃料热值
                 air_usage_rate_kg_per_kwh=1.0,  # 耗气率 (kg/kWh): 每发1度电，需要消耗1.0kg的压缩空气

                 # --- 运行限制 ---
                 min_power_ratio=0.25,
                 soc_upper_limit=0.98,
                 soc_lower_limit=0.2,
                 # ========================== 合理化参数配置 (结束) ==========================
                 cost_per_kwh_fuel=0.3  # 每kWh燃料热值的成本（元）
                 ):

        super().__init__(ess_id, initial_soc, initial_soh)
        self.soh = 1.0

        # --- 规格参数 ---
        self.M_air_max = cavern_max_air_mass_kg
        self.P_comp_rated = compressor_rated_power_w
        self.eta_charge_rate = charge_rate_kg_per_kwh
        self.P_gen_rated = turbine_rated_power_w
        self.rated_power_w = self.P_gen_rated
        self.eta_heat_rate = heat_rate_kj_per_kwh
        self.eta_air_usage = air_usage_rate_kg_per_kwh
        self.soc_max = soc_upper_limit
        self.soc_min = soc_lower_limit
        self.cost_per_kwh_fuel = cost_per_kwh_fuel

        self.P_gen_min = self.P_gen_rated * min_power_ratio
        self.P_comp_min = self.P_comp_rated * min_power_ratio

        # --- 核心状态变量：储气室空气质量 M_air ---
        self.M_air_kg = self.M_air_max * self.soc

        self.mass_history = []
        self.fuel_consumption_history_j = []  # 记录消耗的燃料热量

    def get_soc(self):
        """根据空气质量计算并更新SOC"""
        if self.M_air_max > 0:
            self.soc = self.M_air_kg / self.M_air_max
        else:
            self.soc = 0
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电(压缩)功率 (W)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_comp_rated

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_min: return 0

        # 同时受限于额定功率和剩余空气量
        # 可持续发电时长 (h) = 剩余空气质量 / (额定功率 * 耗气率)
        duration_h = self.M_air_kg / ((self.P_gen_rated / 1000) * self.eta_air_usage)
        # 如果持续时长小于一个时间步，则按比例降低可用功率
        if duration_h < 1.0:  # 假设调度时间步长为1小时
            max_power_by_air = self.P_gen_rated * duration_h
            return min(self.P_gen_rated, max_power_by_air)

        return self.P_gen_rated

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (压缩)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec < self.P_comp_min:
            self.idle_loss(time_s)
            return

        self.state = 'charging'

        # 1. 计算存入的空气质量 (动态方程)
        energy_consumed_kwh = (power_elec * time_s) / 3.6e6
        mass_stored_kg = energy_consumed_kwh * self.eta_charge_rate
        self.M_air_kg += mass_stored_kg

        # 2. 应用储气室容量约束
        self.M_air_kg = min(self.M_air_kg, self.M_air_max * self.soc_max)

        self._record_history_caes(time_s, power_elec, 0)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，并计算燃料消耗"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec < self.P_gen_min:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'

        energy_generated_kwh = (power_elec * time_s) / 3.6e6

        # 1. 计算消耗的空气质量 (动态方程)
        mass_consumed_kg = energy_generated_kwh * self.eta_air_usage
        self.M_air_kg -= mass_consumed_kg

        # 2. 应用储气室容量约束
        self.M_air_kg = max(self.M_air_kg, self.M_air_max * self.soc_min)

        # 3. 计算消耗的燃料热量
        fuel_consumed_kj = energy_generated_kwh * self.eta_heat_rate
        fuel_consumed_j = fuel_consumed_kj * 1000

        self._record_history_caes(time_s, -power_elec, fuel_consumed_j)
        return power_elec, fuel_consumed_j

    def idle_loss(self, time_s):
        """模拟闲置时的洞穴气体泄漏"""
        self.state = 'idle'
        # 简化模型，忽略微小的泄漏
        self._record_history_caes(time_s, 0, 0)

    def _record_history_caes(self, time_delta, power, fuel_consumed_j):
        """记录CAES特有的历史数据"""
        current_soc = self.get_soc()
        super()._record_history(time_delta, power, current_soc)
        self.mass_history.append(self.M_air_kg)
        self.fuel_consumption_history_j.append(fuel_consumed_j)


# --- 单元测试用的示例函数 ---
def simulate_caes_test():
    caes = DiabaticCAES(initial_soc=0.5)

    print(f"CAES Initialized. Comp Power: {caes.P_comp_rated / 1e6} MW, Gen Power: {caes.P_gen_rated / 1e6} MW")
    print(f"Max Air Mass: {caes.M_air_max / 1000} tons")
    print(f"Initial SOC: {caes.get_soc():.3f}\n")

    # 模拟以200MW功率连续压缩空气10小时
    charge_power = 200e6
    charge_time = 10 * 3600
    print(f"--- Charging with {charge_power / 1e6} MW for {charge_time / 3600:.1f}h ---")
    caes.charge(charge_power, charge_time)
    print(f"After charging, SOC: {caes.get_soc():.3f}, Air Mass: {caes.M_air_kg / 1000:.2f} tons\n")

    # 模拟以300MW电功率连续发电5小时
    discharge_power = 300e6
    discharge_time = 5 * 3600
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for {discharge_time / 3600:.1f}h ---")
    actual_p_elec, fuel_j = caes.discharge(discharge_power, discharge_time)
    fuel_gwh = fuel_j / 3.6e12
    print(f"After discharging, SOC: {caes.get_soc():.3f}, Air Mass: {caes.M_air_kg / 1000:.2f} tons")
    print(
        f"  > Generated Elec: {actual_p_elec / 1e6 * (discharge_time / 3600):.2f} MWh, Consumed Fuel (thermal): {fuel_gwh * 1000:.2f} MWh_th\n")


if __name__ == "__main__":
    simulate_caes_test()