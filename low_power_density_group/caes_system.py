# file: low_power_density_group/caes_system.py (统一接口修改版 V1.0)

import math
import numpy as np

# 解决在子文件夹中导入父文件夹模块的问题
import sys
import os

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 修改区域 1: 导入正确的基类 ---
from base_storage_model import BaseStorageModel


# --- 修改区域 2: 让 CAES 继承 BaseStorageModel ---
class DiabaticCAES(BaseStorageModel):
    """
    补燃式压缩空气储能 (D-CAES) 模型 (HESS集成版 - 严格对应论文公式)
    已按照BaseStorageModel进行接口标准化。
    """

    def __init__(self,
                 id,
                 dt_s,
                 initial_soc=0.5,
                 # --- 核心修改：我们只定义顶层参数，与基准表保持一致 ---
                 compressor_rated_power_mw=80.0,  # 充电（压缩）功率
                 turbine_rated_power_mw=80.0,  # 放电（发电）功率
                 rated_capacity_mwh=640.0,  # 额定等效输出电容量

                 # 效率与成本参数
                 charge_rate_kg_per_kwh=7.2,  # 典型值：每kWh电能可以压缩多少kg空气
                 air_usage_rate_kg_per_kwh=4.0,  # 典型值：每发一度电需要消耗多少kg空气
                 heat_rate_kj_per_kwh=4500,  # 典型值：补燃式热耗率 (kJ_fuel / kWh_elec)
                 om_cost_per_mwh=40,  # 电力部分运维成本
                 cost_per_kwh_fuel=0.3,  # 燃料成本 (元/kWh_fuel)

                 # --- 其他关键参数 ---
                 soc_upper_limit=0.98,
                 soc_lower_limit=0.2
                 ):

        # 1. 标准接口初始化
        super().__init__(id, dt_s)

        # 2. 将基准参数赋值给父类的标准属性
        self.soc = initial_soc
        self.power_m_w = turbine_rated_power_mw
        self.capacity_mwh = rated_capacity_mwh
        self.soc_min = soc_lower_limit
        self.soc_max = soc_upper_limit
        self.om_cost_per_mwh = om_cost_per_mwh
        self.soh = 1.0  # CAES SOH基本不变

        # 核心计算：电-电往返效率 (不考虑燃料输入，仅看空气媒介)
        # 1 kWh电能 -> x kg 空气 -> y kWh电能. 效率 = y/1 = x_kg / (y_kwh * air_usage_rate) = charge_rate / air_usage_rate
        if air_usage_rate_kg_per_kwh > 1e-6:
            self.efficiency = charge_rate_kg_per_kwh / air_usage_rate_kg_per_kwh
        else:
            self.efficiency = 0

        # 3. 保留CAES特有的物理参数
        self.P_comp_rated_w = compressor_rated_power_mw * 1e6
        self.eta_charge_rate = charge_rate_kg_per_kwh
        self.P_gen_rated_w = self.power_m_w * 1e6
        self.eta_heat_rate = heat_rate_kj_per_kwh
        self.eta_air_usage = air_usage_rate_kg_per_kwh
        self.cost_per_kwh_fuel = cost_per_kwh_fuel

        # 4. 根据顶层参数，反向推算内部物理参数
        # 核心推算：根据额定输出电容量，反算储气室的最大空气质量
        # Max Air Mass (kg) = Rated Capacity (kWh) * Air Usage Rate (kg/kWh)
        # rated_capacity_mwh * 1000 -> kWh
        self.M_air_max = self.capacity_mwh * 1000 * self.eta_air_usage

        # 5. 初始化核心状态变量：储气室空气质量 M_air (kg)
        self.M_air_kg = self.M_air_max * self.soc

        self.mass_history = []
        self.fuel_consumption_history_j = []
        self.state = 'idle'

    # ==============================================================================
    # --- 新增：核心标准接口 update_state ---
    # ==============================================================================
    def update_state(self, dispatch_power_w):
        """
        根据调度指令（单位：W）更新储能状态。
        """
        if dispatch_power_w > 0:
            # 正功率表示放电 (发电)
            self.discharge(dispatch_power_w, self.dt_s)
        elif dispatch_power_w < 0:
            # 负功率表示充电 (压缩)
            self.charge(abs(dispatch_power_w), self.dt_s)
        else:
            # 零功率表示闲置
            self.idle_loss(self.dt_s)

    # ==============================================================================
    # --- 模型核心物理方法 (完全保留您原有的代码) ---
    # ==============================================================================

    def get_soc(self):
        """根据空气质量计算并更新SOC"""
        if self.M_air_max > 1e-6:
            self.soc = self.M_air_kg / self.M_air_max
        else:
            self.soc = self.soc_min
        return self.soc

    def get_available_charge_power(self):
        """获取当前可用的充电(压缩)功率 (W)"""
        if self.get_soc() >= self.soc_max: return 0
        return self.P_comp_rated_w

    def get_available_discharge_power(self):
        """获取当前可用的放电(发电)功率 (W)"""
        if self.get_soc() <= self.soc_min: return 0

        # 受限于额定功率和剩余空气量
        # 可持续发电时长 (h) = 剩余可用空气质量 / (额定功率kW * 耗气率)
        available_air_kg = self.M_air_kg - self.M_air_max * self.soc_min
        if self.P_gen_rated_w > 0 and self.eta_air_usage > 0:
            duration_h = available_air_kg / ((self.P_gen_rated_w / 1000) * self.eta_air_usage)
            # 如果可持续时长小于一个时间步，则按比例降低可用功率
            if duration_h < (self.dt_s / 3600):
                return duration_h * (3600 / self.dt_s) * self.P_gen_rated_w

        return self.P_gen_rated_w

    def charge(self, power_elec, time_s):
        """按指定电功率充电 (压缩)"""
        power_elec = min(power_elec, self.get_available_charge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'charging'
        energy_consumed_kwh = (power_elec * time_s) / 3.6e6
        mass_stored_kg = energy_consumed_kwh * self.eta_charge_rate
        self.M_air_kg += mass_stored_kg
        self.M_air_kg = min(self.M_air_kg, self.M_air_max * self.soc_max)
        self.fuel_consumption_history_j.append(0)

    def discharge(self, power_elec, time_s):
        """按指定电功率放电 (发电)，并计算燃料消耗"""
        power_elec = min(power_elec, self.get_available_discharge_power())
        if power_elec <= 0:
            self.idle_loss(time_s)
            return

        self.state = 'discharging'
        energy_generated_kwh = (power_elec * time_s) / 3.6e6
        mass_consumed_kg = energy_generated_kwh * self.eta_air_usage

        # 检查空气余量
        available_air_kg = self.M_air_kg - self.M_air_max * self.soc_min
        if mass_consumed_kg > available_air_kg:
            mass_consumed_kg = available_air_kg
            # 根据实际可用空气反算实际发电量
            if self.eta_air_usage > 0:
                energy_generated_kwh = mass_consumed_kg / self.eta_air_usage
                power_elec = (energy_generated_kwh * 3.6e6) / time_s if time_s > 0 else 0
            else:
                power_elec = 0

        self.M_air_kg -= mass_consumed_kg
        self.M_air_kg = max(self.M_air_kg, self.M_air_max * self.soc_min)

        fuel_consumed_kj = energy_generated_kwh * self.eta_heat_rate
        self.fuel_consumption_history_j.append(fuel_consumed_kj * 1000)

    def idle_loss(self, time_s):
        """模拟闲置时的洞穴气体泄漏 (简化为无损)"""
        self.state = 'idle'
        self.fuel_consumption_history_j.append(0)


# --- 单元测试代码 (保持不变) ---
if __name__ == "__main__":
    caes = DiabaticCAES(id='caes_test', dt_s=3600, initial_soc=0.5)

    print(f"CAES Initialized. Comp Power: {caes.P_comp_rated_w / 1e6} MW, Gen Power: {caes.power_m_w} MW")
    print(f"Max Air Mass: {caes.M_air_max / 1000} tons, Equivalent Elec Capacity: {caes.capacity_mwh:.2f} MWh")
    print(f"Initial SOC: {caes.get_soc():.3f}\n")

    # 模拟以200MW功率连续压缩空气10小时
    charge_power = 200e6
    print(f"--- Charging with {charge_power / 1e6} MW for 1h ---")
    caes.update_state(-charge_power)  # 充电
    print(f"After charging, SOC: {caes.get_soc():.3f}, Air Mass: {caes.M_air_kg / 1000:.2f} tons\n")

    # 模拟以300MW电功率连续发电5小时
    discharge_power = 300e6
    print(f"--- Discharging with {discharge_power / 1e6} MW_e for 1h ---")
    caes.update_state(discharge_power)  # 放电
    fuel_j = caes.fuel_consumption_history_j[-1]
    fuel_mwh_th = fuel_j / 3.6e9
    print(f"After discharging, SOC: {caes.get_soc():.3f}, Air Mass: {caes.M_air_kg / 1000:.2f} tons")
    print(f"  > Consumed Fuel (thermal): {fuel_mwh_th:.2f} MWh_th\n")