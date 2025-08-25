# file: PythonProject/ems.py
import numpy as np


class HierarchicalEMS:
    """
    分层协同能源管理系统 (EMS)。
    负责将波动信号分解，并分配给不同特性的储能组。
    """

    def __init__(self, hess_system):
        self.hess = hess_system

    def decompose_signal(self, total_fluctuation_series, current_index, short_window_size=5):
        """
        使用滑动平均滤波对波动信号进行分解。
        """
        # 建立一个短期的历史窗口
        start_index = max(0, current_index - short_window_size)
        window = total_fluctuation_series[start_index: current_index + 1]

        # 计算窗口内的移动平均值，作为中频分量
        p_medium_freq = np.mean(window)

        # 当前总波动与中频分量的差值，作为高频分量
        p_high_freq = total_fluctuation_series[current_index] - p_medium_freq

        return p_high_freq, p_medium_freq

    def distribute_power_to_group(self, group_name, power_demand, dt_s):
        """
        按可用功率和SOC健康度，将功率需求分配给一个组内的所有单元。
        """
        group_units = []
        if group_name == 'fast':
            group_units = list(self.hess.fast_response_units.values())
        elif group_name == 'medium':
            group_units = list(self.hess.medium_response_units.values())

        if not group_units: return 0

        is_charging = power_demand < 0
        total_weight = 0
        unit_weights = {}

        for unit in group_units:
            soc = unit.get_soc()
            soc_health_factor = 1 - abs(soc - 0.5) / 0.5
            avail_power = unit.get_available_charge_power() if is_charging else unit.get_available_discharge_power()
            weight = avail_power * soc_health_factor
            unit_weights[unit.id] = weight
            total_weight += weight

        if total_weight < 1e-3: return 0

        actual_dispatch_total = 0
        for unit in group_units:
            ratio = unit_weights[unit.id] / total_weight if total_weight > 0 else 0
            power_to_dispatch = abs(power_demand) * ratio

            if is_charging:
                unit.charge(power_to_dispatch, dt_s)
                actual_dispatch_total -= power_to_dispatch
            else:
                unit.discharge(power_to_dispatch, dt_s)
                actual_dispatch_total += power_to_dispatch

        return actual_dispatch_total