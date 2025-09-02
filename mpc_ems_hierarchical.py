# file: mpc_ems_hierarchical.py (V2.0 - WPT-Powered Functional Decoupling)
# 备注：重构分层MPC逻辑，上层负责低频经济调度，下层负责中高频协同控制。

import cvxpy as cp
import numpy as np


class HierarchicalMPCEms:
    def __init__(self, hess_system, upper_horizon, lower_horizon):
        self.hess = hess_system
        self.PH_upper = upper_horizon
        self.PH_lower = lower_horizon

        # --- 修改区域 START: 根据功能重新定义储能分组 ---
        # 功能层1：长周期能量型储能 (负责低频经济调度)
        self.energy_assets = [u for u in self.hess.all_units.values() if
                              any(keyword in u.id for keyword in ['phs', 'hes', 'tes', 'caes'])]

        # 功能层2：平滑型储能 (负责中频波动)
        self.smoothing_assets = [u for u in self.hess.all_units.values() if 'ees' in u.id]

        # 功能层3：瞬时功率型储能 (负责高频响应)
        self.power_assets = [u for u in self.hess.all_units.values() if
                             any(keyword in u.id for keyword in ['fw', 'sc', 'smes'])]
        # --- 修改区域 END ---

    def solve_with_fallback(self, problem, is_mip=False):
        # (此辅助函数不变)
        pass

    def solve_upper_level(self, current_soc, net_load_forecast_upper, grid_prices_upper, slow_task_signal_upper):
        """
        上层MPC: 负责能量型储能的低频经济调度。
        """
        # --- 修改区域 START: 完全重写上层MPC逻辑 ---

        # 1. 定义决策变量 (仅针对能量型储能)
        discharge_vars = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        charge_vars = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        soc_vars = {unit.id: cp.Variable(self.PH_upper + 1) for unit in self.energy_assets}
        grid_exchange = cp.Variable(self.PH_upper)  # 从电网购电为正，售电为负

        # 2. 定义目标函数
        # 目标 = 电网交互成本 + 运维成本 + 低频任务跟踪惩罚

        grid_cost = cp.sum(grid_exchange * grid_prices_upper) * (self.hess.dt_upper / 3600)

        om_cost = cp.sum([
            unit.om_cost_per_mwh * (discharge_vars[unit.id] + charge_vars[unit.id])
            for unit in self.energy_assets
        ]) * (self.hess.dt_upper / 3600)

        total_slow_dispatch = cp.sum([
            discharge_vars[unit.id] - charge_vars[unit.id] for unit in self.energy_assets
        ])

        # 核心改动：让能量型储能的联合出力去跟踪低频信号
        tracking_penalty_slow = 1e3 * cp.sum_squares(total_slow_dispatch - slow_task_signal_upper)

        objective = cp.Minimize(grid_cost + om_cost + tracking_penalty_slow)

        # 3. 定义约束条件
        constraints = []

        # 功率平衡约束
        # 净负荷 + 电网购电 = 能量型储能总出力
        constraints.append(net_load_forecast_upper + grid_exchange == total_slow_dispatch)

        for unit in self.energy_assets:
            uid = unit.id
            # SOC 演变约束
            for t in range(self.PH_upper):
                constraints.append(
                    soc_vars[uid][t + 1] == soc_vars[uid][t] -
                    ((discharge_vars[uid][t] / unit.efficiency) - (charge_vars[uid][t] * unit.efficiency)) * (
                                self.hess.dt_upper / 3600) / unit.capacity_mwh
                )

            # 初始SOC约束
            constraints.append(soc_vars[uid][0] == current_soc[uid])

            # SOC 上下限约束
            constraints.append(soc_vars[uid] >= unit.soc_min)
            constraints.append(soc_vars[uid] <= unit.soc_max)

            # 充放电功率约束
            constraints.append(discharge_vars[uid] <= unit.power_m_w * 1e6)
            constraints.append(charge_vars[uid] <= unit.power_m_w * 1e6)

        problem = cp.Problem(objective, constraints)

        if self.solve_with_fallback(problem, is_mip=False):
            return {
                "status": "optimal",
                "dispatch": {unit.id: (discharge_vars[unit.id].value - charge_vars[unit.id].value) for unit in
                             self.energy_assets}
            }
        else:
            return {"status": "failed", "dispatch": {}}

        # --- 修改区域 END ---

    def solve_lower_level(self, current_soc, reference_signals, mid_task_signal, high_task_signal):
        """
        下层MPC: 负责平滑型和功率型储能的中高频协同控制。
        """
        # --- 修改区域 START: 完全重写下层MPC逻辑 ---

        # 1. 定义决策变量 (分别为平滑型和功率型)
        # 平滑型 (EES)
        discharge_smooth = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.smoothing_assets}
        charge_smooth = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.smoothing_assets}
        soc_smooth = {u.id: cp.Variable(self.PH_lower + 1) for u in self.smoothing_assets}

        # 功率型 (FW, SC, SMES)
        discharge_power = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.power_assets}
        charge_power = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.power_assets}
        soc_power = {u.id: cp.Variable(self.PH_lower + 1) for u in self.power_assets}

        # 2. 定义目标函数
        # 目标 = 中频跟踪惩罚 + 高频跟踪惩罚 + 运维成本

        # 中频任务跟踪 (由EES执行)
        total_smooth_dispatch = cp.sum([discharge_smooth[u.id] - charge_smooth[u.id] for u in self.smoothing_assets])
        tracking_penalty_mid = 1e5 * cp.sum_squares(total_smooth_dispatch - mid_task_signal)

        # 高频任务跟踪 (由快速组执行)
        total_power_dispatch = cp.sum([discharge_power[u.id] - charge_power[u.id] for u in self.power_assets])
        tracking_penalty_high = 1e7 * cp.sum_squares(total_power_dispatch - high_task_signal)

        # 运维成本
        om_cost_smooth = cp.sum(
            [u.om_cost_per_mwh * (discharge_smooth[u.id] + charge_smooth[u.id]) for u in self.smoothing_assets]) * (
                                     self.hess.dt_lower / 3600)
        om_cost_power = cp.sum(
            [u.om_cost_per_mwh * (discharge_power[u.id] + charge_power[u.id]) for u in self.power_assets]) * (
                                    self.hess.dt_lower / 3600)

        objective = cp.Minimize(tracking_penalty_mid + tracking_penalty_high + om_cost_smooth + om_cost_power)

        # 3. 定义约束条件
        constraints = []

        # 为平滑型储能建立约束
        for unit in self.smoothing_assets:
            uid = unit.id
            for t in range(self.PH_lower):
                constraints.append(soc_smooth[uid][t + 1] == soc_smooth[uid][t] - (
                            (discharge_smooth[uid][t] / unit.efficiency) - (
                                charge_smooth[uid][t] * unit.efficiency)) * (
                                               self.hess.dt_lower / 3600) / unit.capacity_mwh)
            constraints.append(soc_smooth[uid][0] == current_soc[uid])
            constraints.append(soc_smooth[uid] >= unit.soc_min)
            constraints.append(soc_smooth[uid] <= unit.soc_max)
            constraints.append(discharge_smooth[uid] <= unit.power_m_w * 1e6)
            constraints.append(charge_smooth[uid] <= unit.power_m_w * 1e6)

        # 为功率型储能建立约束
        for unit in self.power_assets:
            uid = unit.id
            for t in range(self.PH_lower):
                constraints.append(soc_power[uid][t + 1] == soc_power[uid][t] - (
                            (discharge_power[uid][t] / unit.efficiency) - (charge_power[uid][t] * unit.efficiency)) * (
                                               self.hess.dt_lower / 3600) / unit.capacity_mwh)
            constraints.append(soc_power[uid][0] == current_soc[uid])
            constraints.append(soc_power[uid] >= unit.soc_min)
            constraints.append(soc_power[uid] <= unit.soc_max)
            constraints.append(discharge_power[uid] <= unit.power_m_w * 1e6)
            constraints.append(charge_power[uid] <= unit.power_m_w * 1e6)

        problem = cp.Problem(objective, constraints)

        final_dispatch = {}
        if self.solve_with_fallback(problem, is_mip=False):
            # 提取平滑型储能的调度结果
            for unit in self.smoothing_assets:
                final_dispatch[unit.id + "_power"] = discharge_smooth[unit.id].value[0] - charge_smooth[unit.id].value[
                    0]
            # 提取功率型储能的调度结果
            for unit in self.power_assets:
                final_dispatch[unit.id + "_power"] = discharge_power[unit.id].value[0] - charge_power[unit.id].value[0]
        else:  # 求解失败，所有下层设备不出力
            for unit in self.smoothing_assets: final_dispatch[unit.id + "_power"] = 0
            for unit in self.power_assets: final_dispatch[unit.id + "_power"] = 0

        # 合并上层慢速储能的当前时刻计划
        for uid, dispatch_plan_lower in reference_signals["slow_asset_dispatch"].items():
            final_dispatch[uid + "_power"] = dispatch_plan_lower[0]

        return final_dispatch
        # --- 修改区域 END ---