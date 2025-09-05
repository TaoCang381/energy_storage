# 文件: mpc_ems_hierarchical.py (V2.3 - 最终修正版)

import cvxpy as cp
import numpy as np


class HierarchicalMPCEms:
    def __init__(self, hess_system, upper_horizon, lower_horizon, dt_upper_s, dt_lower_s):
        self.hess = hess_system
        self.PH_upper = upper_horizon
        self.PH_lower = lower_horizon
        self.dt_upper_s = dt_upper_s
        self.dt_lower_s = dt_lower_s

        # 根据功能对储能单元进行分组
        self.energy_assets = [u for u in self.hess.all_units.values() if
                              any(keyword in u.id for keyword in ['phs', 'hes', 'tes', 'caes'])]
        self.smoothing_assets = [u for u in self.hess.all_units.values() if 'ees' in u.id]
        self.power_assets = [u for u in self.hess.all_units.values() if
                             any(keyword in u.id for keyword in ['fw', 'sc', 'smes'])]

    def solve_with_fallback(self, problem):
        """
        尝试使用多种求解器来解决优化问题，以提高求解成功率。
        """
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError("GUROBI 未能找到最优解。")
            return True
        except (cp.error.SolverError, ImportError):
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    raise cp.error.SolverError("ECOS 未能找到最优解。")
                return True
            except cp.error.SolverError:
                print(f"警告: 所有可用的求解器都求解失败。问题状态: {problem.status}")
                return False

    def solve_upper_level(self, current_soc, net_load_forecast_upper, grid_prices_upper, slow_task_signal_upper):
        """
        上层MPC: 负责能量型储能的低频经济调度和与电网的互动。
        """
        discharge_vars = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        charge_vars = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        soc_vars = {unit.id: cp.Variable(self.PH_upper + 1) for unit in self.energy_assets}
        grid_exchange = cp.Variable(self.PH_upper)

        grid_cost = cp.sum(cp.multiply(grid_exchange / 1e6, grid_prices_upper)) * (self.dt_upper_s / 3600)
        om_cost_mwh = [
            unit.om_cost_per_mwh * (discharge_vars[unit.id] + charge_vars[unit.id]) / 1e6
            for unit in self.energy_assets
        ]
        om_cost = cp.sum(om_cost_mwh) * (self.dt_upper_s / 3600)

        total_slow_dispatch = cp.sum([
            discharge_vars[unit.id] - charge_vars[unit.id] for unit in self.energy_assets
        ]) if self.energy_assets else 0

        tracking_penalty_slow = 1e-1 * cp.sum_squares(total_slow_dispatch - slow_task_signal_upper)
        soc_penalty = cp.sum([cp.sum_squares(soc_vars[unit.id][-1] - 0.5) for unit in self.energy_assets]) * 1e3
        objective = cp.Minimize(grid_cost + om_cost + tracking_penalty_slow + soc_penalty)

        constraints = [net_load_forecast_upper == total_slow_dispatch + grid_exchange]

        for unit in self.energy_assets:
            uid = unit.id
            for t in range(self.PH_upper):
                power_change_w = (charge_vars[uid][t] * unit.efficiency) - (discharge_vars[uid][t] / unit.efficiency)
                energy_change_mwh = power_change_w / 1e6 * (self.dt_upper_s / 3600)
                constraints.append(
                    soc_vars[uid][t + 1] == soc_vars[uid][t] + (energy_change_mwh / unit.capacity_mwh)
                )

            constraints.append(soc_vars[uid][0] == current_soc[uid])
            constraints.extend([
                soc_vars[uid] >= unit.soc_min,
                soc_vars[uid] <= unit.soc_max,
                discharge_vars[uid] <= unit.power_mw * 1e6,
                charge_vars[uid] <= unit.power_mw * 1e6
            ])

        problem = cp.Problem(objective, constraints)

        if self.solve_with_fallback(problem):
            return {
                "status": "optimal",
                "dispatch": {unit.id: (discharge_vars[unit.id].value - charge_vars[unit.id].value) for unit in
                             self.energy_assets},
                "grid_exchange": grid_exchange.value
            }
        else:
            return {"status": "failed", "dispatch": {}, "grid_exchange": np.zeros(self.PH_upper)}

    def solve_lower_level(self, current_soc, slow_asset_dispatch_ref, mid_task_signal, high_task_signal):
        """
        下层MPC: 负责平滑型和功率型储能的中高频协同控制，跟踪任务信号。
        """
        discharge_smooth = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.smoothing_assets}
        charge_smooth = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.smoothing_assets}
        soc_smooth = {u.id: cp.Variable(self.PH_lower + 1) for u in self.smoothing_assets}

        discharge_power = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.power_assets}
        charge_power = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.power_assets}
        soc_power = {u.id: cp.Variable(self.PH_lower + 1) for u in self.power_assets}

        total_smooth_dispatch = cp.sum([discharge_smooth[u.id] - charge_smooth[u.id] for u in
                                        self.smoothing_assets]) if self.smoothing_assets else 0
        tracking_penalty_mid = 1e5 * cp.sum_squares(total_smooth_dispatch - mid_task_signal)

        total_power_dispatch = cp.sum(
            [discharge_power[u.id] - charge_power[u.id] for u in self.power_assets]) if self.power_assets else 0
        tracking_penalty_high = 1e7 * cp.sum_squares(total_power_dispatch - high_task_signal)

        soc_penalty_smooth = cp.sum([cp.sum_squares(soc_smooth[u.id][-1] - 0.5) for u in self.smoothing_assets]) * 1e2
        soc_penalty_power = cp.sum([cp.sum_squares(soc_power[u.id][-1] - 0.5) for u in self.power_assets]) * 1e2

        objective = cp.Minimize(tracking_penalty_mid + tracking_penalty_high + soc_penalty_smooth + soc_penalty_power)

        constraints = []

        for unit in self.smoothing_assets:
            uid = unit.id
            for t in range(self.PH_lower):
                power_change_w = (charge_smooth[uid][t] * unit.efficiency) - (
                            discharge_smooth[uid][t] / unit.efficiency)
                energy_change_mwh = power_change_w / 1e6 * (self.dt_lower_s / 3600)
                constraints.append(soc_smooth[uid][t + 1] == soc_smooth[uid][t] + energy_change_mwh / unit.capacity_mwh)

            constraints.append(soc_smooth[uid][0] == current_soc[uid])
            constraints.extend([
                soc_smooth[uid] >= unit.soc_min, soc_smooth[uid] <= unit.soc_max,
                discharge_smooth[uid] <= unit.power_mw * 1e6, charge_smooth[uid] <= unit.power_mw * 1e6
            ])

        for unit in self.power_assets:
            uid = unit.id
            for t in range(self.PH_lower):
                power_change_w = (charge_power[uid][t] * unit.efficiency) - (discharge_power[uid][t] / unit.efficiency)
                energy_change_mwh = power_change_w / 1e6 * (self.dt_lower_s / 3600)
                constraints.append(soc_power[uid][t + 1] == soc_power[uid][t] + energy_change_mwh / unit.capacity_mwh)

            constraints.append(soc_power[uid][0] == current_soc[uid])
            constraints.extend([
                soc_power[uid] >= unit.soc_min, soc_power[uid] <= unit.soc_max,
                discharge_power[uid] <= unit.power_mw * 1e6, charge_power[uid] <= unit.power_mw * 1e6
            ])

        # ======================= ## 核心修正 ## =======================
        # 慢速资产的参考功率计划是一个字典，其值是Numpy数组。
        # 我们需要用 sum() 将这些数组加起来，得到一个总的功率数组。
        total_slow_dispatch_ref = sum(slow_asset_dispatch_ref.values()) if slow_asset_dispatch_ref else 0

        # 整个混合储能系统的总功率平衡约束
        total_hess_dispatch_plan = total_smooth_dispatch + total_power_dispatch + total_slow_dispatch_ref
        total_task_signal = mid_task_signal + high_task_signal + total_slow_dispatch_ref
        constraints.append(total_hess_dispatch_plan == total_task_signal)
        # =============================================================

        problem = cp.Problem(objective, constraints)
        final_dispatch = {}

        if self.solve_with_fallback(problem):
            for unit in self.smoothing_assets:
                final_dispatch[unit.id] = discharge_smooth[unit.id].value[0] - charge_smooth[unit.id].value[0]
            for unit in self.power_assets:
                final_dispatch[unit.id] = discharge_power[unit.id].value[0] - charge_power[unit.id].value[0]
        else:
            for unit in self.smoothing_assets: final_dispatch[unit.id] = 0
            for unit in self.power_assets: final_dispatch[unit.id] = 0

        return final_dispatch