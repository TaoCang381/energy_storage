# file: mpc_ems_hierarchical.py

import cvxpy as cp
import numpy as np


class HierarchicalMPCEms:
    def __init__(self, hess_system, upper_horizon, lower_horizon):
        self.hess = hess_system
        self.PH_upper = upper_horizon
        self.PH_lower = lower_horizon

        # 根据功能重新定义储能分组
        self.energy_assets = [u for u in self.hess.all_units.values() if
                              any(keyword in u.id for keyword in ['phs', 'hes', 'tes', 'caes'])]
        self.smoothing_assets = [u for u in self.hess.all_units.values() if 'ees' in u.id]
        self.power_assets = [u for u in self.hess.all_units.values() if
                             any(keyword in u.id for keyword in ['fw', 'sc', 'smes'])]

    def solve_with_fallback(self, problem, is_mip=False):
        """
        尝试用多种求解器求解优化问题。
        """
        try:
            # 优先使用高性能的商业求解器（如果安装了）
            problem.solve(solver=cp.GUROBI, verbose=False)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError("GUROBI failed to find an optimal solution.")
            return True
        except (cp.error.SolverError, ImportError):
            try:
                # 备用开源求解器
                problem.solve(solver=cp.ECOS, verbose=False)
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    raise cp.error.SolverError("ECOS failed to find an optimal solution.")
                return True
            except cp.error.SolverError:
                print(f"Warning: Optimization failed with all available solvers. Status: {problem.status}")
                return False

    def solve_upper_level(self, current_soc, net_load_forecast_upper, grid_prices_upper, slow_task_signal_upper):
        """
        上层MPC: 负责能量型储能的低频经济调度。
        """
        discharge_vars = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        charge_vars = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        soc_vars = {unit.id: cp.Variable(self.PH_upper + 1) for unit in self.energy_assets}
        grid_exchange = cp.Variable(self.PH_upper)

        grid_cost = cp.sum(grid_exchange * grid_prices_upper) * (self.hess.dt_upper / 3600)
        om_cost = cp.sum([
            unit.om_cost_per_mwh * (discharge_vars[unit.id] + charge_vars[unit.id])
            for unit in self.energy_assets
        ]) * (self.hess.dt_upper / 3600)

        total_slow_dispatch = cp.sum([
            discharge_vars[unit.id] - charge_vars[unit.id] for unit in self.energy_assets
        ]) if self.energy_assets else 0

        tracking_penalty_slow = 1e3 * cp.sum_squares(total_slow_dispatch - slow_task_signal_upper)
        objective = cp.Minimize(grid_cost + om_cost + tracking_penalty_slow)

        constraints = [net_load_forecast_upper + grid_exchange == total_slow_dispatch]

        for unit in self.energy_assets:
            uid = unit.id
            for t in range(self.PH_upper):
                constraints.append(
                    soc_vars[uid][t + 1] == soc_vars[uid][t] -
                    ((discharge_vars[uid][t] / unit.efficiency) - (charge_vars[uid][t] * unit.efficiency)) *
                    (self.hess.dt_upper / 3600) / unit.capacity_mwh
                )
            constraints.append(soc_vars[uid][0] == current_soc[uid])
            constraints.extend([
                soc_vars[uid] >= unit.soc_min,
                soc_vars[uid] <= unit.soc_max,
                discharge_vars[uid] <= unit.power_mw * 1e6,  # 修正: power_m_w -> power_mw
                charge_vars[uid] <= unit.power_mw * 1e6  # 修正: power_m_w -> power_mw
            ])

        problem = cp.Problem(objective, constraints)

        if self.solve_with_fallback(problem):
            return {
                "status": "optimal",
                "dispatch": {unit.id: (discharge_vars[unit.id].value - charge_vars[unit.id].value) for unit in
                             self.energy_assets}
            }
        else:
            return {"status": "failed", "dispatch": {}}

    def solve_lower_level(self, current_soc, reference_signals, mid_task_signal, high_task_signal):
        """
        下层MPC: 负责平滑型和功率型储能的中高频协同控制。
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

        om_cost_smooth = cp.sum(
            [u.om_cost_per_mwh * (discharge_smooth[u.id] + charge_smooth[u.id]) for u in self.smoothing_assets]) * (
                                     self.hess.dt / 3600)
        om_cost_power = cp.sum(
            [u.om_cost_per_mwh * (discharge_power[u.id] + charge_power[u.id]) for u in self.power_assets]) * (
                                    self.hess.dt / 3600)

        objective = cp.Minimize(tracking_penalty_mid + tracking_penalty_high + om_cost_smooth + om_cost_power)
        constraints = []

        for unit in self.smoothing_assets:
            uid = unit.id
            for t in range(self.PH_lower):
                constraints.append(soc_smooth[uid][t + 1] == soc_smooth[uid][t] - (
                        (discharge_smooth[uid][t] / unit.efficiency) - (charge_smooth[uid][t] * unit.efficiency)) *
                                   (self.hess.dt / 3600) / unit.capacity_mwh)
            constraints.append(soc_smooth[uid][0] == current_soc[uid])
            constraints.extend([
                soc_smooth[uid] >= unit.soc_min,
                soc_smooth[uid] <= unit.soc_max,
                discharge_smooth[uid] <= unit.power_mw * 1e6,  # 修正: power_m_w -> power_mw
                charge_smooth[uid] <= unit.power_mw * 1e6  # 修正: power_m_w -> power_mw
            ])

        for unit in self.power_assets:
            uid = unit.id
            for t in range(self.PH_lower):
                constraints.append(soc_power[uid][t + 1] == soc_power[uid][t] - (
                        (discharge_power[uid][t] / unit.efficiency) - (charge_power[uid][t] * unit.efficiency)) *
                                   (self.hess.dt / 3600) / unit.capacity_mwh)
            constraints.append(soc_power[uid][0] == current_soc[uid])
            constraints.extend([
                soc_power[uid] >= unit.soc_min,
                soc_power[uid] <= unit.soc_max,
                discharge_power[uid] <= unit.power_mw * 1e6,  # 修正: power_m_w -> power_mw
                charge_power[uid] <= unit.power_mw * 1e6  # 修正: power_m_w -> power_mw
            ])

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

        for uid, dispatch_plan_lower in reference_signals.get("slow_asset_dispatch", {}).items():
            if dispatch_plan_lower is not None and len(dispatch_plan_lower) > 0:
                final_dispatch[uid] = dispatch_plan_lower[0]

        return final_dispatch