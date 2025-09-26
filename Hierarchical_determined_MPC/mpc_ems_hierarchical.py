# file: mpc_ems_hierarchical.py (Final Numerically Stable Version V6.0)
# Note: This version separates the lower-level optimization into two distinct problems
# to resolve the numerical instability caused by vastly different asset capacities.

import cvxpy as cp
import numpy as np


class HierarchicalMPCEms:
    def __init__(self, hess_system, upper_horizon, lower_horizon):
        self.hess = hess_system
        self.PH_upper = upper_horizon
        self.PH_lower = lower_horizon
        self.energy_assets = [u for u in self.hess.all_units.values() if
                              any(keyword in u.id for keyword in ['phs', 'hes', 'tes', 'caes'])]
        self.smoothing_assets = [u for u in self.hess.all_units.values() if 'ees' in u.id]
        self.power_assets = [u for u in self.hess.all_units.values() if
                             any(keyword in u.id for keyword in ['fw', 'sc', 'smes'])]

    def solve_with_fallback(self, problem):
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)  # Set verbose=False for final run
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError("GUROBI failed or did not find an optimal solution.")
            return True
        except (cp.error.SolverError, ImportError, AttributeError):
            try:
                problem.solve(solver=cp.ECOS, verbose=False, max_iters=500, abstol=1e-6)
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    raise cp.error.SolverError("ECOS failed or did not find an optimal solution.")
                return True
            except cp.error.SolverError as e:
                print(f"Warning: All solvers failed. Final problem status: {problem.status}, Error: {e}")
                return False

    def solve_upper_level(self, current_soc, grid_prices_upper, slow_task_signal_upper):
        # This function remains unchanged and is already working correctly.
        p_charge_upper_dc = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        p_discharge_upper_dc = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        soc_vars_upper = {unit.id: cp.Variable(self.PH_upper + 1) for unit in self.energy_assets}
        grid_exchange = cp.Variable(self.PH_upper)

        dt_upper_h = (15 * 60) / 3600.0
        grid_cost = cp.sum(cp.multiply(grid_exchange, grid_prices_upper)) * dt_upper_h
        total_om_cost = 0
        for unit in self.energy_assets:
            power_ac_discharge = p_discharge_upper_dc[unit.id] * unit.efficiency
            power_ac_charge = p_charge_upper_dc[unit.id] / unit.efficiency
            total_energy_throughput_mwh = cp.sum(power_ac_discharge + power_ac_charge) * dt_upper_h
            total_om_cost += unit.om_cost_per_mwh * total_energy_throughput_mwh
        objective = cp.Minimize(grid_cost + total_om_cost)

        constraints = []
        total_slow_dispatch_ac = cp.sum([
            p_discharge_upper_dc[u.id] * u.efficiency - p_charge_upper_dc[u.id] / u.efficiency
            for u in self.energy_assets
        ]) if self.energy_assets else 0

        constraints.append(slow_task_signal_upper == total_slow_dispatch_ac + grid_exchange)

        for unit in self.energy_assets:
            uid = unit.id
            for t in range(self.PH_upper):
                energy_change_mwh = (p_charge_upper_dc[uid][t] - p_discharge_upper_dc[uid][t]) * dt_upper_h
                if unit.capacity_mwh > 1e-6:
                    constraints.append(
                        soc_vars_upper[uid][t + 1] == soc_vars_upper[uid][t] + energy_change_mwh / unit.capacity_mwh)
                else:
                    constraints.append(soc_vars_upper[uid][t + 1] == soc_vars_upper[uid][t])

            constraints.append(p_discharge_upper_dc[uid] * unit.efficiency <= unit.power_m_w)
            constraints.append(p_charge_upper_dc[uid] <= unit.power_m_w * unit.efficiency)
            constraints.append(soc_vars_upper[uid][0] == current_soc[uid])
            constraints.extend([soc_vars_upper[uid] >= unit.soc_min, soc_vars_upper[uid] <= unit.soc_max])

        problem = cp.Problem(objective, constraints)
        if self.solve_with_fallback(problem):
            dispatch_ac = {unit.id: (p_discharge_upper_dc[unit.id].value * unit.efficiency) - (
                    p_charge_upper_dc[unit.id].value / unit.efficiency) for unit in self.energy_assets}
            return {"status": "optimal", "dispatch": dispatch_ac, "grid_exchange": grid_exchange.value}
        else:
            return {"status": "failed", "dispatch": {}, "grid_exchange": np.zeros(self.PH_upper)}

    def solve_lower_level(self, current_soc, mid_task_signal, high_task_signal):
        """
        Main modification: This function now coordinates two separate, smaller optimizations.
        """
        final_dispatch = {}

        # --- Optimization 1: Smoothing Assets (e.g., ees) for Mid-Frequency Signal ---
        if self.smoothing_assets:
            p_ch_smooth_dc = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.smoothing_assets}
            p_dis_smooth_dc = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.smoothing_assets}
            soc_smooth = {u.id: cp.Variable(self.PH_lower + 1) for u in self.smoothing_assets}

            total_smooth_dispatch_ac = cp.sum(
                [p_dis_smooth_dc[u.id] * u.efficiency - p_ch_smooth_dc[u.id] / u.efficiency for u in
                 self.smoothing_assets]
            )

            objective_mid = cp.Minimize(100 * cp.sum_squares(total_smooth_dispatch_ac - mid_task_signal))

            constraints_mid = []
            dt_h = self.hess.dt_s / 3600
            for unit in self.smoothing_assets:
                uid = unit.id
                for t in range(self.PH_lower):
                    energy_change_mwh = (p_ch_smooth_dc[uid][t] - p_dis_smooth_dc[uid][t]) * dt_h
                    constraints_mid.append(
                        soc_smooth[uid][t + 1] == soc_smooth[uid][t] + energy_change_mwh / unit.capacity_mwh)

                constraints_mid.append(p_dis_smooth_dc[uid] * unit.efficiency <= unit.power_m_w)
                constraints_mid.append(p_ch_smooth_dc[uid] <= unit.power_m_w * unit.efficiency)
                constraints_mid.append(soc_smooth[uid][0] == current_soc[uid])
                constraints_mid.extend([soc_smooth[uid] >= unit.soc_min, soc_smooth[uid] <= unit.soc_max])

            problem_mid = cp.Problem(objective_mid, constraints_mid)
            if self.solve_with_fallback(problem_mid):
                for unit in self.smoothing_assets:
                    dispatch_ac = (p_dis_smooth_dc[unit.id].value[0] * unit.efficiency) - (
                                p_ch_smooth_dc[unit.id].value[0] / unit.efficiency)
                    final_dispatch[unit.id] = dispatch_ac if dispatch_ac is not None else 0
            else:
                for unit in self.smoothing_assets: final_dispatch[unit.id] = 0

        # --- Optimization 2: Power Assets (e.g., fw, sc, smes) for High-Frequency Signal ---
        if self.power_assets:
            p_ch_power_dc = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.power_assets}
            p_dis_power_dc = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.power_assets}
            soc_power = {u.id: cp.Variable(self.PH_lower + 1) for u in self.power_assets}

            total_power_dispatch_ac = cp.sum(
                [p_dis_power_dc[u.id] * u.efficiency - p_ch_power_dc[u.id] / u.efficiency for u in self.power_assets]
            )

            objective_high = cp.Minimize(1000 * cp.sum_squares(total_power_dispatch_ac - high_task_signal))

            constraints_high = []
            dt_h = self.hess.dt_s / 3600
            for unit in self.power_assets:
                uid = unit.id
                for t in range(self.PH_lower):
                    energy_change_mwh = (p_ch_power_dc[uid][t] - p_dis_power_dc[uid][t]) * dt_h
                    constraints_high.append(
                        soc_power[uid][t + 1] == soc_power[uid][t] + energy_change_mwh / unit.capacity_mwh)

                constraints_high.append(p_dis_power_dc[uid] * unit.efficiency <= unit.power_m_w)
                constraints_high.append(p_ch_power_dc[uid] <= unit.power_m_w * unit.efficiency)
                constraints_high.append(soc_power[uid][0] == current_soc[uid])
                constraints_high.extend([soc_power[uid] >= unit.soc_min, soc_power[uid] <= unit.soc_max])

            problem_high = cp.Problem(objective_high, constraints_high)
            if self.solve_with_fallback(problem_high):
                for unit in self.power_assets:
                    dispatch_ac = (p_dis_power_dc[unit.id].value[0] * unit.efficiency) - (
                                p_ch_power_dc[unit.id].value[0] / unit.efficiency)
                    final_dispatch[unit.id] = dispatch_ac if dispatch_ac is not None else 0
            else:
                for unit in self.power_assets: final_dispatch[unit.id] = 0

        return final_dispatch