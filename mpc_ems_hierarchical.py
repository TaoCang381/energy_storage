# file: mpc_ems_hierarchical.py (V1.8 - Fully Linearized Version)
# 备注：将上下层模型均转化为线性规划，以确保求解器兼容性和鲁棒性。
#       简化了上层模型的混合整数部分，降低求解难度。

import cvxpy as cp
import numpy as np


class HierarchicalMPCEms:
    def __init__(self, hess_system, upper_horizon, lower_horizon):
        self.hess = hess_system
        self.PH_upper = upper_horizon
        self.PH_lower = lower_horizon

        self.slow_assets = [u for u in self.hess.all_units.values() if
                            any(keyword in u.id for keyword in ['phs', 'hes', 'tes', 'caes'])]
        self.fast_assets = [u for u in self.hess.all_units.values() if
                            any(keyword in u.id for keyword in ['fw', 'sc', 'smes', 'ees'])]

    def solve_with_fallback(self, problem, is_mip=False):
        """
        Tries to solve with GUROBI first, then falls back to an open-source solver.
        """
        try:
            problem.solve(solver=cp.GUROBI, verbose=False, FeasibilityTol=1e-5, OptimalityTol=1e-5, TimeLimit=30)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                return True
        except (cp.error.SolverError, ValueError):
            print("    GUROBI failed. Falling back to open-source solver...")

        solver = cp.CBC if is_mip else None
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                return True
        except Exception as e:
            print(f"    Fallback solver also failed: {e}")

        return False

    def solve_upper_layer(self, initial_soc_dict, pred_wind, pred_solar, pred_load, grid_prices, dt_h_upper):
        print("  Running Upper Layer MPC...")
        charge_power = {u.id: cp.Variable(self.PH_upper, nonneg=True) for u in self.slow_assets}
        discharge_power = {u.id: cp.Variable(self.PH_upper, nonneg=True) for u in self.slow_assets}
        soc = {u.id: cp.Variable(self.PH_upper + 1) for u in self.slow_assets}
        grid_power = cp.Variable(self.PH_upper)
        constraints = []

        # ========================= 修改 1: 简化混合整数模型 =========================
        # 仅为需要启停控制的单元（HES, CAES）定义布尔变量，大大降低模型复杂度
        milp_units_ids = {u.id for u in self.slow_assets if 'hes' in u.id or 'caes' in u.id}
        u_charge = {uid: cp.Variable(self.PH_upper, boolean=True) for uid in milp_units_ids}
        u_discharge = {uid: cp.Variable(self.PH_upper, boolean=True) for uid in milp_units_ids}
        # ========================= 修改结束 =========================

        soc_slack_min_upper = {u.id: cp.Variable(self.PH_upper + 1, nonneg=True) for u in self.slow_assets}
        soc_slack_max_upper = {u.id: cp.Variable(self.PH_upper + 1, nonneg=True) for u in self.slow_assets}
        P_grid_max_w = 500e6
        constraints += [grid_power <= P_grid_max_w, grid_power >= -P_grid_max_w]

        for unit in self.slow_assets:
            uid = unit.id
            constraints += [soc[uid][0] == initial_soc_dict[uid]]
            constraints += [soc[uid] >= unit.soc_min - soc_slack_min_upper[uid]]
            constraints += [soc[uid] <= unit.soc_max + soc_slack_max_upper[uid]]

            # 应用混合整数约束或线性约束
            if uid in milp_units_ids:
                # 对于 HES 和 CAES，使用布尔变量进行启停控制
                constraints += [charge_power[uid] <= getattr(unit, 'P_comp_rated', 1e9) * u_charge[uid]]
                constraints += [discharge_power[uid] <= getattr(unit, 'P_gen_rated', 1e9) * u_discharge[uid]]
                constraints += [u_charge[uid] + u_discharge[uid] <= 1]  # 不能同时充放电
            else:
                # 对于 PHS 和 TES，作为常规线性储能处理，不使用布尔变量
                constraints += [charge_power[uid] <= getattr(unit, 'P_pump_rated', unit.rated_power_w)]
                constraints += [discharge_power[uid] <= getattr(unit, 'P_gen_rated', unit.rated_power_w)]

            # 储能动态模型 (与之前一致)
            for t in range(self.PH_upper):
                if 'phs' in uid:
                    flow_in = (charge_power[uid][t] * unit.eta_pump) / (1000 * 9.81 * unit.h_eff)
                    flow_out = discharge_power[uid][t] / (1000 * 9.81 * unit.h_eff * unit.eta_gen)
                    delta_v = (flow_in - flow_out) * (dt_h_upper * 3600)
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_v / unit.V_ur_max]
                elif 'hes' in uid:
                    m_dot_ely = (charge_power[uid][t] / 1000) / unit.eta_ely_kwh_kg
                    m_dot_fc = (discharge_power[uid][t] / 1000) / (33.3 * unit.eta_fc_elec)
                    delta_m_kg = (m_dot_ely - m_dot_fc) * dt_h_upper
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_m_kg / unit.M_tank_max]
                elif 'tes' in uid:
                    heat_in = charge_power[uid][t] * unit.eta_e2h
                    heat_out = discharge_power[uid][t] / unit.eta_h2e
                    heat_loss = unit.H_tes_max_J * unit.theta_loss * (dt_h_upper * 3600)
                    delta_h = (heat_in - heat_out) * (dt_h_upper * 3600) - heat_loss
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_h / unit.H_tes_max_J]
                elif 'caes' in uid:
                    mass_in_kg_per_h = (charge_power[uid][t] / 1000) * unit.eta_charge_rate
                    mass_out_kg_per_h = (discharge_power[uid][t] / 1000) * unit.eta_air_usage
                    delta_m_kg = (mass_in_kg_per_h - mass_out_kg_per_h) * dt_h_upper
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_m_kg / unit.M_air_max]

        # 目标函数部分（已是线性）
        all_discharge_slow = [p for p in discharge_power.values()]
        all_charge_slow = [c for c in charge_power.values()]
        total_slow_power = cp.sum(all_discharge_slow) - cp.sum(all_charge_slow)
        net_load = pred_load - (pred_wind + pred_solar)
        fast_asset_power_balance = net_load - total_slow_power - grid_power
        slack_power_positive = cp.Variable(self.PH_upper, nonneg=True, name="slack_pos")
        slack_power_negative = cp.Variable(self.PH_upper, nonneg=True, name="slack_neg")
        constraints += [fast_asset_power_balance == slack_power_positive - slack_power_negative]
        grid_cost = cp.sum(cp.multiply(grid_prices / 1e6, grid_power) * dt_h_upper)
        om_cost_factors = {'phs': 0.005, 'hes': 0.1, 'tes': 0.01, 'caes': 0.02}
        om_cost = sum(
            cp.sum(om_cost_factors[uid.split('_')[0]] * (charge_power[uid] + discharge_power[uid]) * dt_h_upper / 1000)
            for uid in charge_power)
        slack_penalty = 1e6 * cp.sum(slack_power_positive + slack_power_negative)
        soc_penalty_upper = sum(
            1e5 * (cp.sum(soc_slack_min_upper[uid]) + cp.sum(soc_slack_max_upper[uid])) for uid in
            soc_slack_min_upper)
        objective = cp.Minimize(grid_cost + om_cost + slack_penalty + soc_penalty_upper)
        problem = cp.Problem(objective, constraints)

        if self.solve_with_fallback(problem, is_mip=True):
            return {
                "slow_asset_dispatch": {uid: discharge_power[uid].value - charge_power[uid].value for uid in
                                        charge_power},
                "fast_asset_net_power_ref": fast_asset_power_balance.value
            }
        else:
            print(f"FATAL: Upper Layer MPC failed with all available solvers.")
            return None

    def solve_lower_layer(self, initial_soc_dict, reference_signals, dt_h_lower, steps_per_upper):
        print("    Running Lower Layer MPC...")
        charge_power = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.fast_assets}
        discharge_power = {u.id: cp.Variable(self.PH_lower, nonneg=True) for u in self.fast_assets}
        soc = {u.id: cp.Variable(self.PH_lower + 1) for u in self.fast_assets}
        constraints = []
        soc_slack_min = {u.id: cp.Variable(self.PH_lower + 1, nonneg=True) for u in self.fast_assets}
        soc_slack_max = {u.id: cp.Variable(self.PH_lower + 1, nonneg=True) for u in self.fast_assets}
        for unit in self.fast_assets:
            uid = unit.id
            constraints += [soc[uid][0] == initial_soc_dict[uid]]
            constraints += [soc[uid] >= unit.soc_min - soc_slack_min[uid]]
            constraints += [soc[uid] <= unit.soc_max + soc_slack_max[uid]]
            constraints += [charge_power[uid] <= unit.rated_power_w]
            constraints += [discharge_power[uid] <= unit.rated_power_w]
            for t in range(self.PH_lower):
                capacity_kwh = getattr(unit, 'nominal_capacity_kwh', 1e-6)
                if 'fw' in uid:
                    e_total_J = 0.5 * unit.J * (unit.omega_max ** 2 - unit.omega_min ** 2)
                    capacity_kwh = e_total_J / 3.6e6
                elif 'sc' in uid:
                    e_total_J = 0.5 * unit.C_sc * (unit.V_max ** 2 - unit.V_min ** 2)
                    capacity_kwh = e_total_J / 3.6e6
                elif 'smes' in uid:
                    e_total_J = 0.5 * unit.L_smes * unit.I_max ** 2
                    capacity_kwh = e_total_J / 3.6e6

                net_power_w = discharge_power[uid][t] - charge_power[uid][t]
                delta_e_kwh = net_power_w * dt_h_lower / 1000
                constraints += [
                    soc[uid][t + 1] == soc[uid][t] - delta_e_kwh / (capacity_kwh if capacity_kwh > 1e-6 else 1e-6)]

        all_discharge = [p for p in discharge_power.values()]
        all_charge = [p for p in charge_power.values()]
        total_fast_power = cp.sum(all_discharge) - cp.sum(all_charge)

        fast_power_ref_broadcast = np.repeat(
            reference_signals["fast_asset_net_power_ref"][:self.PH_lower // steps_per_upper + 1], steps_per_upper)[
                                   :self.PH_lower]

        # ========================= 修改 2: 线性化功率追踪惩罚 =========================
        tracking_error_pos = cp.Variable(self.PH_lower, nonneg=True)
        tracking_error_neg = cp.Variable(self.PH_lower, nonneg=True)
        constraints += [total_fast_power - fast_power_ref_broadcast == tracking_error_pos - tracking_error_neg]
        tracking_penalty_cost = 1e-4 * cp.sum(tracking_error_pos + tracking_error_neg)
        # ========================= 修改结束 =========================

        om_cost_factors = {'fw': 0.05, 'sc': 0.02, 'smes': 0.02, 'ees': 0.08}
        om_cost = sum(
            cp.sum(om_cost_factors[uid.split('_')[0]] * (charge_power[uid] + discharge_power[uid]) * dt_h_lower / 1000)
            for uid in charge_power)

        # ========================= 修改 3: 线性化SOC惩罚 =========================
        soc_penalty = sum(
            1e5 * (cp.sum(soc_slack_min[uid]) + cp.sum(soc_slack_max[uid])) for uid in soc_slack_min)
        # ========================= 修改结束 =========================

        objective = cp.Minimize(om_cost + tracking_penalty_cost + soc_penalty)
        problem = cp.Problem(objective, constraints)

        if self.solve_with_fallback(problem, is_mip=False):
            final_dispatch = {}
            # 注意：此处慢速储能功率需要从参考信号中获取第一个时间步的值
            for uid, dispatch_val_series in reference_signals["slow_asset_dispatch"].items():
                # 确保dispatch_val_series是Numpy数组，以防其是None或标量
                if dispatch_val_series is not None and dispatch_val_series.ndim > 0:
                    final_dispatch[uid + "_power"] = dispatch_val_series[0]
                else:
                    final_dispatch[uid + "_power"] = 0  # 如果上层求解失败，给一个默认值

            for unit in self.fast_assets:
                net_power = 0
                if discharge_power[unit.id].value is not None and charge_power[unit.id].value is not None:
                    net_power = discharge_power[unit.id].value[0] - charge_power[unit.id].value[0]
                final_dispatch[unit.id + "_power"] = net_power
            return final_dispatch
        else:
            print(f"FATAL: Lower Layer MPC failed with all available solvers.")
            return None