# file: PythonProject/mpc_ems.py (V8.0 - 八储能混合整数最终版)

import cvxpy as cp
import numpy as np


class MPCEnergyManagementSystem:
    def __init__(self, hess_system, prediction_horizon):
        self.hess = hess_system
        self.PH = prediction_horizon

    def solve(self, current_soc_dict, predicted_wind, predicted_solar, predicted_load, grid_prices_per_mwh):
        all_units = list(self.hess.all_units.values())
        dt_h = (60 * 15) / 3600.0

        # --- 1. 定义变量 ---
        charge_power = {u.id: cp.Variable(self.PH, nonneg=True) for u in all_units}
        discharge_power = {u.id: cp.Variable(self.PH, nonneg=True) for u in all_units}
        soc = {u.id: cp.Variable(self.PH + 1) for u in all_units}
        grid_power = cp.Variable(self.PH)
        slack_shortage = cp.Variable(self.PH, nonneg=True)
        slack_surplus = cp.Variable(self.PH, nonneg=True)

        # --- 为HES和CAES引入二进制变量 ---
        u_hes_ely, u_hes_fc, u_hes_ely_start, u_hes_fc_start = None, None, None, None
        u_caes_comp, u_caes_gen, u_caes_comp_start, u_caes_gen_start = None, None, None, None

        if self.hess.all_units.get("hes_01"):
            u_hes_ely = cp.Variable(self.PH, boolean=True)
            u_hes_fc = cp.Variable(self.PH, boolean=True)
            u_hes_ely_start = cp.Variable(self.PH, boolean=True)
            u_hes_fc_start = cp.Variable(self.PH, boolean=True)

        if self.hess.all_units.get("caes_01"):
            u_caes_comp = cp.Variable(self.PH, boolean=True)
            u_caes_gen = cp.Variable(self.PH, boolean=True)
            u_caes_comp_start = cp.Variable(self.PH, boolean=True)
            u_caes_gen_start = cp.Variable(self.PH, boolean=True)

        constraints = []
        # --- 2. 添加储能单元自身约束 ---
        for unit in all_units:
            constraints += [soc[unit.id][0] == current_soc_dict[unit.id]]
            constraints += [charge_power[unit.id] <= getattr(unit, 'P_comp_rated', unit.rated_power_w)]
            constraints += [discharge_power[unit.id] <= getattr(unit, 'P_gen_rated', unit.rated_power_w)]
            constraints += [soc[unit.id] >= unit.soc_min, soc[unit.id] <= unit.soc_max]

            for t in range(self.PH):
                # ========================= 动态方程约束 (已包含全部8种储能) =========================
                uid = unit.id
                if 'ees' in uid:
                    delta_e = (discharge_power[uid][t] / unit.eta_dis - charge_power[uid][
                        t] * unit.eta_ch) * dt_h / 1000
                    constraints += [soc[uid][t + 1] == soc[uid][t] - (delta_e / unit.nominal_capacity_kwh)]
                elif 'fw' in uid:
                    e_total_J = 0.5 * unit.J * (unit.omega_max ** 2 - unit.omega_min ** 2)
                    capacity_kwh = e_total_J / 3.6e6
                    net_power_w = discharge_power[uid][t] - charge_power[uid][t]
                    delta_e_kwh = net_power_w * dt_h / 1000
                    constraints += [soc[uid][t + 1] == soc[uid][t] - (
                                delta_e_kwh / (capacity_kwh if capacity_kwh > 1e-6 else 1e-6))]
                elif 'phs' in uid:
                    flow_in = (charge_power[uid][t] * unit.eta_pump) / (1000 * 9.81 * unit.h_eff)
                    flow_out = discharge_power[uid][t] / (1000 * 9.81 * unit.h_eff * unit.eta_gen)
                    delta_v = (flow_in - flow_out) * (dt_h * 3600)
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_v / unit.V_ur_max]
                elif 'tes' in uid:
                    heat_in = charge_power[uid][t] * unit.eta_e2h
                    heat_out = discharge_power[uid][t] / unit.eta_h2e
                    heat_loss = unit.H_tes_max_J * unit.theta_loss * (dt_h * 3600)
                    delta_h = (heat_in - heat_out) * (dt_h * 3600) - heat_loss
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_h / unit.H_tes_max_J]
                elif 'hes' in uid:
                    m_dot_ely = (charge_power[uid][t] / 1000) / unit.eta_ely_kwh_kg
                    m_dot_fc = (discharge_power[uid][t] / 1000) / (33.3 * unit.eta_fc_elec)
                    delta_m_kg = (m_dot_ely - m_dot_fc) * dt_h
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_m_kg / unit.M_tank_max]
                elif 'sc' in uid:
                    e_total_J = 0.5 * unit.C_sc * (unit.V_max ** 2 - unit.V_min ** 2)
                    capacity_kwh = e_total_J / 3.6e6
                    net_power_w = discharge_power[uid][t] - charge_power[uid][t]
                    delta_e_kwh = net_power_w * dt_h / 1000
                    constraints += [soc[uid][t + 1] == soc[uid][t] - (
                                delta_e_kwh / (capacity_kwh if capacity_kwh > 1e-6 else 1e-6))]
                elif 'smes' in uid:
                    e_total_J = 0.5 * unit.L_smes * unit.I_max ** 2
                    capacity_kwh = e_total_J / 3.6e6
                    net_power_w = discharge_power[uid][t] - charge_power[uid][t]
                    delta_e_kwh = net_power_w * dt_h / 1000
                    constraints += [soc[uid][t + 1] == soc[uid][t] - (
                                delta_e_kwh / (capacity_kwh if capacity_kwh > 1e-6 else 1e-6))]
                elif 'caes' in uid:
                    mass_in_kg_per_h = (charge_power[uid][t] / 1000) * unit.eta_charge_rate
                    mass_out_kg_per_h = (discharge_power[uid][t] / 1000) * unit.eta_air_usage
                    delta_m_kg = (mass_in_kg_per_h - mass_out_kg_per_h) * dt_h
                    constraints += [soc[uid][t + 1] == soc[uid][t] + delta_m_kg / unit.M_air_max]

        # --- 3. 添加混合整数约束 ---
        if u_hes_ely is not None:
            constraints += [charge_power["hes_01"] <= self.hess.all_units["hes_01"].P_ely_rated * u_hes_ely]
            constraints += [discharge_power["hes_01"] <= self.hess.all_units["hes_01"].P_fc_rated * u_hes_fc]
            constraints += [u_hes_ely + u_hes_fc <= 1]
            # (启停逻辑省略以简化，你可以在此基础上添加)
        if u_caes_comp is not None:
            constraints += [charge_power["caes_01"] <= self.hess.all_units["caes_01"].P_comp_rated * u_caes_comp]
            constraints += [discharge_power["caes_01"] <= self.hess.all_units["caes_01"].P_gen_rated * u_caes_gen]
            constraints += [u_caes_comp + u_caes_gen <= 1]
            # (启停逻辑省略以简化)

        # --- 4. 系统级约束 ---
        total_hess_power = sum(discharge_power.values()) - sum(charge_power.values())
        constraints += [(
                                    predicted_wind + predicted_solar + total_hess_power + grid_power) - predicted_load == slack_surplus - slack_shortage]
        grid_max_power_w = 400e6  # 增大了电网交互限额
        constraints += [grid_power <= grid_max_power_w, grid_power >= -grid_max_power_w]

        # --- 5. 定义目标函数 ---
        grid_cost = cp.sum(cp.multiply(grid_prices_per_mwh / 1e6, grid_power) * dt_h)
        om_cost = 0
        for unit in all_units:
            if 'hes' not in unit.id and 'caes' not in unit.id:  # 对非启停储能计算吞吐量成本
                total_throughput_w = charge_power[unit.id] + discharge_power[unit.id]
                om_cost += cp.sum(getattr(unit, 'cost_per_kwh', 0.01) * (total_throughput_w * dt_h / 1000))
        # (启停成本省略以简化)

        penalty_price_per_mwh = 10000
        slack_cost = penalty_price_per_mwh * cp.sum(slack_shortage + slack_surplus) * dt_h / 1e6
        objective = cp.Minimize(grid_cost + om_cost + slack_cost)

        # --- 6. 求解问题 ---
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            optimal_dispatch = {"grid_power": grid_power.value}
            for unit in all_units:
                net_power = 0
                if discharge_power[unit.id].value is not None and charge_power[unit.id].value is not None:
                    net_power = discharge_power[unit.id].value[0] - charge_power[unit.id].value[0]
                optimal_dispatch[unit.id + "_power"] = net_power
            return optimal_dispatch
        else:
            print(f"FATAL: MPC problem is '{problem.status}'. Check model constraints and parameters.")
            return None