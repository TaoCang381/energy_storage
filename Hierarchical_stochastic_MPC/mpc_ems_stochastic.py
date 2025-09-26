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

    def solve_stochastic_upper_level(self, current_soc, grid_prices_upper, net_load_forecast_upper, scenarios,
                                     probabilities):
        """
        两阶段随机优化版本的上层MPC求解器。
        - net_load_forecast_upper: 净负荷预测值 (负荷 - 可再生能源)。正值表示需要供电。
        - scenarios: 预测误差场景, shape: [num_scenarios, PH_upper]。
        - probabilities: 每个场景的概率。
        """

        # --- 1. 获取参数 ---
        num_scenarios = len(probabilities)
        dt_upper_h = (15 * 60) / 3600.0  # 15分钟转为小时

        # --- 2. 定义决策变量 ---
        # 【第一阶段变量】: "Here-and-Now"决策, 与场景s无关
        # 慢速储能(energy_assets)的充放电功率和SOC
        p_charge_upper_dc = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        p_discharge_upper_dc = {unit.id: cp.Variable(self.PH_upper, nonneg=True) for unit in self.energy_assets}
        soc_vars_upper = {unit.id: cp.Variable(self.PH_upper + 1) for unit in self.energy_assets}

        # 【第二阶段变量】: "Recourse"决策, 必须为每个场景s都定义一套
        # 快速储能(smoothing_assets)的充放电功率和SOC
        p_ch_smooth_dc = {(s, u.id): cp.Variable(self.PH_upper, nonneg=True) for s in range(num_scenarios) for u in
                          self.smoothing_assets}
        p_dis_smooth_dc = {(s, u.id): cp.Variable(self.PH_upper, nonneg=True) for s in range(num_scenarios) for u in
                           self.smoothing_assets}
        soc_smooth = {(s, u.id): cp.Variable(self.PH_upper + 1) for s in range(num_scenarios) for u in
                      self.smoothing_assets}

        # 电网交互功率也是第二阶段的补救措施
        grid_exchange = cp.Variable((num_scenarios, self.PH_upper))

        # --- 3. 构建目标函数: Min(第一阶段成本 + 第二阶段期望成本) ---
        # 第一阶段成本: 慢速储能的运维成本 (与场景无关)
        cost_stage1 = 0
        for unit in self.energy_assets:
            power_ac_discharge = p_discharge_upper_dc[unit.id] * unit.efficiency
            power_ac_charge = p_charge_upper_dc[unit.id] / unit.efficiency
            total_energy_throughput_mwh = cp.sum(power_ac_discharge + power_ac_charge) * dt_upper_h
            cost_stage1 += unit.om_cost_per_mwh * total_energy_throughput_mwh

        # 第二阶段成本: 对每个场景s，计算其运行成本(电网+快速储能)，再按概率加权求和
        cost_stage2_expected = 0
        for s in range(num_scenarios):
            # 场景s下的电网成本
            grid_cost_s = cp.sum(cp.multiply(grid_exchange[s, :], grid_prices_upper)) * dt_upper_h

            # 场景s下的快速储能运维成本
            om_cost_smooth_s = 0
            for unit in self.smoothing_assets:
                power_ac_dis_s = p_dis_smooth_dc[s, unit.id] * unit.efficiency
                power_ac_ch_s = p_ch_smooth_dc[s, unit.id] / unit.efficiency
                om_cost_smooth_s += unit.om_cost_per_mwh * cp.sum(power_ac_dis_s + power_ac_ch_s) * dt_upper_h

            cost_stage2_expected += probabilities[s] * (grid_cost_s + om_cost_smooth_s)

        objective = cp.Minimize(cost_stage1 + cost_stage2_expected)

        # --- 4. 构建约束条件 ---
        constraints = []

        # 【第一阶段约束】: 慢速储能的物理约束 (与场景无关)
        for unit in self.energy_assets:
            uid = unit.id
            constraints.append(soc_vars_upper[uid][0] == current_soc[uid])
            constraints.extend([soc_vars_upper[uid] >= unit.soc_min, soc_vars_upper[uid] <= unit.soc_max])
            constraints.append(p_discharge_upper_dc[uid] * unit.efficiency <= unit.power_m_w)
            constraints.append(p_charge_upper_dc[uid] <= unit.power_m_w * unit.efficiency)
            for t in range(self.PH_upper):
                energy_change_mwh = (p_charge_upper_dc[uid][t] - p_discharge_upper_dc[uid][t]) * dt_upper_h
                # 避免除以零
                if unit.capacity_mwh > 1e-6:
                    constraints.append(
                        soc_vars_upper[uid][t + 1] == soc_vars_upper[uid][t] + energy_change_mwh / unit.capacity_mwh)
                else:  # 如果容量为0，则SOC不变
                    constraints.append(soc_vars_upper[uid][t + 1] == soc_vars_upper[uid][t])

        # 慢速储能总出力 (AC侧)，这是第一阶段决策，对所有场景都一样
        total_slow_dispatch_ac = cp.sum([
            p_discharge_upper_dc[u.id] * u.efficiency - p_charge_upper_dc[u.id] / u.efficiency
            for u in self.energy_assets
        ]) if self.energy_assets else 0

        # 【第二阶段约束】: 对每个场景s，功率平衡和快速储能物理约束都必须成立
        for s in range(num_scenarios):
            # 场景s下的快速储能总出力(AC侧)
            total_smooth_dispatch_ac_s = cp.sum([
                p_dis_smooth_dc[s, u.id] * u.efficiency - p_ch_smooth_dc[s, u.id] / u.efficiency
                for u in self.smoothing_assets
            ]) if self.smoothing_assets else 0

            # 场景s下的实际净负荷 = 预测值 + 场景误差
            net_load_actual_s = net_load_forecast_upper + scenarios[s, :]

            # 场景s下的功率平衡约束
            constraints.append(
                net_load_actual_s == total_slow_dispatch_ac + total_smooth_dispatch_ac_s + grid_exchange[s, :])

            # 场景s下的快速储能物理约束
            for unit in self.smoothing_assets:
                uid = unit.id
                constraints.append(soc_smooth[s, uid][0] == current_soc[uid])
                constraints.extend([soc_smooth[s, uid] >= unit.soc_min, soc_smooth[s, uid] <= unit.soc_max])
                constraints.append(p_dis_smooth_dc[s, uid] * unit.efficiency <= unit.power_m_w)
                constraints.append(p_ch_smooth_dc[s, uid] <= unit.power_m_w * unit.efficiency)
                for t in range(self.PH_upper):
                    energy_change_mwh = (p_ch_smooth_dc[s, uid][t] - p_dis_smooth_dc[s, uid][t]) * dt_upper_h
                    if unit.capacity_mwh > 1e-6:
                        constraints.append(
                            soc_smooth[s, uid][t + 1] == soc_smooth[s, uid][t] + energy_change_mwh / unit.capacity_mwh)
                    else:
                        constraints.append(soc_smooth[s, uid][t + 1] == soc_smooth[s, uid][t])

        # --- 5. 求解问题 ---
        problem = cp.Problem(objective, constraints)
        if self.solve_with_fallback(problem):
            # 成功求解后，我们只需要返回第一阶段的决策结果
            # 因为这才是当前需要执行的日前计划
            dispatch_ac = {unit.id: (p_discharge_upper_dc[unit.id].value * unit.efficiency) -
                                    (p_charge_upper_dc[unit.id].value / unit.efficiency)
                           for unit in self.energy_assets}

            # 注意：电网计划现在是多场景的，可以返回期望值或第一个场景的值作为参考
            grid_exchange_plan = grid_exchange.value[0, :]  # 或者 np.mean(grid_exchange.value, axis=0)

            return {"status": "optimal", "dispatch": dispatch_ac, "grid_exchange": grid_exchange_plan}
        else:
            # 求解失败，返回零计划
            failed_dispatch = {unit.id: np.zeros(self.PH_upper) for unit in self.energy_assets}
            return {"status": "failed", "dispatch": failed_dispatch, "grid_exchange": np.zeros(self.PH_upper)}

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