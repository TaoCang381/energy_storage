# file: hess_system.py

class HybridEnergyStorageSystem:
    """
    混合储能系统 (HESS) 的容器和管理器。
    """

    def __init__(self, dt_lower_s):
        """
        初始化HESS系统。
        :param dt_lower_s: 系统仿真的基础时间步长（秒），通常是下层MPC的步长。
        """
        self.dt = dt_lower_s
        self.all_units = {}

    def add_unit(self, unit):
        """将一个储能单元添加到系统中"""
        self.all_units[unit.id] = unit

    def get_all_soc(self):
        """获取所有储能单元的当前SOC。"""
        soc_dict = {}
        for unit_id, unit_obj in self.all_units.items():
            soc_dict[unit_id] = unit_obj.SOC
        return soc_dict

    def update_states(self, dispatch_signals, duration_s):
        """
        根据调度信号更新所有储能单元的状态。
        :param dispatch_signals: 一个字典，key是单元ID，value是该单元的功率(W)。
        :param duration_s: 本次状态更新持续的时间（秒）。
        """
        for unit_id, unit_obj in self.all_units.items():
            power_w = dispatch_signals.get(unit_id, 0)  # 如果没有信号则认为功率为0
            unit_obj.update_state(power_w, duration_s)