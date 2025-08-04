# file: PythonProject/hess_system.py

class HybridEnergyStorageSystem:
    """
    混合储能系统 (HESS) 的容器和管理器。
    根据储能单元的响应速度和应用场景进行分组。
    """

    def __init__(self):
        # 秒级响应：超容, SMES
        self.fast_response_units = {}
        # 分钟/小时级响应：飞轮, 锂电, 钠电, 铅酸
        self.medium_response_units = {}
        # 小时/天/周级响应：液流, 抽蓄, 氢能, CAES, 热储能
        self.long_duration_units = {}
        self.all_units = {}

    def add_unit(self, unit, group):
        """将一个储能单元添加到指定的组中"""
        if group == 'fast':
            self.fast_response_units[unit.id] = unit
        elif group == 'medium':
            self.medium_response_units[unit.id] = unit
        elif group == 'long':
            self.long_duration_units[unit.id] = unit

        self.all_units[unit.id] = unit

    def get_group_available_power(self, group_name):
        """获取指定组的总可用充/放电功率"""
        units_to_check = []
        if group_name == 'fast':
            units_to_check = self.fast_response_units.values()
        elif group_name == 'medium':
            units_to_check = self.medium_response_units.values()
        elif group_name == 'long':
            units_to_check = self.long_duration_units.values()

        total_charge = sum(unit.get_available_charge_power() for unit in units_to_check)
        total_discharge = sum(unit.get_available_discharge_power() for unit in units_to_check)

        return {'charge': total_charge, 'discharge': total_discharge}