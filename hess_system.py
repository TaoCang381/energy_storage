# file: hess_system.py (统一接口修改版 V1.0)
# 备注：此类作为所有标准化储能模型的容器和管理器。
#       修改了方法以匹配BaseStorageModel的接口规范。

class HybridEnergyStorageSystem:
    """
    混合储能系统 (HESS) 的容器和管理器。
    负责统一管理所有继承自BaseStorageModel的储能单元。
    """

    def __init__(self, dt_s):
        """
        初始化HESS系统。
        :param dt_s: 系统仿真的基础时间步长（秒）。
        """
        self.dt_s = dt_s
        self.all_units = {}

    def add_unit(self, unit):
        """
        将一个储能单元添加到系统中。
        我们期望传入的unit是BaseStorageModel的子类。
        """
        if unit.id in self.all_units:
            raise ValueError(f"ID为 '{unit.id}' 的储能单元已存在。")
        self.all_units[unit.id] = unit
        print(f"成功添加储能单元: {unit.id} (类型: {type(unit).__name__})")

    def get_all_soc(self):
        """
        获取所有储能单元的当前SOC。

        返回:
        一个字典，key是单元ID，value是该单元的SOC。
        """
        # --- 修改区域: 调用每个单元的get_soc()方法 ---
        return {unit_id: unit_obj.get_soc() for unit_id, unit_obj in self.all_units.items()}

    def update_all_states(self, dispatch_signals):
        """
        【新方法】根据当前时间步的调度信号，更新所有储能单元的状态。

        参数:
        dispatch_signals (dict): 一个字典，key是单元ID，value是该单元的功率指令(W)。
                                   正数表示放电，负数表示充电。
        """
        # --- 修改区域: 适配新的update_state接口 ---
        for unit_id, unit_obj in self.all_units.items():
            # 从调度信号字典中获取对应ID的功率指令，如果找不到则默认为0
            power_w = dispatch_signals.get(unit_id, 0)

            # 调用每个储能单元自己的update_state方法进行更新
            # 注意：新的update_state方法不再需要dt_s作为参数，因为它在初始化时已被存为内部属性
            unit_obj.update_state(power_w)