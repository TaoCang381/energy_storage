# file: base_storage_model.py
# 备注：这是一个所有储能模型的“父类”或“基类”。
#       它负责处理所有储能单元共有的属性和方法。

class BaseStorageModel:
    def __init__(self, id, dt_s):
        """
        所有储能模型的通用构造函数。

        参数:
        id (str): 储能单元的唯一标识符 (例如 'fw', 'ees', 'phs').
        dt_s (int): 仿真步长 (秒).
        """
        if not isinstance(id, str) or not id:
            raise ValueError("储能单元必须有一个有效的字符串ID。")

        self.id = id
        self.dt_s = dt_s

        # 初始化一些所有储能都应有的通用状态变量
        self.soc = 0.5  # 初始SOC
        self.power_m_w = 0.0  # 额定功率
        self.capacity_mwh = 0.0  # 额定容量
        self.efficiency = 1.0  # 默认效率
        self.soc_min = 0.0
        self.soc_max = 1.0
        self.om_cost_per_mwh = 0.0  # 运维成本

        # 您可以在这里继续添加其他所有储能都共有的参数...

    def update_state(self, dispatch_power):
        """
        一个通用的状态更新方法的“占位符”。
        每个具体的储能模型都应该重写(override)这个方法，实现自己的SOC更新逻辑。
        """
        raise NotImplementedError("每个储能子类都必须实现自己的 update_state 方法。")

    def get_soc(self):
        """
        获取当前SOC.
        """
        return self.soc