# file: PythonProject/base_storage_model.py

class EnergyStorageUnit:
    """
    所有储能单元的抽象基类 (Abstract Base Class)。
    它定义了所有储能模型必须实现的通用接口，
    确保EMS可以用统一的方式与它们交互。
    """

    # ========================== 错误修正区 (开始) ==========================
    # 升级__init__方法，使其可以接收所有储能共有的初始状态参数
    def __init__(self, ess_id, initial_soc=0.5, initial_soh=1.0):
        # ========================== 错误修正区 (结束) ==========================
        self.id = ess_id
        self.state = 'idle'
        self.soc = initial_soc  # 使用统一的soc属性
        self.soh = initial_soh  # 使用统一的soh属性

        # 历史记录（可选，具体模型可自行扩展）
        self.time_history = []
        self.power_history = []
        self.soc_history = []

    def charge(self, power, time_s):
        """按指定功率和时间充电"""
        raise NotImplementedError(f"{self.id}: 每个储能模型必须实现charge方法")

    def discharge(self, power, time_s):
        """按指定功率和时间放电"""
        raise NotImplementedError(f"{self.id}: 每个储能模型必须实现discharge方法")

    def get_soc(self):
        """获取当前荷电状态 (State of Charge)"""
        # 提供一个默认实现，具体模型可以重写此方法
        return self.soc

    def get_soh(self):
        """获取当前健康状态 (State of Health)"""
        return self.soh

    def get_available_charge_power(self):
        """获取当前可用的最大充电功率"""
        raise NotImplementedError(f"{self.id}: 每个储能模型必须实现get_available_charge_power方法")

    def get_available_discharge_power(self):
        """获取当前可用的最大放电功率"""
        raise NotImplementedError(f"{self.id}: 每个储能模型必须实现get_available_discharge_power方法")

    def idle_loss(self, time_s):
        """计算闲置损失"""
        raise NotImplementedError(f"{self.id}: 每个储能模型必须实现idle_loss方法")

    def _record_history(self, time_delta, power, soc):
        """内部方法：记录历史数据"""
        current_time = self.time_history[-1] + time_delta if self.time_history else time_delta
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.soc_history.append(soc)