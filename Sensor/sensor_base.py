from typing import List, Dict, Any, Optional
from utils.logger import get_logger
from abc import ABC, abstractmethod

class Sensor(ABC):
    """
    基础传感器类
    
    所有传感器的基类，定义了传感器的基本接口和通用功能。
    
    ===== 使用说明 =====
    
    1. 初始化传感器：
       sensor = ConcreteSensor()  # 具体传感器类
    
    2. 设置采集数据类型：
       sensor.set_collect_info(["color", "depth"])
    
    3. 获取数据：
       # 获取全部原始数据
       data = sensor.get_information()
       
       # 根据collect_info过滤获取数据
       filtered_data = sensor.get()
    
    4. 清理资源：
       sensor.cleanup()
    
    ===== 对外接口 =====
    - __init__(): 初始化传感器
    - set_collect_info(collect_info): 设置采集数据类型
    - get_information(): 获取传感器全部原始信息（抽象方法）
    - get(): 根据collect_info过滤获取数据
    - cleanup(): 清理资源
    
    ===== 内部方法（不应外部调用）=====
    - 所有以_开头的方法都是内部方法
    """
    def __init__(self):
        """初始化基础传感器"""
        self.name = "sensor"
        self.type = "sensor"
        self.collect_info: Optional[List[str]] = None
        self.logger = get_logger(self.name)
        self.logger.debug("基础传感器初始化完成")
    
    def set_collect_info(self, collect_info: List[str]) -> None:
        """
        设置需要采集的数据类型
        Args:
            collect_info: 需要采集的数据类型列表，如["color", "depth"]
        """
        self.collect_info = collect_info
        self.logger.info(f"设置采集数据类型: {collect_info}")
    
    def get(self) -> Optional[Dict[str, Any]]:
        """
        获取传感器数据（根据 collect_info 过滤）
        Returns:
            Dict[str, Any]: 包含指定类型数据的字典，如果collect_info未设置则返回全部原始数据
        """
        info = self.get_information()
        if info is None:
            self.logger.warning("get_information() 返回 None")
            return None
        if not self.collect_info:
            return info
        result = {}
        for key in self.collect_info:
            value = info.get(key)
            if value is None:
                self.logger.error(f"{key} 信息为 None 或未包含在 info 中")
                raise ValueError(f"{key} 数据为 None 或未找到")
            result[key] = value
        return result

    @abstractmethod
    def get_information(self) -> Optional[Dict[str, Any]]:
        """
        获取传感器全部原始信息（抽象方法，子类必须实现）
        Returns:
            Dict[str, Any]: 传感器原始数据字典
        """
        pass

    def cleanup(self):
        """
        清理传感器资源（基类只做空实现，子类如有资源需重写）
        """
        self.logger.debug("基础传感器清理完成")

    def __repr__(self) -> str:
        """返回传感器的字符串表示"""
        return f"Base Sensor, can't be used directly\nname: {self.name}\ntype: {self.type}"