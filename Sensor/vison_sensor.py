from utils.logger import get_logger
from .sensor_base import Sensor
from collections import deque
import threading
from typing import Dict, Any, Optional
import numpy as np

class VisionSensor(Sensor):
    """
    视觉传感器基类,不可实例化使用
    
    继承自Sensor类，专门用于处理图像数据的传感器。
    支持彩色图像、深度图像和点云数据的采集。
    实现了多线程数据采集和帧缓冲机制，具体采集逻辑由子类实现。
    
    ===== 使用说明 =====
    
    1. 初始化传感器：
       sensor = VisionSensor(buffer_size=1)
    
    2. 设置采集数据类型：
       sensor.set_collect_info(["color", "depth"])
    
    3. 启动数据采集：
       sensor._start_collection()
    
    4. 获取数据：
       # 非阻塞获取最新帧
       data = sensor.get_information()
       
       # 阻塞获取即时帧
       data = sensor.get_immediate_image()
       
       # 根据collect_info过滤获取数据
       filtered_data = sensor.get()
    
    5. 停止采集：
       sensor._stop_collection()
    
    6. 清理资源：
       sensor.cleanup()
    
    ===== 对外接口 =====
    - __init__(buffer_size): 初始化传感器
    - set_collect_info(collect_info): 设置采集数据类型
    - _start_collection(): 启动数据采集线程
    - _stop_collection(): 停止数据采集线程
    - get_information(): 获取最新帧全部数据（非阻塞）
    - get_immediate_image(): 获取即时帧数据（阻塞）
    - get(): 根据collect_info过滤获取数据
    - cleanup(): 清理资源
    
    ===== 内部方法（不应外部调用）=====
    - _acquire_frame(): 由子类实现的帧采集方法
    - _thread_loop(): 数据采集线程循环
    """
    def __init__(self, buffer_size: int = 1):
        super().__init__()
        self.name = "vision_sensor"
        self.type = "vision_sensor"
        self.collect_info = None
        self.logger = get_logger(self.name)
        self.frame_buffer = deque(maxlen=buffer_size)
        self._thread = None
        self._exit_event = threading.Event()
        self._keep_running = False
        self.logger.info(f"视觉传感器初始化完成，缓冲区大小: {buffer_size}")

    def _start_collection(self):
        """启动数据采集线程"""
        if self._thread and self._thread.is_alive():
            self.logger.warning("采集线程已在运行")
            return
        self._keep_running = True
        self._exit_event.clear()
        self._thread = threading.Thread(target=self._thread_loop, daemon=True)
        self._thread.start()
        self.logger.info("采集线程已启动")

    def _stop_collection(self):
        """停止数据采集线程"""
        self._keep_running = False
        self._exit_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self.logger.info("采集线程已停止")
        elif self._thread:
            # 线程存在但未启动，重置线程对象
            self._thread = None

    def _thread_loop(self):
        """线程循环：持续采集数据并存入缓冲区（内部方法，外部不应直接调用）"""
        self.logger.debug("采集线程开始运行")
        while not self._exit_event.is_set():
            try:
                frame = self._acquire_frame()
                if frame:
                    self.frame_buffer.append(frame)
            except Exception as e:
                self.logger.error(f"采集线程异常: {str(e)}")
        self.logger.debug("采集线程结束运行")

    def _acquire_frame(self) -> Optional[Dict[str, np.ndarray]]:
        """
        由子类实现：采集一帧数据（内部方法，外部不应直接调用）
        Returns:
            Dict[str, np.ndarray]: 包含图像数据的字典
            格式说明：
            - "color": BGR彩色图像 (H, W, 3)，与OpenCV格式保持一致
            - "depth": 深度图像 (H, W)，单位为毫米(mm)
        """
        raise NotImplementedError("子类必须实现 _acquire_frame() 方法")

    def get_information(self) -> Optional[Dict[str, np.ndarray]]:
        """
        非阻塞获取队列最新帧的全部原始数据
        Returns:
            Optional[Dict[str, np.ndarray]]: 最新帧全部数据
        """
        if not self.frame_buffer:
            self.logger.debug("缓冲区为空，无可用数据")
            return None
        return self.frame_buffer[-1]

    def get_immediate_image(self) -> Optional[Dict[str, np.ndarray]]:
        """
        阻塞采集一帧最新数据（直接调用底层采集）
        Returns:
            Optional[Dict[str, np.ndarray]]: 采集到的最新一帧全部数据
        """
        self.logger.debug("执行即时图像采集")
        return self._acquire_frame()

    def cleanup(self):
        """
        清理传感器资源（线程清理，子类如有额外资源，必须重写）
        """
        self.logger.info("开始清理视觉传感器资源")
        self._stop_collection()

    def __enter__(self):
        """
        上下文管理器入口
        Returns:
            self: 返回传感器实例
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口，确保资源被正确释放
        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪信息
        """
        if exc_type is not None:
            self.logger.error(f"传感器使用过程中发生异常: {exc_type.__name__}: {exc_val}")
        self.cleanup()

