import cv2
import numpy as np
from .vison_sensor import VisionSensor
from utils.logger import get_logger
import time

class RgbCameraSensor(VisionSensor):
    """
    RGB摄像头传感器类
    
    基于OpenCV的RGB摄像头传感器实现，支持多线程数据采集，
    提供彩色图像的实时获取功能。
    
    ===== 使用说明 =====
    
    1. 初始化传感器：
       sensor = RgbCameraSensor(name="my_camera")
    
    2. 设置相机参数：
       sensor.set_up(camera_id=0)
    
    3. 设置采集数据类型：
       sensor.set_collect_info(["color"])
    
    4. 获取数据：
       # 非阻塞获取最新帧
       data = sensor.get_information()
       
       # 阻塞获取即时帧
       data = sensor.get_immediate_image()
       
       # 根据collect_info过滤获取数据
       filtered_data = sensor.get()
    
    5. 清理资源：
       sensor.cleanup()
    
    ===== 对外接口 =====
    - __init__(name): 初始化传感器
    - set_up(camera_id): 设置相机参数
    - set_collect_info(collect_info): 设置采集数据类型
    - get_information(): 获取最新帧全部数据（非阻塞）
    - get_immediate_image(): 获取即时帧数据（阻塞）
    - get(): 根据collect_info过滤获取数据
    - cleanup(): 清理资源
    
    ===== 数据格式 =====
    返回数据为字典格式：
    {
        "color": np.ndarray  # BGR彩色图像 (H, W, 3)，与OpenCV格式保持一致
    }
    """
    def __init__(self, name="rgb_camera"):
        super().__init__(buffer_size=1)
        self.camera_id = None
        self.cap = None
        self.name = name
        self.type = "rgb_camera"
        self.logger = get_logger(self.name)
        self.logger.info(f"初始化RGB摄像头传感器: {self.name}")

    def set_up(self, camera_id=0):
        """
        设置RGB摄像头
        Args:
            camera_id (int): 摄像头ID，默认为0
        Raises:
            RuntimeError: 当无法打开摄像头时抛出
        """
        self.camera_id = camera_id
        self.set_collect_info(["color"])
        self.logger.info(f"开始设置RGB摄像头，ID: {self.camera_id}")
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开摄像头 {self.camera_id}")
                raise RuntimeError(f"无法打开摄像头 {self.camera_id}")
            
            self._start_collection()
            self.logger.info(f"RGB摄像头启动成功: {self.name}")
            time.sleep(1) # 等待摄像头初始化
            
        except Exception as e:
            self.logger.error(f"RGB摄像头初始化失败: {str(e)}")
            self.cleanup()
            raise RuntimeError(f"Failed to initialize RGB camera: {str(e)}")

    def _acquire_frame(self):
        """
        采集一帧彩色图像（内部方法，外部不应直接调用）
        Returns:
            dict: {"color": BGR彩色图像}
        """
        if self.cap is None or not self.cap.isOpened():
            self.logger.error("摄像头未初始化")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("读取摄像头帧失败")
                return None
            return {"color": frame}
            
        except Exception as e:
            self.logger.error(f"帧采集失败: {str(e)}")
            return None

    def cleanup(self):
        """清理资源，停止采集线程和摄像头"""
        try:
            self.logger.info("开始清理RGB摄像头传感器资源")
            # 先停止采集线程
            self._stop_collection()
            # 再释放摄像头资源
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.logger.info("摄像头资源已释放")
        except Exception as e:
            self.logger.error(f"清理过程中发生错误: {str(e)}")
    
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.cleanup()

    def __repr__(self) -> str:
        return (f"RgbCameraSensor\n"
                f"name: {self.name}\n"
                f"type: {self.type}\n"
                f"camera_id: {self.camera_id}\n"
                f"cap: {'initialized' if self.cap else 'None'}")
