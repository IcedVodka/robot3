import pyrealsense2 as rs
import numpy as np
from collections import deque
import threading
from utils.logger import get_logger
from .vison_sensor import VisionSensor
import time

def print_realsense_devices():
    """
    打印所有连接的RealSense深度相机数量及序列号
    """    
    try:
        context = rs.context()
        devices = list(context.query_devices())
        
        if not devices:
            print("未找到任何RealSense设备")
            return
        
        print(f"找到 {len(devices)} 个RealSense深度相机:")
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"  序列号: {serial}")
        
    except Exception as e:
        print(f"获取RealSense设备列表失败: {str(e)}") 


class RealsenseSensor(VisionSensor):

    """
    RealSense相机传感器类
    
    基于Intel RealSense相机的传感器实现，支持多线程数据采集，
    提供彩色图像和深度图像的实时获取功能。
    
    ===== 使用说明 =====
    
    1. 初始化传感器：
       sensor = RealsenseSensor("camera_name")
    
    2. 设置相机参数：
       sensor.set_up(camera_serial="123456789", is_depth=True)
    
    3. 设置采集数据类型：
       sensor.set_collect_info(["color", "depth"])
    
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
    - set_up(camera_serial, is_depth): 设置相机参数
    - set_collect_info(collect_info): 设置采集数据类型
    - get_information(): 获取最新帧全部数据（非阻塞）
    - get_immediate_image(): 获取即时帧数据（阻塞）
    - get(): 根据collect_info过滤获取数据
    - cleanup(): 清理资源
    
    ===== 数据格式 =====
    返回数据为字典格式：
    {
        "color": np.ndarray,  # BGR彩色图像 (H, W, 3)，与OpenCV格式保持一致
        "depth": np.ndarray   # 深度图像 (H, W)，单位为毫米(mm) - 仅在is_depth=True时提供
    }
    """
    def __init__(self, name: str):
        super().__init__(buffer_size=1)
        self.name = name
        self.type = "realsense"
        self.logger = get_logger(self.name)
        self.pipeline = None
        self.config = None        
        self.is_depth = False
        self._pipeline_started = False
        self.resolution = [640, 480]  # 默认分辨率
        self.logger.info(f"初始化RealSense传感器: {name}")

    def set_up(self, camera_serial: str, is_depth: bool = True, resolution: list = None):
        """
        设置RealSense相机
        Args:
            camera_serial: 相机序列号
            is_depth: 是否启用深度流，默认False
            resolution: 分辨率，包含两个值的list，默认[640, 480]，会保存到self.resolution
        Raises:
            RuntimeError: 当找不到设备或启动失败时抛出
        """
        self.is_depth = is_depth
        self.set_collect_info(["color", "depth"])
        if resolution is not None:
            if not (isinstance(resolution, list) and len(resolution) == 2):
                raise ValueError("resolution参数必须为包含两个值的list，如[640, 480]")
            self.resolution = resolution
        width, height = self.resolution[0], self.resolution[1] 
        self.logger.info(f"开始设置相机，序列号: {camera_serial}, 深度模式: {is_depth}, 分辨率: {width}x{height}")

        try:
            # 初始化RealSense上下文并检查连接的设备
            self.context = rs.context()
            self.devices = list(self.context.query_devices())
            if not self.devices:
                self.logger.error("未找到RealSense设备")
                raise RuntimeError("No RealSense devices found")

            self.logger.info(f"找到 {len(self.devices)} 个RealSense设备")

            # 根据序列号查找设备
            device_idx = self._find_device_by_serial(self.devices, camera_serial)
            if device_idx is None:
                self.logger.error(f"未找到序列号为 {camera_serial} 的相机")
                raise RuntimeError(f"Could not find camera with serial number {camera_serial}")

            self.logger.info(f"找到目标设备，索引: {device_idx}")

            # 配置管道
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            # 启用指定设备
            self.config.enable_device(camera_serial)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
            if is_depth:
                self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
                self.logger.info("已启用深度流")

            self.pipeline.start(self.config)
            self._pipeline_started = True
            self._start_collection()
            self.logger.info(f"相机启动成功: {self.name} (SN: {camera_serial})")
            time.sleep(1) # 等待摄像头初始化

        except Exception as e:
            self.logger.error(f"相机初始化失败: {str(e)}")
            self.cleanup()
            raise RuntimeError(f"Failed to initialize camera: {str(e)}")

    def _acquire_frame(self):
        """
        采集一帧彩色/深度图像（内部方法，外部不应直接调用）
        Returns:
            dict: {"color": BGR彩色图像, "depth": 深度图像(毫米单位)}
        """
        if not self.pipeline:
            self.logger.error("Pipeline 未初始化")
            return None
            
        try:
            frames = self.pipeline.wait_for_frames(5000)
            result = {}
            
            if not self.collect_info:
                return None
                
            # 获取彩色图像
            if "color" in self.collect_info:
                color_frame = frames.get_color_frame()
                if color_frame:
                    # RealSense默认输出BGR格式，直接使用，与OpenCV保持一致
                    color_image = np.asanyarray(color_frame.get_data())
                    result["color"] = color_image
                else:
                    self.logger.warning("未获取到彩色帧")
                    
            # 获取深度图像
            if self.is_depth and "depth" in self.collect_info:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    # 深度图像为16位整数，单位为毫米(mm)
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result["depth"] = depth_image
                else:
                    self.logger.warning("未获取到深度帧")
                    
            return result if result else None
            
        except Exception as e:
            self.logger.error(f"帧采集失败: {str(e)}")
            return None

    def cleanup(self):
        """清理资源，停止采集线程和pipeline"""
        try:
            self.logger.info("开始清理RealSense传感器资源")
            # 先停止采集线程
            self._stop_collection()
            # 再停止pipeline
            if hasattr(self, 'pipeline') and self.pipeline and self._pipeline_started:
                try:
                    self.pipeline.stop()
                    self._pipeline_started = False
                    self.logger.info("Pipeline已停止")
                except Exception as e:
                    self.logger.warning(f"Pipeline停止失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"清理过程中发生错误: {str(e)}")
    
    def get_intrinsics(self):
        """
        获取当前分辨率下的彩色图像和深度图像内参。
        Returns:
            dict: { 'color': {...}, 'depth': {...} }
        """
        if not self.pipeline or not self._pipeline_started:
            self.logger.error("Pipeline未启动，无法获取内参")
            return None
        try:
            # 获取一次帧，确保profile可用
            frames = self.pipeline.wait_for_frames(2000)
            color_intr = None
            depth_intr = None
            if frames:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.is_depth else None
                if color_frame:
                    color_profile = color_frame.get_profile()
                    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                    color_intr = {
                        'ppx': color_intrinsics.ppx,
                        'ppy': color_intrinsics.ppy,
                        'fx': color_intrinsics.fx,
                        'fy': color_intrinsics.fy
                    }
                if depth_frame:
                    depth_profile = depth_frame.get_profile()
                    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
                    depth_intr = {
                        'ppx': depth_intrinsics.ppx,
                        'ppy': depth_intrinsics.ppy,
                        'fx': depth_intrinsics.fx,
                        'fy': depth_intrinsics.fy
                    }
            return {'color': color_intr, 'depth': depth_intr}
        except Exception as e:
            self.logger.error(f"获取内参失败: {str(e)}")
            return None
    
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.cleanup()

    def __repr__(self) -> str:
        return (f"RealsenseSensor\n"
                f"name: {self.name}\n"
                f"type: {self.type}\n"
                f"is_depth: {self.is_depth}\n"
                f"pipeline: {'initialized' if self.pipeline else 'None'}")

    def _find_device_by_serial(self, devices, serial):
        """
        根据序列号查找设备索引（内部方法，外部不应直接调用）
        Args:
            devices: RealSense设备列表
            serial: 目标设备的序列号
        Returns:
            int: 设备在列表中的索引，如果未找到返回None
        """
        for i, dev in enumerate(devices):
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return i
        return None




