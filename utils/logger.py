import logging
import os
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置颜色
    }
    
    def format(self, record):
        # 获取原始格式化的消息
        formatted = super().format(record)
        
        # 添加颜色
        if record.levelname in self.COLORS:
            formatted = f"{self.COLORS[record.levelname]}{formatted}{self.COLORS['RESET']}"
        
        return formatted


def setup_logger(log_level=logging.INFO, log_file=None, enable_color=True):
    """
    设置日志系统
    
    Args:
        log_level: 日志级别
        log_file: 日志文件名，如果为None则自动根据时间生成文件名
        enable_color: 是否启用颜色输出
    """
    # 如果未提供日志文件名，则根据当前时间生成
    if log_file is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"robot_system_{current_time}.log"
    # 创建logs目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 日志文件路径
    log_path = os.path.join(log_dir, log_file)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 根据是否启用颜色选择格式化器
    if enable_color:
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(colored_formatter)
    else:
        console_handler.setFormatter(formatter)
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 记录系统启动信息
    root_logger.info("=" * 50)
    root_logger.info("机器人系统启动")
    root_logger.info(f"日志级别: {logging.getLevelName(log_level)}")
    root_logger.info(f"日志文件: {log_path}")
    root_logger.info(f"颜色输出: {'启用' if enable_color else '禁用'}")
    root_logger.info("=" * 50)


def get_logger(name):
    """
    获取指定名称的logger
    
    Args:
        name: logger名称
        
    Returns:
        logging.Logger: logger实例
    """
    return logging.getLogger(name)


# 使用示例
if __name__ == "__main__":
    # 设置日志系统（启用颜色）
    setup_logger(log_level=logging.DEBUG, enable_color=True)
    
    # 获取不同模块的logger
    sensor_logger = get_logger("Sensor.RGB_Camera")
    robot_logger = get_logger("Robot.Controller")
    module_logger = get_logger("Module.HandDetection")
    
    # 记录不同级别的日志
    sensor_logger.info("RGB相机初始化成功")
    robot_logger.warning("机械臂连接超时")
    module_logger.error("手势检测模块加载失败")
    sensor_logger.debug("传感器数据: [1, 2, 3, 4]") 