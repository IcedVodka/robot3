"""
路径管理工具模块
用于处理项目根目录的导入问题
"""
import sys
import os
from pathlib import Path


def add_project_root_to_path():
    """
    将项目根目录添加到 Python 路径中
    这样可以在任何子目录中导入根目录的模块
    """
    # 获取当前文件的目录
    current_dir = Path(__file__).parent
    
    # 获取项目根目录（robot3目录）
    project_root = current_dir.parent
    
    # 将项目根目录添加到 sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def get_project_root():
    """
    获取项目根目录路径
    """
    return Path(__file__).parent.parent


# 自动添加项目根目录到路径
add_project_root_to_path()
