#!/usr/bin/env python3
"""
简单的HDF5文件读取脚本
读取最新的hdf5文件夹，如果有label的hdf5就读取这个，打印元信息和label信息，没有就读取hdf5，打印元信息
"""

import os
import glob
import h5py
import json
import numpy as np
from pathlib import Path


def get_latest_hdf5_folder(save_dir="./save/"):
    """获取最新的HDF5文件夹"""
    # 查找所有以grasp_开头的文件夹
    pattern = os.path.join(save_dir, "grasp_*")
    folders = glob.glob(pattern)
    
    if not folders:
        print(f"在 {save_dir} 中没有找到HDF5文件夹")
        return None
    
    # 按修改时间排序，获取最新的
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder


def print_hdf5_info(hdf5_path):
    """打印HDF5文件的基本信息"""
    print(f"\n=== HDF5文件信息: {os.path.basename(hdf5_path)} ===")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 打印文件属性
        print("\n--- 文件属性 ---")
        for key, value in f.attrs.items():
            if key in ['realsense1', 'realsense2']:
                try:
                    # 解析JSON字符串
                    parsed_value = json.loads(value)
                    print(f"{key}: {json.dumps(parsed_value, indent=2)}")
                except:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
        
        # 打印数据集信息
        print("\n--- 数据集信息 ---")
        def print_group_info(group, prefix=""):
            for key, value in group.items():
                if isinstance(value, h5py.Group):
                    print(f"{prefix}{key}/ (组)")
                    print_group_info(value, prefix + "  ")
                else:
                    print(f"{prefix}{key}: {value.shape} {value.dtype}")
        
        print_group_info(f)
        
        # 打印数据统计信息
        print("\n--- 数据统计 ---")
        if 'robot' in f:
            if 'joint' in f['robot']:
                joints = f['robot/joint'][:]
                print(f"关节数据: 形状 {joints.shape}, 范围 [{joints.min():.3f}, {joints.max():.3f}]")
            
            if 'pose' in f['robot']:
                poses = f['robot/pose'][:]
                print(f"位姿数据: 形状 {poses.shape}, 位置范围 [{poses[:, :3].min():.3f}, {poses[:, :3].max():.3f}]")
            
            if 'gripper_state' in f['robot']:
                gripper = f['robot/gripper_state'][:]
                print(f"手爪状态: 形状 {gripper.shape}, 唯一值 {np.unique(gripper)}")
        
        if 'timestamps' in f and 'sample_time' in f['timestamps']:
            timestamps = f['timestamps/sample_time'][:]
            print(f"时间戳: 形状 {timestamps.shape}, 范围 [{timestamps.min():.3f}, {timestamps.max():.3f}]")
            print(f"数据长度: {len(timestamps)} 帧")


def print_label_info(hdf5_path):
    """打印HDF5文件中的label信息"""
    print(f"\n=== Label信息 ===")
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'label' not in f:
            print("没有找到label数据")
            return
        
        label_group = f['label']
        
        for camera_name in ['realsense1', 'realsense2']:
            if camera_name in label_group:
                print(f"\n--- {camera_name} 标注信息 ---")
                camera_group = label_group[camera_name]
                
                # 打印heatmap信息
                if 'heatmap' in camera_group:
                    heatmap = camera_group['heatmap'][:]
                    print(f"热力图: 形状 {heatmap.shape}, 范围 [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                
                # 打印label_map信息
                if 'label_map' in camera_group:
                    label_map = camera_group['label_map'][:]
                    unique_labels = np.unique(label_map)
                    print(f"标签图: 形状 {label_map.shape}, 唯一标签 {unique_labels}")
                    
                    # 统计每个标签的像素数量
                    for label in unique_labels:
                        if label > 0:  # 跳过背景标签0
                            count = np.sum(label_map == label)
                            print(f"  标签 {label}: {count} 像素")
                
                # 打印annotations信息
                if 'annotations' in camera_group:
                    annotations = camera_group['annotations'][:]
                    print(f"标注点: {len(annotations)} 个")
                    if len(annotations) > 0:
                        print("  标注点详情:")
                        for i, ann in enumerate(annotations):
                            print(f"    点{i+1}: 位置({ann[0]}, {ann[1]}), 标签{ann[2]}")


def main():
    """主函数"""
    print("=== HDF5文件读取工具 ===")
    
    # 获取最新文件夹
    latest_folder = get_latest_hdf5_folder()
    
    if latest_folder is None:
        return
    
    print(f"找到最新HDF5文件夹: {latest_folder}")
    
    # 查找HDF5文件
    hdf5_files = glob.glob(os.path.join(latest_folder, "*.h5"))
    
    if not hdf5_files:
        print("未找到HDF5文件")
        return
    
    # 优先查找带label的HDF5文件
    labeled_hdf5 = None
    regular_hdf5 = None
    
    for hdf5_file in hdf5_files:
        if '_labeled.h5' in hdf5_file:
            labeled_hdf5 = hdf5_file
        else:
            regular_hdf5 = hdf5_file
    
    # 选择要读取的文件
    if labeled_hdf5:
        print(f"\n找到带标注的HDF5文件: {os.path.basename(labeled_hdf5)}")
        hdf5_to_read = labeled_hdf5
    else:
        print(f"\n找到普通HDF5文件: {os.path.basename(regular_hdf5)}")
        hdf5_to_read = regular_hdf5
    
    # 读取并打印信息
    print_hdf5_info(hdf5_to_read)
    
    # 如果是带标注的文件，打印label信息
    if labeled_hdf5:
        print_label_info(hdf5_to_read)
    
    print(f"\n=== 读取完成 ===")


if __name__ == "__main__":
    main()
