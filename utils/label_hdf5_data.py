#!/usr/bin/env python3
"""
数据标注脚本，读取最新HDF5文件夹中的realsense1和realsense2彩色视频，进行交互式标注
"""

import os
import cv2
import glob
import numpy as np
import h5py
from pathlib import Path


def get_latest_hdf5_folder(save_dir="./save/"):
    """获取最新的HDF5文件夹"""
    # 查找所有以grasp_开头的文件夹
    pattern = os.path.join(save_dir, "task1_*")
    folders = glob.glob(pattern)
    
    if not folders:
        print(f"在 {save_dir} 中没有找到HDF5文件夹")
        return None
    
    # 按修改时间排序，获取最新的
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder


class DataAnnotator:
    def __init__(self, frame, window_name="Data Annotation"):
        self.frame = frame.copy()
        self.height, self.width = frame.shape[:2]
        self.window_name = window_name
        
        # 初始化两个数组
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.label_map = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 存储标注点
        self.annotations = []  # [(x, y, label), ...]
        
        # 显示参数
        self.radius = 50
        self.heatmap_value = 100
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取用户输入的标签
            print(f"\n点击位置: ({x}, {y})")
            while True:
                try:
                    label = int(input("请输入标签 (0-5): "))
                    if 0 <= label <= 5:
                        break
                    else:
                        print("标签必须在0-5之间，请重新输入")
                except ValueError:
                    print("请输入有效的数字")
            
            # 记录标注
            self.annotations.append((x, y, label))
            print(f"已标注点 ({x}, {y}) 为标签 {label}")
            
            # 更新热力图和标签图
            self.update_maps(x, y, label)
            
            # 重新显示
            self.show_annotation(self.window_name)
    
    def update_maps(self, x, y, label):
        """更新热力图和标签图"""
        # 创建高斯核 - 修复：确保中心值为1
        kernel_size = self.radius * 2 + 1
        kernel = cv2.getGaussianKernel(kernel_size, self.radius / 3)
        kernel_2d = kernel @ kernel.T
        
        # 确保高斯核中心值为1（cv2.getGaussianKernel已经是归一化的）
        # 直接使用，因为中心值已经是1
        
        # 计算影响区域
        y1 = max(0, y - self.radius)
        y2 = min(self.height, y + self.radius + 1)
        x1 = max(0, x - self.radius)
        x2 = min(self.width, x + self.radius + 1)
        
        # 计算核的裁剪区域
        ky1 = self.radius - (y - y1)
        ky2 = self.radius + (y2 - y)
        kx1 = self.radius - (x - x1)
        kx2 = self.radius + (x2 - x)
        
        # 更新热力图 - 修复：直接使用高斯核，中心值已经是1
        heatmap_patch = self.heatmap[y1:y2, x1:x2]
        kernel_patch = kernel_2d[ky1:ky2, kx1:kx2]
        self.heatmap[y1:y2, x1:x2] = np.maximum(heatmap_patch, kernel_patch * self.heatmap_value)       

        
        # 更新标签图
        mask = kernel_patch > 0.0001  # 阈值过滤
        self.label_map[y1:y2, x1:x2][mask] = label
    
    def show_annotation(self, window_name):
        """显示标注结果 - 实时显示热力图叠加在原始图像上"""
        # 原始帧
        original = self.frame.copy()
        
        # 热力图（转换为彩色显示）
        heatmap_max = self.heatmap.max()
        if heatmap_max > 0:
            heatmap_normalized = (self.heatmap * 255 / heatmap_max).astype(np.uint8)
        else:
            heatmap_normalized = np.zeros_like(self.heatmap, dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # 原始帧叠加热力图
        overlay = cv2.addWeighted(original, 0.7, heatmap_colored, 0.3, 0)
        
        # 添加标注点标记
        for i, (x, y, label) in enumerate(self.annotations):
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)  # 绿色圆点
            cv2.putText(overlay, str(label), (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 添加说明文字
        cv2.putText(overlay, f"Annotations: {len(self.annotations)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, "Click to annotate, 'q' to quit, 'r' to reset, 's' to save", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, overlay)
    
    def show_final_results(self, window_name):
        """显示最终结果 - 2x2布局"""
        h, w = self.height, self.width
        
        # 原始帧
        original = self.frame.copy()
        
        # 热力图（转换为彩色显示）
        heatmap_max = self.heatmap.max()
        if heatmap_max > 0:
            heatmap_normalized = (self.heatmap * 255 / heatmap_max).astype(np.uint8)
        else:
            heatmap_normalized = np.zeros_like(self.heatmap, dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # 标签图（转换为彩色显示）
        label_colored = cv2.applyColorMap(
            (self.label_map * 255 / 5).astype(np.uint8), 
            cv2.COLORMAP_HSV
        )
        
        # 原始帧叠加热力图
        overlay = cv2.addWeighted(original, 0.7, heatmap_colored, 0.3, 0)
        
        # 创建2x2布局
        if h > w:
            # 垂直布局
            top_row = np.hstack([original, heatmap_colored])
            bottom_row = np.hstack([label_colored, overlay])
            display = np.vstack([top_row, bottom_row])
        else:
            # 水平布局
            left_col = np.vstack([original, heatmap_colored])
            right_col = np.vstack([label_colored, overlay])
            display = np.hstack([left_col, right_col])
        
        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # 在图像上添加标题
        cv2.putText(display, "Original", (10, 20), font, font_scale, color, thickness)
        cv2.putText(display, "Heatmap", (w + 10, 20), font, font_scale, color, thickness)
        cv2.putText(display, "Labels", (10, h + 20), font, font_scale, color, thickness)
        cv2.putText(display, "Overlay", (w + 10, h + 20), font, font_scale, color, thickness)
        
        cv2.imshow(window_name, display)
        return overlay
    
    def get_annotation_data(self):
        """获取标注数据"""
        return {
            'heatmap': self.heatmap,
            'label_map': self.label_map,
            'annotations': self.annotations
        }
    
    def save_annotation_image(self, save_path, video_name):
        """保存标注图片"""
        # 生成并保存热力图叠加原图
        original = self.frame.copy()
        heatmap_max = self.heatmap.max()
        if heatmap_max > 0:
            heatmap_normalized = (self.heatmap * 255 / heatmap_max).astype(np.uint8)
        else:
            heatmap_normalized = np.zeros_like(self.heatmap, dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.7, heatmap_colored, 0.3, 0)
        
        # 保存叠加图像
        image_path = os.path.join(save_path, f"{video_name}_heatmap_overlay.jpg")
        cv2.imwrite(image_path, overlay)
        print(f"标注图片已保存: {image_path}")
        return image_path


def load_first_frame(video_path):
    """读取视频文件的第一帧"""
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None
    
    # 读取第一帧
    ret, frame = cap.read()
    
    if not ret:
        print(f"无法读取视频第一帧: {video_path}")
        cap.release()
        return None
    
    cap.release()
    return frame


def create_labeled_hdf5(original_hdf5_path, annotation_data_realsense1, annotation_data_realsense2):
    """创建带标注的HDF5文件"""
    # 构建新的HDF5文件路径
    base_name = os.path.splitext(original_hdf5_path)[0]
    labeled_hdf5_path = f"{base_name}_labeled.h5"
    
    print(f"创建标注HDF5文件: {labeled_hdf5_path}")
    
    # 复制原始HDF5文件
    with h5py.File(original_hdf5_path, 'r') as src:
        with h5py.File(labeled_hdf5_path, 'w') as dst:
            # 复制所有原始数据
            def copy_group(src_group, dst_group):
                for key, value in src_group.items():
                    if isinstance(value, h5py.Group):
                        new_group = dst_group.create_group(key)
                        copy_group(value, new_group)
                    else:
                        dst_group.copy(value, key)
            
            copy_group(src, dst)
            
            # 创建label组
            label_group = dst.create_group('label')
            
            # 为realsense1创建标注数据
            realsense1_group = label_group.create_group('realsense1')
            realsense1_group.create_dataset('heatmap', data=annotation_data_realsense1['heatmap'])
            realsense1_group.create_dataset('label_map', data=annotation_data_realsense1['label_map'])
            
            # 保存annotations为结构化数组
            annotations_array = np.array(annotation_data_realsense1['annotations'], 
                                       dtype=[('x', 'i4'), ('y', 'i4'), ('label', 'i4')])
            realsense1_group.create_dataset('annotations', data=annotations_array)
            
            # 为realsense2创建标注数据
            realsense2_group = label_group.create_group('realsense2')
            realsense2_group.create_dataset('heatmap', data=annotation_data_realsense2['heatmap'])
            realsense2_group.create_dataset('label_map', data=annotation_data_realsense2['label_map'])
            
            # 保存annotations为结构化数组
            annotations_array = np.array(annotation_data_realsense2['annotations'], 
                                       dtype=[('x', 'i4'), ('y', 'i4'), ('label', 'i4')])
            realsense2_group.create_dataset('annotations', data=annotations_array)
    
    print(f"标注HDF5文件已创建: {labeled_hdf5_path}")
    return labeled_hdf5_path


def annotate_video(video_path, video_name, save_path):
    """对单个视频进行标注"""
    print(f"\n=== 标注 {video_name} ===")
    
    # 读取第一帧
    frame = load_first_frame(video_path)
    if frame is None:
        return None
    
    print(f"视频尺寸: {frame.shape}")
    print("操作说明:")
    print("1. 在图像上点击选择要标注的点")
    print("2. 输入标签数字 (0-5)")
    print("3. 可以标注多个点")
    print("4. 按 'q' 键完成当前视频标注并保存图片")
    print("5. 按 'r' 键重置所有标注")
    
    # 设置窗口名称
    window_name = f"Data Annotation - {video_name}"
    
    # 创建标注器
    annotator = DataAnnotator(frame, window_name)
    
    # 设置鼠标回调
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, annotator.mouse_callback)
    
    # 显示初始状态
    annotator.show_annotation(window_name)
    
    # 主循环
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # 完成当前视频标注
            print(f"\n完成 {video_name} 标注...")
            final_window_name = f"Final Results - {video_name}"
            overlay = annotator.show_final_results(final_window_name)
            print("按任意键保存图片并继续...")
            cv2.waitKey(0)
            
            # 销毁最终结果窗口
            cv2.destroyWindow(final_window_name)
            
            # 保存标注图片
            annotator.save_annotation_image(save_path, video_name)
            
            cv2.destroyWindow(window_name)
            return annotator.get_annotation_data()
        elif key == ord('r'):
            # 重置标注
            print(f"\n重置 {video_name} 标注...")
            # 销毁当前窗口
            cv2.destroyWindow(window_name)
            # 重新创建标注器
            annotator = DataAnnotator(frame, window_name)
            # 重新创建窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, annotator.mouse_callback)
            annotator.show_annotation(window_name)
    
    return None


def main():
    """主函数"""
    # 获取最新文件夹
    latest_folder = get_latest_hdf5_folder("./save/sun/")
    
    if latest_folder is None:
        return
    
    print(f"找到最新HDF5文件夹: {latest_folder}")
    
    # 构建视频文件路径
    realsense1_video = os.path.join(latest_folder, "realsense1_color.mp4")
    realsense2_video = os.path.join(latest_folder, "realsense2_color.mp4")
    
    # 检查视频文件是否存在
    if not os.path.exists(realsense1_video):
        print(f"realsense1视频文件不存在: {realsense1_video}")
        return
    
    if not os.path.exists(realsense2_video):
        print(f"realsense2视频文件不存在: {realsense2_video}")
        return
    
    print("\n=== 数据标注工具 ===")
    print("将依次对realsense1和realsense2进行标注")
    
    # 第一步：标注realsense1视频
    print("\n" + "="*50)
    print("第一步：标注 realsense1 视频")
    print("="*50)
    annotation_data_realsense1 = annotate_video(realsense1_video, "realsense1", latest_folder)
    if annotation_data_realsense1 is None:
        print("realsense1标注失败")
        return
    
    print(f"realsense1标注完成，标注点数量: {len(annotation_data_realsense1['annotations'])}")
    
    # 第二步：标注realsense2视频
    print("\n" + "="*50)
    print("第二步：标注 realsense2 视频")
    print("="*50)
    annotation_data_realsense2 = annotate_video(realsense2_video, "realsense2", latest_folder)
    if annotation_data_realsense2 is None:
        print("realsense2标注失败")
        return
    
    print(f"realsense2标注完成，标注点数量: {len(annotation_data_realsense2['annotations'])}")
    
    # 第三步：查找原始HDF5文件并创建带标注的HDF5文件
    print("\n" + "="*50)
    print("第三步：创建带标注的HDF5文件")
    print("="*50)
    
    hdf5_files = glob.glob(os.path.join(latest_folder, "*.h5"))
    if not hdf5_files:
        print("未找到HDF5文件")
        return
    
    # 使用第一个HDF5文件（通常只有一个）
    original_hdf5_path = hdf5_files[0]
    
    # 创建带标注的HDF5文件
    labeled_hdf5_path = create_labeled_hdf5(original_hdf5_path, annotation_data_realsense1, annotation_data_realsense2)
    
    print(f"\n" + "="*50)
    print("标注完成！")
    print("="*50)
    print(f"原始HDF5文件: {original_hdf5_path}")
    print(f"标注HDF5文件: {labeled_hdf5_path}")
    print(f"realsense1标注点数量: {len(annotation_data_realsense1['annotations'])}")
    print(f"realsense2标注点数量: {len(annotation_data_realsense2['annotations'])}")
    print(f"保存的图片文件:")
    print(f"  - {os.path.join(latest_folder, 'realsense1_heatmap_overlay.jpg')}")
    print(f"  - {os.path.join(latest_folder, 'realsense2_heatmap_overlay.jpg')}")


if __name__ == "__main__":
    main()