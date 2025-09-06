#!/usr/bin/env python3
"""
RealSense深度图与彩色图对齐功能演示

这个脚本演示了如何使用RealSense的对齐功能来解决深度图视角范围比彩色图更广的问题。
"""

import cv2
import numpy as np
import time
from Sensor.depth_camera import RealsenseSensor

def main():
    # 相机序列号（请根据实际情况修改）
    camera_serial = "207522073950"  # 请替换为你的相机序列号
    
    print("RealSense对齐功能演示")
    print("=" * 50)
    
    # 创建传感器实例
    sensor = RealsenseSensor("alignment_demo")
    
    try:
        # 设置相机（启用对齐功能）
        print("正在初始化相机（启用对齐）...")
        sensor.set_up(
            camera_serial=camera_serial,
            is_depth=True,
            resolution=[640, 480],
            enable_alignment=True  # 启用对齐
        )
        
        print("相机初始化完成！")
        print("按 'q' 退出，按 'a' 切换对齐模式")
        
        # 创建显示窗口
        cv2.namedWindow('Color Image', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth Image (Aligned)', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth Image (Original)', cv2.WINDOW_AUTOSIZE)
        
        alignment_enabled = True
        
        while True:
            # 获取数据
            data = sensor.get_information()
            if data is None:
                continue
                
            color_img = data.get('color')
            depth_img = data.get('depth')
            
            if color_img is None or depth_img is None:
                continue
            
            # 显示彩色图像
            cv2.imshow('Color Image', color_img)
            
            # 显示对齐后的深度图
            if alignment_enabled:
                # 深度图已经对齐，直接显示
                depth_display = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                cv2.imshow('Depth Image (Aligned)', depth_display)
            else:
                # 显示原始深度图（未对齐）
                depth_display = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                cv2.imshow('Depth Image (Original)', depth_display)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # 切换对齐模式
                alignment_enabled = not alignment_enabled
                sensor.set_alignment(alignment_enabled)
                print(f"对齐模式: {'启用' if alignment_enabled else '禁用'}")
            
            time.sleep(0.03)  # 约30fps
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        sensor.cleanup()
        print("资源已清理")

if __name__ == "__main__":
    main()
