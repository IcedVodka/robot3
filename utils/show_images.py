from Sensor.depth_camera import RealsenseSensor
import cv2
import numpy as np
import os
from datetime import datetime

# 创建截图保存目录
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

left_camera = RealsenseSensor("left_camera")    
right_camera = RealsenseSensor("right_camera")

left_camera.set_up("327122078945")
right_camera.set_up("207522073950")

# 创建窗口
cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)

# 设置窗口大小
cv2.resizeWindow('Left Camera', 640, 480)
cv2.resizeWindow('Right Camera', 640, 480)

print("按 's' 键截图保存")
print("按 'q' 键退出程序")

try:
    while True:
        # 获取实时图像
        left_bgr_frame = left_camera.get_information()['color']
        right_bgr_frame = right_camera.get_information()['color']
        
        # 显示图像
        cv2.imshow('Left Camera', left_bgr_frame)
        cv2.imshow('Right Camera', right_bgr_frame)
        
        # 等待按键，1ms延迟
        key = cv2.waitKey(1) & 0xFF
        
        # 按 's' 键截图
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            left_filename = os.path.join(screenshot_dir, f"left_camera_{timestamp}.jpg")
            right_filename = os.path.join(screenshot_dir, f"right_camera_{timestamp}.jpg")
            
            cv2.imwrite(left_filename, left_bgr_frame)
            cv2.imwrite(right_filename, right_bgr_frame)
            print(f"截图已保存: {left_filename}, {right_filename}")
        
        # 按 'q' 键退出
        elif key == ord('q'):
            print("程序退出")
            break

finally:
    # 清理资源
    cv2.destroyAllWindows()
    left_camera.cleanup()
    right_camera.cleanup()






