import argparse
import time
import os
import datetime
import json
from typing import List, Optional

import numpy as np
import h5py
import cv2
from tqdm import tqdm


from utils.logger import setup_logger
from Controller.realman_controller import RealmanController
from Sensor.depth_camera import RealsenseSensor
from utils.keyboard import start_kb_listener
from utils.hand_serial import SerialAngleReader
from utils.angle_mapper import AngleLinearMapper

task_condition = {
            "outputs_path": "./save/", 
            "task_name": "grasp", 
            "task_description": "Open the drawer, take out the green cup, place it on the green plate, and then put the medicine bottle in the cup.",
            "save_freq": 30,
            "joint_dof": 6,
            "depth_alpha": 0.03,  # 深度图转换为8位时的缩放参数
        }
realsense1_serial = "207522073950"
realsense2_serial = "327122078945"
realsense1_intr = {'color': {'ppx': 324.6168212890625, 'ppy': 246.23873901367188, 'fx': 605.5936279296875, 'fy': 604.6068725585938}, 
                   'depth': {'ppx': 320.8646545410156, 'ppy': 236.83616638183594, 'fx': 382.53472900390625, 'fy': 382.53472900390625}}
realsense2_intr = {'color': {'ppx': 327.67547607421875, 'ppy': 245.18077087402344, 'fx': 608.4032592773438, 'fy': 608.064208984375}, 
                   'depth': {'ppx': 323.422119140625, 'ppy': 241.96876525878906, 'fx': 388.0012512207031, 'fy': 388.0012512207031}}



class data_collection:
    def __init__(self) -> None:
        self.cfg = None
        self.file = None
        self.step_idx = 0
        self.initialized = False
        self.run_dir = None
        self.h5_path = None
        self.joint_dof = 6
        # 内存帧缓存：按相机、模态分别存储
        self.frames = {
            "realsense1": {"color": [], "depth": []},
            "realsense2": {"color": [], "depth": []},
        }
        self.meta = {}

    def set_up(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        task_name = self.cfg.get("task_name", "session")
        outputs_path = self.cfg.get("outputs_path", "./outputs/")
        os.makedirs(outputs_path, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{task_name}_{ts}"
        # 运行输出目录：包含h5、summary与视频
        self.run_dir = os.path.join(outputs_path, base)
        os.makedirs(self.run_dir, exist_ok=True)
        fname = f"{base}.h5"
        self.h5_path = os.path.join(self.run_dir, fname)
        self.file = h5py.File(self.h5_path, "w")
        
        # 预创建分组（仅非图像相关）
        self.file.require_group("robot")
        self.file.require_group("timestamps")

        self.save_freq = int(self.cfg.get("save_freq", 30))

        # 顶层元信息（任务+相机，一次性写入）
        self.file.attrs["task_name"] = task_name
        self.file.attrs["task_description"] = self.cfg.get("task_description", "")
        self.file.attrs["created_at"] = ts
        self.file.attrs["depth_alpha"] = float(self.cfg.get("depth_alpha", 0.03))  # 深度转换参数
        self.file.attrs["realsense1"] = json.dumps(realsense1_intr or {}, ensure_ascii=False)
        self.file.attrs["realsense2"] = json.dumps(realsense2_intr or {}, ensure_ascii=False)
    

    def _init_datasets(self, data1: dict, data2: dict, state: dict) -> None:
        # 固定维度建表（仅非图像数据）
        robot_grp = self.file["robot"]
        ts_grp = self.file["timestamps"]
        
        # 机器人状态（固定关节自由度与位姿维度）
        self.joint_dof = int(self.cfg.get("joint_dof", 6))
        self.ds_slave_joint = robot_grp.create_dataset(
            "joint", shape=(0, self.joint_dof), maxshape=(None, self.joint_dof), chunks=(1, self.joint_dof), dtype=np.float64
        )
        # 位姿数据固定为6: x, y, z, rx, ry, rz
        self.ds_slave_pose = robot_grp.create_dataset(
            "pose", shape=(0, 6), maxshape=(None, 6), chunks=(1, 6), dtype=np.float64
        )
        

        # 时间戳（统一采样时间）
        self.ds_time = ts_grp.create_dataset("sample_time", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64)

        self.initialized = True

    def _append_ds(self, ds, sample):
        if ds is None or sample is None:
            return
        cur = ds.shape[0]
        ds.resize((cur + 1,) + ds.shape[1:])
        ds[cur] = sample

    def collect(self, data1: Optional[dict], data2: Optional[dict], robot_state: Optional[dict]) -> None:
        if self.file is None:
            return

        t = time.time()

        if not self.initialized:
            self._init_datasets(data1 or {}, data2 or {}, robot_state or {})

       # 处理图像数据（不写入HDF5，仅缓存）
        color1 = data1.get("color") if data1 else None      
        depth1 = data1.get("depth") if data1 else None
        color2 = data2.get("color") if data2 else None
        depth2 = data2.get("depth") if data2 else None
        # 同步缓存到内存用于保存时拼接视频
        if color1 is not None:
            self.frames["realsense1"]["color"].append(np.array(color1))
        if depth1 is not None:
            self.frames["realsense1"]["depth"].append(np.array(depth1))
        if color2 is not None:
            self.frames["realsense2"]["color"].append(np.array(color2))
        if depth2 is not None:
            self.frames["realsense2"]["depth"].append(np.array(depth2))

        # 追加写入（仅写入非图像数据）
        self._append_ds(self.ds_time, t)

        # 关节与位姿数据（固定维度）
        joints = (robot_state or {}).get("joint")
        if isinstance(joints, (list, tuple)) and len(joints) == self.ds_slave_joint.shape[1]:
            cur = self.ds_slave_joint.shape[0]
            self.ds_slave_joint.resize((cur + 1, self.ds_slave_joint.shape[1]))
            self.ds_slave_joint[cur, :] = np.asarray(joints, dtype=np.float64)

        pose = (robot_state or {}).get("pose")
        if isinstance(pose, (list, tuple)) and len(pose) == self.ds_slave_pose.shape[1]:
            cur = self.ds_slave_pose.shape[0]
            self.ds_slave_pose.resize((cur + 1, self.ds_slave_pose.shape[1]))
            self.ds_slave_pose[cur, :] = np.asarray(pose, dtype=np.float64)

        self.step_idx += 1
        if self.step_idx % max(1, self.save_freq) == 0:
            self.file.flush()

    def _prepare_depth_for_video(self, depth: np.ndarray) -> np.ndarray:
        """
        准备深度数据用于视频写入，使用convertScaleAbs转换为8位
        Args:
            depth: 原始深度数据 (H, W)，单位为毫米
        Returns:
            np.ndarray: 处理后的深度数据，8位格式
        """
        # 从配置中获取alpha参数，默认为0.03
        alpha = float(self.cfg.get("depth_alpha", 0.03)) if self.cfg else 0.03
        
        if depth.ndim == 2:
            # 使用cv2.convertScaleAbs将深度数据转换为8位
            # alpha参数用于缩放深度值到合适的范围
            depth_8bit = cv2.convertScaleAbs(depth, alpha=alpha)
            return depth_8bit
        if depth.ndim == 3 and depth.shape[-1] == 1:
            return self._prepare_depth_for_video(depth[..., 0])
        return cv2.convertScaleAbs(depth, alpha=alpha)

    def _write_video(self, frames: list, out_path: str, fps: int = 20, is_depth: bool = False) -> None:
        if cv2 is None or not frames:
            return
        h, w = frames[0].shape[:2]
        
        # 对于深度视频，使用8位灰度格式
        if is_depth:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h), isColor=False)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
            
        try:
            iterator = frames
            if tqdm is not None:
                desc = f"Writing {os.path.basename(out_path)}"
                iterator = tqdm(frames, desc=desc, unit="frame")
            for frm in iterator:
                if is_depth:
                    # 深度视频：直接写入8位灰度数据
                    if frm.dtype != np.uint8:
                        frm = frm.astype(np.uint8)
                    vw.write(frm)
                else:
                    # 彩色视频：转换为BGR格式
                    if frm.ndim == 2:
                        frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
                    vw.write(frm)
        finally:
            vw.release()
    

    def save(self) -> None:
        if self.file is not None:
            # 关闭前确保将相机元信息顶层属性写入完成
            self.file.flush()
            self.file.close()
            self.file = None
            self.initialized = False


            # 将缓存帧写为视频
            fps = int(self.cfg.get("save_freq", 30))
            for cam in ["realsense1", "realsense2"]:
                # 彩色
                color_frames = self.frames.get(cam, {}).get("color", [])
                if color_frames:
                    self._write_video(color_frames, os.path.join(self.run_dir, f"{cam}_color.mp4"), fps=fps, is_depth=False)
                # 深度（保持原始16位数据）
                depth_frames_src = self.frames.get(cam, {}).get("depth", [])
                if depth_frames_src:
                    depth_frames = [self._prepare_depth_for_video(x) for x in depth_frames_src]
                    self._write_video(depth_frames, os.path.join(self.run_dir, f"{cam}_depth.mp4"), fps=fps, is_depth=True)

            # 释放内存缓存
            self.frames = {
                "realsense1": {"color": [], "depth": []},
                "realsense2": {"color": [], "depth": []},
            }

class Teleop:
    def __init__(self, master_ip: str, slave_ip: str, port: int = 8080, rate: float = 30.0, hand: bool = False, record: bool = False) -> None:
        self.master = RealmanController(name="Master")
        self.slave = RealmanController(name="Slave",is_hand= True)
        self.master_ip = master_ip
        self.slave_ip = slave_ip
        self.port = port
        self.rate = max(1e-6, rate)
        self.period_s = 1.0 / self.rate
        self.running = False
        self.paused = False
        self.hand_enabled = bool(hand)
        self.record_enabled = bool(record)
        self._last_ts: Optional[float] = None

        self.data_collection = data_collection()

        self.realsense1 = RealsenseSensor(name="Realsense1")
        self.realsense2 = RealsenseSensor(name="Realsense2")
        
        # 初始化手部角度读取器和映射器
        self.angle_reader = SerialAngleReader()
        self.angle_mapper = AngleLinearMapper(
            in_mins=[33, 50, 50, 51, 70, 90],
            in_maxs=[167, 167, 171, 169, 171, 160],
            out_mins=[0, 0, 0, 0, 0, 0],
            out_maxs=[65535, 65535, 65535, 65535, 65535, 65535],
            clip=True,
            round_output=True,
            reverse_mapping=[True, True, True, True, True, False]
        )

    def connect(self) -> None:
        self.master.set_up(self.master_ip, self.port)
        self.slave.set_up(self.slave_ip, self.port)
        # 启用手爪控制（共享同一连接）
        if self.hand_enabled:
            self.slave.is_hand = True
        self.realsense1.set_up(realsense1_serial)
        self.realsense2.set_up(realsense2_serial)
        
        # 只在启用记录时设置数据收集
        if self.record_enabled:
            print("启用数据记录")
            self.data_collection.set_up(task_condition)
        
        # 启动角度读取器
        self.angle_reader.start()

    def step(self) -> None:
        state = self.master.get_state()
        joints = state.get("joint") if isinstance(state, dict) else None
        if joints is not None:
            self.slave.set_arm_joints(joints)
        
        # 读取手部角度并映射
        if self.hand_enabled:
            raw_angles = self.angle_reader.get_latest()
            if raw_angles is not None:
                mapped_angles = self.angle_mapper.map_angles(raw_angles)
                self.slave.set_hand_pos(mapped_angles, block=False)
        
        # 只在启用记录时收集数据
        if self.record_enabled:
            data1 = self.realsense1.get_information()
            data2 = self.realsense2.get_information()
            data3 = self.slave.get_state()
            self.data_collection.collect(data1, data2, data3)   
        

    def start(self) -> None:
        self.connect()
        print(f"开始遥操作：{self.master_ip} -> {self.slave_ip}，频率 {self.rate} Hz。")
        print(f"手爪控制: {'启用' if self.hand_enabled else '禁用'}")
        print(f"数据记录: {'启用' if self.record_enabled else '禁用'}")
        print("键盘控制: [e]暂停/继续  [q]结束")

        self.running = True
        self.paused = False
        self._last_ts = None

        t, stop_evt, q = start_kb_listener()
        try:
            while self.running:
                if self._last_ts is None:
                    self._last_ts = time.perf_counter()

                # 消费队列中的按键事件
                while not q.empty():
                    key = q.get_nowait()
                    if key in ('q', 'Q'):
                        self.stop()
                        break
                    elif key in ('e', 'E'):
                        self.paused = not self.paused
                        print("已暂停" if self.paused else "继续运行")

                if not self.paused and self.running:
                    self.step()

                # 定频
                now = time.perf_counter()
                elapsed = now - self._last_ts
                sleep_t = self.period_s - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                self._last_ts = time.perf_counter()
        finally:
            stop_evt.set()
            t.join(timeout=1.0)

    def stop(self) -> None:
        self.running = False
        self.realsense1.cleanup()
        self.realsense2.cleanup()
        self.angle_reader.stop()
        if self.record_enabled:
            self.data_collection.save()


def main():
    parser = argparse.ArgumentParser(description="Realman 主从遥操作示例：主臂关节 -> 从臂关节 透传")
    parser.add_argument("--master-ip", required=True, help="主臂 IP 地址")
    parser.add_argument("--slave-ip", required=True, help="从臂 IP 地址")
    parser.add_argument("--port", type=int, default=8080, help="机械臂端口，默认 8080")
    parser.add_argument("--rate", type=float, default=30.0, help="发送频率 Hz，默认 30Hz")
    parser.add_argument("--hand", action="store_true", help="启用手爪控制")
    parser.add_argument("--record", action="store_true", help="启用数据记录")
    args = parser.parse_args()
    teleop = Teleop(master_ip=args.master_ip, slave_ip=args.slave_ip, port=args.port, rate=args.rate, hand=args.hand, record=args.record)
    try:
        teleop.start()
    except KeyboardInterrupt:
        teleop.stop()




if __name__ == "__main__":
    setup_logger()
    main()

