import argparse
import time
import os
import datetime
from typing import List, Optional

import numpy as np
import h5py

from Controller.realman_controller import RealmanController
from Sensor.depth_camera import RealsenseSensor
from utils.keyboard import kb_setup, kb_restore, kb_read_char, start_kb_listener

task_condition = {
            "save_path": "./save/", 
            "task_name": "test_1", 
            "task_description": "xxxxxxxxx",
            "save_format": "hdf5", 
            "save_freq": 30,
        }
realsense1_serial = "1234567890"
realsense2_serial = "1234567891"
realsense1_intr = {
    "color": {"ppx": 320.1, "ppy": 240.5, "fx": 600.2, "fy": 599.8},
    "depth": {"ppx": 320.0, "ppy": 240.0, "fx": 601.0, "fy": 600.0},
}
realsense2_intr = {
    "color": {"ppx": 319.8, "ppy": 241.0, "fx": 602.3, "fy": 601.1},
    "depth": {"ppx": 319.7, "ppy": 240.2, "fx": 603.0, "fy": 602.5},
}

def extract_joints(state: dict) -> Optional[List[float]]:
    # 常见键名兼容处理
    for key in ["joint", "joints", "joint_degree", "joint_degrees", "q"]:
        if key in state and isinstance(state[key], (list, tuple)):
            return list(state[key])
    return None


class data_collection:
    def __init__(self) -> None:
        self.cfg = None
        self.file = None
        self.step_idx = 0
        self.initialized = False
        self.meta = {
            "realsense1": None,
            "realsense2": None,
        }

    def set_up(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        save_path = self.cfg.get("save_path", "./save/")
        task_name = self.cfg.get("task_name", "session")
        os.makedirs(save_path, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{task_name}_{ts}.h5"
        self.file = h5py.File(os.path.join(save_path, fname), "w")

        # 顶层元信息
        self.file.attrs["task_name"] = task_name
        self.file.attrs["task_description"] = self.cfg.get("task_description", "")
        self.file.attrs["save_format"] = "hdf5"
        self.file.attrs["created_at"] = ts

        # 预创建分组
        self.file.require_group("sensors/realsense1")
        self.file.require_group("sensors/realsense2")
        self.file.require_group("robot/slave")
        self.file.require_group("timestamps")

        self.save_freq = int(self.cfg.get("save_freq", 30))

    def set_camera_meta(self, realsense1_intr: Optional[dict], realsense2_intr: Optional[dict]) -> None:
        self.meta["realsense1"] = realsense1_intr or {}
        self.meta["realsense2"] = realsense2_intr or {}

        # 将内参保存为属性
        for cam, intr in [("realsense1", self.meta["realsense1"]), ("realsense2", self.meta["realsense2"])]:
            grp = self.file.require_group(f"sensors/{cam}")
            if intr:
                sub = grp.require_group("intrinsics")
                for k, v in intr.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            sub.attrs[f"{k}.{kk}"] = vv
                    else:
                        sub.attrs[k] = v

    def _init_datasets(self, data1: dict, data2: dict, state: dict) -> None:
        # 基于第一帧动态建表
        rs1_grp = self.file["sensors/realsense1"]
        rs2_grp = self.file["sensors/realsense2"]
        robot_grp = self.file["robot/slave"]
        ts_grp = self.file["timestamps"]

        def create_image_ds(grp, name, sample):
            if sample is None:
                return None
            arr = np.asarray(sample)
            maxshape = (None,) + arr.shape
            chunks = (1,) + arr.shape
            return grp.create_dataset(name, shape=(0,) + arr.shape, maxshape=maxshape, chunks=chunks, dtype=arr.dtype)

        # 相机1
        self.ds_rs1_color = create_image_ds(rs1_grp, "color", data1.get("color") if data1 else None)
        self.ds_rs1_depth = create_image_ds(rs1_grp, "depth", data1.get("depth") if data1 else None)

        # 相机2
        self.ds_rs2_color = create_image_ds(rs2_grp, "color", data2.get("color") if data2 else None)
        self.ds_rs2_depth = create_image_ds(rs2_grp, "depth", data2.get("depth") if data2 else None)

        # 机器人状态（保存关节与手爪状态）
        joints = extract_joints(state) if state else None
        if joints is None:
            joints = []
        joints = np.asarray(joints, dtype=np.float64)
        self.ds_slave_joint = robot_grp.create_dataset(
            "joint", shape=(0, joints.shape[0] if joints.ndim else 0), maxshape=(None, joints.shape[0] if joints.ndim else 0), chunks=(1, joints.shape[0] if joints.ndim else 1), dtype=np.float64
        )
        # 手爪状态：0=开, 1=合
        self.ds_gripper_state = robot_grp.create_dataset(
            "gripper_state", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.int8
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

    def collect(self, data1: Optional[dict], data2: Optional[dict], robot_state: Optional[dict], gripper_state: Optional[int] = None) -> None:
        if self.file is None:
            return

        t = time.time()

        if not self.initialized:
            self._init_datasets(data1 or {}, data2 or {}, robot_state or {})

        # 处理图像数据
        color1 = data1.get("color") if data1 else None
        depth1 = data1.get("depth") if data1 else None
        color2 = data2.get("color") if data2 else None
        depth2 = data2.get("depth") if data2 else None

        # 对齐形状为创建时形状
        def safe_cast(sample, ds):
            if ds is None or sample is None:
                return None
            arr = np.asarray(sample)
            # 简化起见，仅在形状一致时写入
            return arr if arr.shape == ds.shape[1:] else None

        color1 = safe_cast(color1, self.ds_rs1_color)
        depth1 = safe_cast(depth1, self.ds_rs1_depth)
        color2 = safe_cast(color2, self.ds_rs2_color)
        depth2 = safe_cast(depth2, self.ds_rs2_depth)

        joints = extract_joints(robot_state or {}) or []
        joints = np.asarray(joints, dtype=np.float64)
        if self.ds_slave_joint.shape[1] == 0 and joints.size > 0:
            # 若首次关节维度未知（极少见），则重新定义不做，直接跳过保存
            pass

        # 追加写入
        self._append_ds(self.ds_time, t)
        self._append_ds(self.ds_rs1_color, color1)
        self._append_ds(self.ds_rs1_depth, depth1)
        self._append_ds(self.ds_rs2_color, color2)
        self._append_ds(self.ds_rs2_depth, depth2)
        if gripper_state is not None:
            try:
                self._append_ds(self.ds_gripper_state, np.int8(gripper_state))
            except Exception:
                pass

        # 关节
        cur = self.ds_slave_joint.shape[0]
        self.ds_slave_joint.resize((cur + 1, self.ds_slave_joint.shape[1]))
        if joints.size == self.ds_slave_joint.shape[1]:
            self.ds_slave_joint[cur, :] = joints

        self.step_idx += 1
        if self.step_idx % max(1, self.save_freq) == 0:
            self.file.flush()

    def save(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None
            self.initialized = False

class Teleop:
    def __init__(self, master_ip: str, slave_ip: str, port: int = 8080, rate: float = 30.0, hand: bool = False) -> None:
        self.master = RealmanController(name="Master")
        self.slave = RealmanController(name="Slave")
        self.master_ip = master_ip
        self.slave_ip = slave_ip
        self.port = port
        self.rate = max(1e-6, rate)
        self.period_s = 1.0 / self.rate
        self.running = False
        self.paused = False
        self.hand_enabled = bool(hand)
        self._last_ts: Optional[float] = None
        # 手爪状态：0=开, 1=合
        self.gripper_state: int = 0

        self.data_collection = data_collection()

        self.realsense1 = RealsenseSensor(name="Realsense1")
        self.realsense2 = RealsenseSensor(name="Realsense2")

    def connect(self) -> None:
        self.master.set_up(self.master_ip, self.port)
        self.slave.set_up(self.slave_ip, self.port)
        # 启用手爪控制（共享同一连接）
        if self.hand_enabled:
            self.slave.is_hand = True
        self.realsense1.set_up(realsense1_serial)
        self.realsense2.set_up(realsense2_serial)
        self.data_collection.set_up(task_condition)
        # 使用用户提供的相机内参
        self.data_collection.set_camera_meta(realsense1_intr, realsense2_intr)

    def step(self) -> None:
        state = self.master.get_state()
        joints = extract_joints(state)
        if joints is not None:
            try:
                self.slave.set_arm_joints(joints)
            except Exception:
                pass
        data1 = self.realsense1.get_information()
        data2 = self.realsense2.get_information()
        data3 = self.slave.get_state()
        self.data_collection.collect(data1, data2, data3, gripper_state=self.gripper_state)   
        

    def start(self) -> None:
        self.connect()
        print(f"开始遥操作：{self.master_ip} -> {self.slave_ip}，频率 {self.rate} Hz。")
        print("键盘控制: [p]暂停/继续  [q]结束  [o]手爪开  [c]手爪合")

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
                    elif key in ('p', 'P'):
                        self.paused = not self.paused
                        print("已暂停" if self.paused else "继续运行")
                    elif key in ('o', 'O') and self.hand_enabled:
                        try:
                            self.slave.release_hand(block=False)
                            print("手爪开")
                            self.gripper_state = 0
                        except Exception:
                            pass
                    elif key in ('c', 'C') and self.hand_enabled:
                        try:
                            self.slave.grip_hand(block=False)
                            print("手爪合")
                            self.gripper_state = 1
                        except Exception:
                            pass

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
        self.data_collection.save()


def main():
    parser = argparse.ArgumentParser(description="Realman 主从遥操作示例：主臂关节 -> 从臂关节 透传")
    parser.add_argument("--master-ip", required=True, help="主臂 IP 地址")
    parser.add_argument("--slave-ip", required=True, help="从臂 IP 地址")
    parser.add_argument("--port", type=int, default=8080, help="机械臂端口，默认 8080")
    parser.add_argument("--rate", type=float, default=30.0, help="发送频率 Hz，默认 30Hz")
    parser.add_argument("--hand", action="store_true", help="启用手爪控制(按 o 开, c 合)")
    args = parser.parse_args()
    teleop = Teleop(master_ip=args.master_ip, slave_ip=args.slave_ip, port=args.port, rate=args.rate, hand=args.hand)
    try:
        teleop.start()
    except KeyboardInterrupt:
        teleop.stop()




if __name__ == "__main__":
    main()

