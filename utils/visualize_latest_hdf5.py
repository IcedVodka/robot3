import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_latest_run_dir(candidates: List[Path]) -> Optional[Path]:
    """
    在候选根目录列表中查找最近一次的运行目录：
    规则：遍历根目录下的所有子目录，选择包含单个 .h5 文件的目录，按 h5 文件的修改时间或目录修改时间排序，返回最新的。
    """
    latest: Optional[Tuple[float, Path]] = None
    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            h5_list = list(d.glob("*.h5"))
            if not h5_list:
                continue
            # 取该目录下任意一个 h5 的修改时间作为该 run 的时间戳依据
            newest_h5 = max(h5_list, key=lambda p: p.stat().st_mtime)
            mtime = newest_h5.stat().st_mtime
            if latest is None or mtime > latest[0]:
                latest = (mtime, d)
    return latest[1] if latest else None


def load_h5_datasets(h5_path: Path):
    with h5py.File(str(h5_path), 'r') as f:
        attrs = {k: f.attrs.get(k) for k in f.attrs.keys()}
        # 解析可能为 JSON 字符串的相机内参
        for k in ["realsense1", "realsense2"]:
            v = attrs.get(k)
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', errors='ignore')
            if isinstance(v, str):
                try:
                    attrs[k] = json.loads(v)
                except Exception:
                    pass

        joints = f["robot/joint"][:] if "robot/joint" in f else np.empty((0, 0))
        poses = f["robot/pose"][:] if "robot/pose" in f else np.empty((0, 0))
        grip = f["robot/gripper_state"][:] if "robot/gripper_state" in f else np.empty((0,))
        ts = f["timestamps/sample_time"][:] if "timestamps/sample_time" in f else np.empty((0,))

    return attrs, joints, poses, grip, ts


def write_metadata_txt(run_dir: Path, attrs: dict, h5_file: Path) -> Path:
    out_txt = run_dir / "metadata.txt"
    lines = []
    lines.append(f"h5_file: {h5_file.name}")
    for k in ["task_name", "task_description", "created_at"]:
        if k in attrs:
            v = attrs.get(k)
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', errors='ignore')
            lines.append(f"{k}: {v}")
    for cam in ["realsense1", "realsense2"]:
        v = attrs.get(cam)
        try:
            v_str = json.dumps(v, ensure_ascii=False)
        except Exception:
            v_str = str(v)
        lines.append(f"{cam}: {v_str}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    return out_txt


def _prep_plot_axes(ts: np.ndarray) -> np.ndarray:
    if ts is not None and ts.size > 0:
        x = ts - ts[0]
    else:
        x = np.arange(len(ts)) if ts is not None else np.array([])
    return x


def plot_and_save_joints(run_dir: Path, joints: np.ndarray, ts: np.ndarray) -> Optional[Path]:
    if joints is None or joints.size == 0:
        return None
    x = _prep_plot_axes(ts)
    plt.figure(figsize=(10, 6))
    n_dof = joints.shape[1] if joints.ndim == 2 else 0
    for i in range(n_dof):
        plt.plot(x, joints[:, i], label=f"joint_{i+1}")
    plt.xlabel("time (s)")
    plt.ylabel("angle")
    plt.title("Joints")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    out_path = run_dir / "joints.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    return out_path


def plot_and_save_pose(run_dir: Path, poses: np.ndarray, ts: np.ndarray) -> Optional[Path]:
    if poses is None or poses.size == 0:
        return None
    x = _prep_plot_axes(ts)
    plt.figure(figsize=(10, 6))
    labels = ["x", "y", "z", "rx", "ry", "rz"]
    n = min(poses.shape[1] if poses.ndim == 2 else 0, 6)
    for i in range(n):
        plt.plot(x, poses[:, i], label=labels[i] if i < len(labels) else f"pose_{i}")
    plt.xlabel("time (s)")
    plt.ylabel("value")
    plt.title("Pose [x,y,z,rx,ry,rz]")
    plt.legend(loc="best", ncol=3)
    plt.tight_layout()
    out_path = run_dir / "pose.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    return out_path


def plot_and_save_gripper(run_dir: Path, grip: np.ndarray, ts: np.ndarray) -> Optional[Path]:
    if grip is None or grip.size == 0:
        return None
    x = _prep_plot_axes(ts)
    plt.figure(figsize=(10, 3))
    plt.step(x, grip.astype(float), where='post')
    plt.yticks([0, 1], ["open(0)", "close(1)"])
    plt.xlabel("time (s)")
    plt.ylabel("state")
    plt.title("Gripper State")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = run_dir / "gripper.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    return out_path


def _safe_read_frame(cap: cv2.VideoCapture, size: Tuple[int, int]) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    if frame is None:
        return None
    if size is not None:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def stitch_four_videos(run_dir: Path,
                       out_name: str = "combined_2x2.mp4",
                       order: Tuple[str, str, str, str] = (
                           "realsense1_color.mp4",
                           "realsense1_depth.mp4",
                           "realsense2_color.mp4",
                           "realsense2_depth.mp4",
                       )) -> Optional[Path]:
    paths = [run_dir / name for name in order]
    if not all(p.exists() for p in paths):
        # 若四个视频不全则跳过
        return None

    caps = [cv2.VideoCapture(str(p)) for p in paths]
    try:
        # 以第一个视频作为尺寸与 fps 参考
        w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        # 统一尺寸
        target_size = (w, h)

        # 帧数取四路中的最小
        frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        total_frames = min(frame_counts)
        if total_frames <= 0:
            return None

        out_w, out_h = w * 2, h * 2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = run_dir / out_name
        vw = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 25.0, (out_w, out_h))

        try:
            for _ in range(total_frames):
                frames = [_safe_read_frame(caps[i], target_size) for i in range(4)]
                if any(f is None for f in frames):
                    break
                tl, bl, tr, br = frames
                top = np.hstack([tl, tr])
                bottom = np.hstack([bl, br])
                grid = np.vstack([top, bottom])
                vw.write(grid)
        finally:
            vw.release()
        return out_path
    finally:
        for c in caps:
            c.release()


def main():
    parser = argparse.ArgumentParser(description="可视化最近一次 HDF5 记录，并拼接 2x2 视频")
    parser.add_argument("--base", type=str, default="./save", help="运行输出根目录，默认 ./save；若无则回退到 ./outputs")
    args = parser.parse_args()

    base1 = Path(args.base).resolve()
    base2 = Path("./outputs").resolve()

    run_dir = find_latest_run_dir([base1, base2])
    if run_dir is None:
        print("未找到包含 .h5 的运行目录。")
        sys.exit(1)
    h5_files = sorted(run_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not h5_files:
        print("该运行目录下未找到 .h5 文件。")
        sys.exit(1)
    h5_path = h5_files[0]

    # 读取 HDF5 数据
    attrs, joints, poses, grip, ts = load_h5_datasets(h5_path)

    # 写元信息
    meta_txt = write_metadata_txt(run_dir, attrs, h5_path)
    print(f"元信息已写入: {meta_txt}")

    # 画图
    jp = plot_and_save_joints(run_dir, joints, ts)
    pp = plot_and_save_pose(run_dir, poses, ts)
    gp = plot_and_save_gripper(run_dir, grip, ts)
    if jp: print(f"已保存: {jp}")
    if pp: print(f"已保存: {pp}")
    if gp: print(f"已保存: {gp}")

    # 拼接视频
    out_video = stitch_four_videos(run_dir)
    if out_video is not None:
        print(f"拼接视频已保存: {out_video}")
    else:
        print("未找到齐全的四路视频或拼接失败，跳过拼接。")


if __name__ == "__main__":
    main()


