import os
import glob
import json
import math
import argparse
from datetime import datetime

import h5py
import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latest_h5(save_dir: str) -> str:
    pattern = os.path.join(save_dir, "*.h5")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {save_dir}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dataset_exists(f: h5py.File, path: str) -> bool:
    try:
        f[path]
        return True
    except KeyError:
        return False


def read_attr_dict(hobj) -> dict:
    out = {}
    for k, v in hobj.attrs.items():
        try:
            out[str(k)] = v.item() if hasattr(v, "shape") and v.shape == () else v
        except Exception:
            try:
                out[str(k)] = v.decode("utf-8")  # bytes → str
            except Exception:
                out[str(k)] = str(v)
    return out


def write_summary_txt(f: h5py.File, out_txt: str) -> None:
    lines = []
    # top-level attrs
    lines.append("# HDF5 Summary\n")
    attrs = read_attr_dict(f)
    for k in sorted(attrs.keys()):
        lines.append(f"{k}: {attrs[k]}")
    lines.append("")

    # sensors info
    for cam in ["realsense1", "realsense2"]:
        base = f"sensors/{cam}"
        if base in f:
            lines.append(f"[{cam}]")
            for name in ["color", "depth"]:
                ds_path = f"{base}/{name}"
                if dataset_exists(f, ds_path):
                    ds = f[ds_path]
                    lines.append(f"- {name}: shape={ds.shape}, dtype={ds.dtype}")
            intr_path = f"{base}/intrinsics"
            if intr_path in f:
                intr_attrs = read_attr_dict(f[intr_path])
                lines.append(f"- intrinsics: {json.dumps(intr_attrs, ensure_ascii=False)}")
            lines.append("")

    # robot info
    if "robot/slave" in f:
        lines.append("[robot/slave]")
        for name in ["joint", "pose", "gripper_state"]:
            p = f"robot/slave/{name}"
            if dataset_exists(f, p):
                ds = f[p]
                lines.append(f"- {name}: shape={ds.shape}, dtype={ds.dtype}")
        lines.append("")

    # timestamps
    if dataset_exists(f, "timestamps/sample_time"):
        ds = f["timestamps/sample_time"]
        lines.append(f"[timestamps]\n- sample_time: shape={ds.shape}, dtype={ds.dtype}")

    with open(out_txt, "w", encoding="utf-8") as wf:
        wf.write("\n".join(lines) + "\n")


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 2:
        d = depth.astype(np.float32)
        finite = np.isfinite(d)
        if not np.any(finite):
            return np.zeros_like(d, dtype=np.uint8)
        vmin = np.percentile(d[finite], 1)
        vmax = np.percentile(d[finite], 99)
        if vmax <= vmin:
            vmax = vmin + 1.0
        d = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        d = (d * 255.0).astype(np.uint8)
        return d
    # if single channel with last dim 1
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return normalize_depth_to_uint8(depth[..., 0])
    # already looks like color
    return depth.astype(np.uint8)


def write_video(frames: list, out_path: str, fps: int = 20) -> None:
    if cv2 is None or not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    try:
        for frm in frames:
            if frm.ndim == 2:
                frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
            vw.write(frm)
    finally:
        vw.release()


def concat_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a is None:
        return b
    if b is None:
        return a
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    H = max(ha, hb)
    Wa = wa
    Wb = wb
    def pad(img, target_h):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        top = (target_h - h) // 2
        bottom = target_h - h - top
        if img.ndim == 2:
            img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img
    a2 = pad(a, H)
    b2 = pad(b, H)
    if a2.ndim == 2:
        a2 = cv2.cvtColor(a2, cv2.COLOR_GRAY2BGR)
    if b2.ndim == 2:
        b2 = cv2.cvtColor(b2, cv2.COLOR_GRAY2BGR)
    return np.concatenate([a2, b2], axis=1)


def export_videos(f: h5py.File, out_dir: str, fps: int = 20) -> None:
    if cv2 is None:
        return
    ensure_dir(out_dir)

    # Collect per-camera videos
    color_frames_all = []
    depth_frames_all = []

    # We'll also build side-by-side streams if both cams exist
    color_sbs = []
    depth_sbs = []

    cams = []
    for cam in ["realsense1", "realsense2"]:
        base = f"sensors/{cam}"
        if base in f:
            cams.append(cam)

    max_len = 0
    # Determine max frames count for iteration
    for cam in cams:
        for name in ["color", "depth"]:
            p = f"sensors/{cam}/{name}"
            if dataset_exists(f, p):
                max_len = max(max_len, f[p].shape[0])

    # Iterate frames
    iterator = range(max_len)
    if tqdm is not None:
        iterator = tqdm(iterator, total=max_len, desc="Exporting videos", unit="frame")

    for i in iterator:
        cam_frames_color = []
        cam_frames_depth = []

        # Per-cam extraction
        for cam in cams:
            c_path = f"sensors/{cam}/color"
            d_path = f"sensors/{cam}/depth"
            c_img = None
            d_img = None
            if dataset_exists(f, c_path):
                ds = f[c_path]
                if i < ds.shape[0]:
                    c_img = np.array(ds[i])
            if dataset_exists(f, d_path):
                ds = f[d_path]
                if i < ds.shape[0]:
                    d_img = np.array(ds[i])
                    d_img = normalize_depth_to_uint8(d_img)

            if c_img is not None:
                cam_frames_color.append(c_img)
            if d_img is not None:
                cam_frames_depth.append(d_img)

        # Merge all cams of same modality side-by-side per frame
        def merge_list(frames_list):
            if not frames_list:
                return None
            out = frames_list[0]
            for k in range(1, len(frames_list)):
                out = concat_side_by_side(out, frames_list[k])
            return out

        merged_color = merge_list(cam_frames_color)
        merged_depth = merge_list(cam_frames_depth)

        if merged_color is not None:
            color_frames_all.append(merged_color)
        if merged_depth is not None:
            depth_frames_all.append(merged_depth)

        # If both color and depth exist and lengths match, also build sbs of color|depth
        if merged_color is not None and merged_depth is not None:
            color_sbs.append(concat_side_by_side(merged_color, merged_depth))

    # Write videos
    if color_frames_all:
        write_video(color_frames_all, os.path.join(out_dir, "color.mp4"), fps=fps)
    if depth_frames_all:
        write_video(depth_frames_all, os.path.join(out_dir, "depth.mp4"), fps=fps)
    if color_sbs:
        write_video(color_sbs, os.path.join(out_dir, "color_depth_sbs.mp4"), fps=fps)


def plot_and_save_joints(f: h5py.File, out_png: str) -> None:
    if not dataset_exists(f, "robot/slave/joint"):
        return
    joints = np.array(f["robot/slave/joint"])  # (N, 6)
    t = None
    if dataset_exists(f, "timestamps/sample_time"):
        t = np.array(f["timestamps/sample_time"])  # (N,)
        if t.shape[0] != joints.shape[0]:
            t = None

    plt.figure(figsize=(10, 4))
    num_j = joints.shape[1] if joints.ndim == 2 else 0
    x = t if t is not None else np.arange(joints.shape[0])
    for j in range(num_j):
        plt.plot(x, joints[:, j], label=f"J{j+1}")
    plt.xlabel("time" if t is not None else "index")
    plt.ylabel("joint")
    if num_j <= 12:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_and_save_gripper(f: h5py.File, out_png: str) -> None:
    if not dataset_exists(f, "robot/slave/gripper_state"):
        return
    g = np.array(f["robot/slave/gripper_state"]).astype(np.int32)
    t = None
    if dataset_exists(f, "timestamps/sample_time"):
        t = np.array(f["timestamps/sample_time"])  # (N,)
        if t.shape[0] != g.shape[0]:
            t = None
    x = t if t is not None else np.arange(g.shape[0])
    plt.figure(figsize=(10, 2.8))
    plt.step(x, g, where="post", label="gripper (0=open,1=close)")
    plt.ylim(-0.2, 1.2)
    plt.yticks([0, 1])
    plt.xlabel("time" if t is not None else "index")
    plt.ylabel("state")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_and_save_pose_xyz(f: h5py.File, out_png: str) -> None:
    if not dataset_exists(f, "robot/slave/pose"):
        return
    pose = np.array(f["robot/slave/pose"])  # (N, 6) expected
    if pose.ndim != 2 or pose.shape[1] < 3:
        return
    t = None
    if dataset_exists(f, "timestamps/sample_time"):
        t = np.array(f["timestamps/sample_time"])  # (N,)
        if t.shape[0] != pose.shape[0]:
            t = None
    x = t if t is not None else np.arange(pose.shape[0])
    plt.figure(figsize=(10, 4))
    labels = ["x", "y", "z"]
    for j in range(3):
        plt.plot(x, pose[:, j], label=labels[j])
    plt.xlabel("time" if t is not None else "index")
    plt.ylabel("position")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize latest HDF5: summary, videos, joints plot")
    parser.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "save"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    latest = find_latest_h5(args.save_dir)
    base = os.path.splitext(os.path.basename(latest))[0]
    run_dir = os.path.join(args.out_dir, base)
    ensure_dir(run_dir)

    with h5py.File(latest, "r") as f:
        # summary
        write_summary_txt(f, os.path.join(run_dir, "summary.txt"))
        # videos
        export_videos(f, os.path.join(run_dir, "videos"), fps=args.fps)
        # joints plot
        plot_and_save_joints(f, os.path.join(run_dir, "joints.png"))
        # gripper plot
        plot_and_save_gripper(f, os.path.join(run_dir, "gripper.png"))
        # pose xyz plot
        plot_and_save_pose_xyz(f, os.path.join(run_dir, "pose_xyz.png"))

    print(f"Done. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()


