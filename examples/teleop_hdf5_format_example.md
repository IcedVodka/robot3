# 修改后的 HDF5 与视频输出格式示例

## 数据结构概览（HDF5）

```
grasp_20250904_103419.h5
├── 属性 (Attributes)
│   ├── task_name: "grasp"
│   ├── task_description: "xxxxxxxxx"│  
│   ├── created_at: "20250904_103419"
│   ├── realsense1: "{...}"  # 相机1内参（JSON字符串）
│   └── realsense2: "{...}"  # 相机2内参（JSON字符串）
│
├── robot/
│   ├── joint: (N, Dq) - 关节角度
│   ├── pose: (N, 6)  - 位姿 [x, y, z, rx, ry, rz]
│   └── gripper_state: (N,) - 手爪状态 (0:开, 1:合)
│
└── timestamps/
    └── sample_time: (N,) - Unix时间戳 (float64)
```

说明：图像数据不再写入 HDF5，仅在内存缓存并最终导出为视频。

## 视频输出文件

- realsense1_color.mp4
- realsense1_depth.mp4（深度归一化到 8bit）
- realsense2_color.mp4
- realsense2_depth.mp4（深度归一化到 8bit）

说明：视频文件与 HDF5 位于同一运行输出目录，帧率 `fps = save_freq`。

## 机器人状态数据示例

```python
robot_state = {
    'joint': [22.084999084472656, -25.274999618530273, 103.55400085449219,
              20.600000381469727, 10.034000396728516, -1.2569999694824219],
    'pose': [-0.219004, -0.09839, 0.520461, 1.446, 1.238, -1.241],
    'err': {'err_len': 1, 'err': ['0']}
}
```

## HDF5 中保存的数据

- 关节数据集 `robot/joint`: 形状 (N, Dq)
- 位姿数据集 `robot/pose`: 形状 (N, 6)，[x, y, z, rx, ry, rz]
- 手爪状态 `robot/gripper_state`: 形状 (N,)
- 时间戳 `timestamps/sample_time`: 形状 (N,)
- 顶层属性包含 `realsense1`/`realsense2`（JSON 格式内参）

## 数据同步

- 每个时间步 t_i 保存：机器人关节、位姿、手爪状态与时间戳
- 相机图像仅写入视频文件，不进入 HDF5

## 读取示例

```python
import h5py
import json

with h5py.File('grasp_20250904_103419.h5', 'r') as f:
    # 读取属性
    task_name = f.attrs.get('task_name', '')
    intr1 = json.loads(f.attrs.get('realsense1', '{}'))
    intr2 = json.loads(f.attrs.get('realsense2', '{}'))

    # 读取数据集
    joints = f['robot/joint'][:]
    poses = f['robot/pose'][:]
    timestamps = f['timestamps/sample_time'][:]
    gripper_states = f['robot/gripper_state'][:]

    print(len(timestamps), joints.shape, poses.shape, gripper_states.shape)
```
