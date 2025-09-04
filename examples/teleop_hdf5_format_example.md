# 修改后的HDF5数据格式示例

## 数据结构概览

```
grasp_20250904_103419.h5
├── 属性 (Attributes)
│   ├── task_name: "grasp"
│   ├── task_description: "xxxxxxxxx"
│   ├── save_format: "hdf5"
│   └── created_at: "20250904_103419"
│
├── sensors/
│   ├── realsense1/
│   │   ├── color: (N, H, W, 3) - RGB图像数据
│   │   ├── depth: (N, H, W) - 深度图像数据
│   │   └── intrinsics/ (属性)
│   │       ├── color.ppx: 320.1
│   │       ├── color.ppy: 240.5
│   │       ├── color.fx: 600.2
│   │       ├── color.fy: 599.8
│   │       ├── depth.ppx: 320.0
│   │       ├── depth.ppy: 240.0
│   │       ├── depth.fx: 601.0
│   │       └── depth.fy: 600.0
│   │
│   └── realsense2/
│       ├── color: (N, H, W, 3) - RGB图像数据
│       ├── depth: (N, H, W) - 深度图像数据
│       └── intrinsics/ (属性)
│           ├── color.ppx: 319.8
│           ├── color.ppy: 241.0
│           ├── color.fx: 602.3
│           ├── color.fy: 601.1
│           ├── depth.ppx: 319.7
│           ├── depth.ppy: 240.2
│           ├── depth.fx: 603.0
│           └── depth.fy: 602.5
│
├── robot/
│   └── slave/
│       ├── joint: (N, 6) - 关节角度数据 [新增]
│       │   └── 示例: [22.085, -25.275, 103.554, 20.600, 10.034, -1.257]
│       ├── pose: (N, 6) - 位姿数据 [新增]
│       │   └── 示例: [-0.219004, -0.09839, 0.520461, 1.446, 1.238, -1.241]
│       │   └── 格式: [x, y, z, rx, ry, rz] (位置+旋转)
│       └── gripper_state: (N,) - 手爪状态
│           └── 0: 开, 1: 合
│
└── timestamps/
    └── sample_time: (N,) - 时间戳数据
        └── 格式: Unix时间戳 (float64)
```

## 数据示例

### 机器人状态数据示例
基于您提供的robot_state格式：
```python
robot_state = {
    'joint': [22.084999084472656, -25.274999618530273, 103.55400085449219, 
              20.600000381469727, 10.034000396728516, -1.2569999694824219],
    'pose': [-0.219004, -0.09839, 0.520461, 1.446, 1.238, -1.241],
    'err': {'err_len': 1, 'err': ['0']}
}
```

### HDF5中保存的数据
- **joint数据集**: 形状 (N, 6)，保存6个关节的角度值
- **pose数据集**: 形状 (N, 6)，保存6DOF位姿 [x, y, z, rx, ry, rz]

### 数据同步
所有数据按时间戳同步保存：
- 每个时间步t_i对应：
  - 2个相机的RGB和深度图像
  - 机器人的关节角度和位姿
  - 手爪状态
  - 时间戳

## 主要修改内容

1. **新增extract_pose函数**: 从robot_state中提取pose数据
2. **扩展_init_datasets方法**: 创建pose数据集
3. **更新collect方法**: 同时保存joint和pose数据
4. **数据格式**: pose数据为6维向量，表示[x, y, z, rx, ry, rz]

## 使用示例

```python
import h5py

# 读取HDF5文件
with h5py.File('grasp_20250904_103419.h5', 'r') as f:
    # 读取关节数据
    joints = f['robot/slave/joint'][:]  # 形状: (N, 6)
    
    # 读取位姿数据 [新增]
    poses = f['robot/slave/pose'][:]    # 形状: (N, 6)
    
    # 读取时间戳
    timestamps = f['timestamps/sample_time'][:]  # 形状: (N,)
    
    # 读取手爪状态
    gripper_states = f['robot/slave/gripper_state'][:]  # 形状: (N,)
    
    print(f"数据点数量: {len(timestamps)}")
    print(f"关节数据形状: {joints.shape}")
    print(f"位姿数据形状: {poses.shape}")
    
    # 查看第一帧数据
    print(f"第一帧关节: {joints[0]}")
    print(f"第一帧位姿: {poses[0]}")
```
