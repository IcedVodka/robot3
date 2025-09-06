# 修改后的 HDF5 与视频输出格式示例

## 数据结构概览（HDF5）

```
grasp_20250904_103419.h5
├── 属性 (Attributes)
│   ├── task_name: "grasp"
│   ├── task_description: "xxxxxxxxx"│  
│   ├── created_at: "20250904_103419"
│   ├── depth_alpha: 0.03  # 深度图转换为8位时的缩放参数
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
- realsense1_depth.mp4（深度图使用cv2.convertScaleAbs转换为8bit，alpha=0.03）
- realsense2_color.mp4
- realsense2_depth.mp4（深度图使用cv2.convertScaleAbs转换为8bit，alpha=0.03）

说明：视频文件与 HDF5 位于同一运行输出目录，帧率 `fps = save_freq`。深度视频使用cv2.convertScaleAbs进行线性缩放转换，alpha参数可在task_condition中配置。

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
- 顶层属性包含 `realsense1`/`realsense2`（JSON 格式内参）和 `depth_alpha`（深度转换参数）

## 深度图处理说明

深度图使用 `cv2.convertScaleAbs` 方法进行线性缩放转换：

```python
# 深度转换公式
depth_8bit = cv2.convertScaleAbs(depth, alpha=depth_alpha)
# 其中 depth_alpha 默认为 0.03，可在 task_condition 中配置
```

- **输入**：原始16位深度数据（单位：毫米）
- **输出**：8位灰度图像（范围：0-255）
- **转换方式**：线性缩放 `result = |input * alpha|`
- **配置参数**：`depth_alpha` 在 HDF5 元信息中保存，便于复现

## 数据同步

- 每个时间步 t_i 保存：机器人关节、位姿、手爪状态与时间戳
- 相机图像仅写入视频文件，不进入 HDF5

## 标注后的数据结构（带标签的HDF5）

使用 `utils/label_hdf5_data.py` 脚本对视频进行标注后，会生成带 `_labeled.h5` 后缀的HDF5文件，包含额外的标注数据：

```
grasp_20250904_103419_labeled.h5
├── 属性 (Attributes) - 与原始文件相同
│   ├── task_name: "grasp"
│   ├── task_description: "xxxxxxxxx"
│   ├── created_at: "20250904_103419"
│   ├── depth_alpha: 0.03
│   ├── realsense1: "{...}"
│   └── realsense2: "{...}"
│
├── robot/ - 与原始文件相同
│   ├── joint: (N, Dq)
│   ├── pose: (N, 6)
│   └── gripper_state: (N,)
│
├── timestamps/ - 与原始文件相同
│   └── sample_time: (N,)
│
└── label/ - 新增标注数据组
    ├── realsense1/
    │   ├── heatmap: (H, W) - 热力图，float32类型
    │   ├── label_map: (H, W) - 标签图，uint8类型，值范围0-5
    │   └── annotations: (M, 3) - 标注点数组，结构化数组
    │       ├── x: int32 - 像素x坐标
    │       ├── y: int32 - 像素y坐标
    │       └── label: int32 - 标签值(0-5)
    │
    └── realsense2/
        ├── heatmap: (H, W) - 热力图，float32类型
        ├── label_map: (H, W) - 标签图，uint8类型，值范围0-5
        └── annotations: (M, 3) - 标注点数组，结构化数组
            ├── x: int32 - 像素x坐标
            ├── y: int32 - 像素y坐标
            └── label: int32 - 标签值(0-5)
```

### 标注数据说明

- **heatmap**: 高斯热力图，基于标注点生成，用于可视化标注区域
- **label_map**: 像素级标签图，每个像素对应一个标签值(0-5)
- **annotations**: 原始标注点列表，包含坐标和标签信息
- **标签值含义**: 0-5 分别代表不同的语义类别（具体含义由用户定义）

### 标注图片输出

标注过程中会生成以下图片文件：
- `realsense1_heatmap_overlay.jpg` - realsense1热力图叠加原图
- `realsense2_heatmap_overlay.jpg` - realsense2热力图叠加原图

## 读取示例

### 原始HDF5文件读取

```python
import h5py
import json

with h5py.File('grasp_20250904_103419.h5', 'r') as f:
    # 读取属性
    task_name = f.attrs.get('task_name', '')
    depth_alpha = f.attrs.get('depth_alpha', 0.03)  # 深度转换参数
    intr1 = json.loads(f.attrs.get('realsense1', '{}'))
    intr2 = json.loads(f.attrs.get('realsense2', '{}'))

    # 读取数据集
    joints = f['robot/joint'][:]
    poses = f['robot/pose'][:]
    timestamps = f['timestamps/sample_time'][:]
    gripper_states = f['robot/gripper_state'][:]

    print(f"任务: {task_name}")
    print(f"深度转换参数: {depth_alpha}")
    print(f"数据长度: {len(timestamps)}, 关节形状: {joints.shape}, 位姿形状: {poses.shape}, 手爪形状: {gripper_states.shape}")
```

### 标注HDF5文件读取

```python
import h5py
import json
import numpy as np

with h5py.File('grasp_20250904_103419_labeled.h5', 'r') as f:
    # 读取原始数据（与原始文件相同）
    joints = f['robot/joint'][:]
    poses = f['robot/pose'][:]
    timestamps = f['timestamps/sample_time'][:]
    gripper_states = f['robot/gripper_state'][:]
    
    # 读取标注数据
    if 'label' in f:
        # realsense1标注数据
        rs1_heatmap = f['label/realsense1/heatmap'][:]
        rs1_label_map = f['label/realsense1/label_map'][:]
        rs1_annotations = f['label/realsense1/annotations'][:]
        
        # realsense2标注数据
        rs2_heatmap = f['label/realsense2/heatmap'][:]
        rs2_label_map = f['label/realsense2/label_map'][:]
        rs2_annotations = f['label/realsense2/annotations'][:]
        
        print(f"realsense1标注点数量: {len(rs1_annotations)}")
        print(f"realsense2标注点数量: {len(rs2_annotations)}")
        print(f"热力图形状: {rs1_heatmap.shape}")
        print(f"标签图形状: {rs1_label_map.shape}")
        
        # 打印标注点信息
        for i, (x, y, label) in enumerate(rs1_annotations):
            print(f"realsense1标注点{i}: ({x}, {y}) -> 标签{label}")
```

### 标注数据可视化

```python
import cv2
import numpy as np

# 读取标注数据
with h5py.File('grasp_20250904_103419_labeled.h5', 'r') as f:
    heatmap = f['label/realsense1/heatmap'][:]
    label_map = f['label/realsense1/label_map'][:]

# 热力图可视化
heatmap_normalized = (heatmap * 255 / heatmap.max()).astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

# 标签图可视化
label_colored = cv2.applyColorMap(
    (label_map * 255 / 5).astype(np.uint8), 
    cv2.COLORMAP_HSV
)

cv2.imshow('Heatmap', heatmap_colored)
cv2.imshow('Label Map', label_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
