import pyrealsense2 as rs

def get_device_intrinsics(pipeline: rs.pipeline, width: int, height: int) -> dict:
    """
    接收一个已启动的管道，并返回相机的内参。
    """
    try:
        frames = pipeline.wait_for_frames(3000)
    except RuntimeError as e:
        print(f"Error waiting for frames: {e}")
        return {"color": None, "depth": None}

    color_intr = None
    depth_intr = None

    color_frame = frames.get_color_frame()
    if color_frame:
        color_profile = color_frame.get_profile()
        ci = color_profile.as_video_stream_profile().get_intrinsics()
        color_intr = {"ppx": ci.ppx, "ppy": ci.ppy, "fx": ci.fx, "fy": ci.fy}

    depth_frame = frames.get_depth_frame()
    if depth_frame:
        depth_profile = depth_frame.get_profile()
        di = depth_profile.as_video_stream_profile().get_intrinsics()
        depth_intr = {"ppx": di.ppx, "ppy": di.ppy, "fx": di.fx, "fy": di.fy}

    return {"color": color_intr, "depth": depth_intr}

def main() -> None:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("No RealSense device detected.")
        return

    # Sort by serial for stable naming
    device_infos = []
    for dev in devices:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            serial = "unknown"
        device_infos.append(serial)
    device_infos.sort()

    intr_outputs = []
    for idx, serial in enumerate(device_infos, start=1):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        
        try:
            # You must enable streams for each pipeline.
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start the pipeline for the current device
            pipeline.start(config)

            # Get intrinsics using the now-started pipeline
            intr = get_device_intrinsics(pipeline, 640, 480)
            
            name = f"realsense{idx}_intr"
            intr_outputs.append((name, serial, intr))
            
        finally:
            # Stop the pipeline for the current device
            pipeline.stop()
            
    for name, serial, intr in intr_outputs:
        print(f"{name.replace('_intr', '_serial')} = \"{serial}\"")
        print(f"{name} = {intr}")

if __name__ == "__main__":
    main()