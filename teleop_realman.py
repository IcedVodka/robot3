import time
import serial

# 串口配置
SERIAL_PORT = "/dev/ttyUSB0"  # 根据实际设备修改
BAUDRATE = 9600
TIMEOUT = 1


def main() -> None:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=TIMEOUT)
        print(f"已打开串口: {ser.name}, 波特率: {BAUDRATE}")
    except Exception as e:
        print(f"打开串口失败: {e}")
        return

    last_ts = time.time()
    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            if "：" not in raw:
                continue

            angle_str = raw.split("：")[-1]
            angle_list = []
            for x in angle_str.split(","):
                s = x.strip()
                if not s:
                    continue
                try:
                    angle_list.append(int(s))
                except ValueError:
                    continue


            now = time.time()
            dt = max(now - last_ts, 1e-6)
            fps = 1.0 / dt
            last_ts = now

            print(f"angle_list={angle_list} | fps={fps:.1f}")
    except KeyboardInterrupt:
        print("已停止")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
