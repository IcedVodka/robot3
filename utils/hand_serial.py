import threading
from collections import deque
from typing import List, Optional

import serial

# 串口配置（根据实际设备修改）
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 9600
TIMEOUT = 1


class SerialAngleReader:
    """后台线程监听串口，非阻塞获取最新角度数据。"""

    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUDRATE, timeout: int = TIMEOUT) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._buffer: deque[List[int]] = deque(maxlen=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ser: Optional[serial.Serial] = None

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass

    def get_latest(self) -> Optional[List[int]]:
        """非阻塞返回当前最新的角度列表；若暂时无数据则返回 None。"""
        if not self._buffer:
            return None
        return list(self._buffer[-1])

    def _run(self) -> None:
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except Exception:
            # 打开失败直接退出线程，保持简单
            return

        while not self._stop_event.is_set():
            try:
                raw = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not raw or "：" not in raw:
                continue

            angle_str = raw.split("：")[-1]
            angle_list: List[int] = []
            for x in angle_str.split(","):
                s = x.strip()
                if not s:
                    continue
                try:
                    angle_list.append(int(s))
                except ValueError:
                    continue

            if angle_list:
                self._buffer.append(angle_list)


# 简单示例：作为脚本运行时打印最新数据
if __name__ == "__main__":
    import time

    reader = SerialAngleReader()
    reader.start()
    try:
        while True:
            angles = reader.get_latest()
            if angles is not None:
                print(f"angle_list={angles}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
