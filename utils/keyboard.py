import sys

try:
    import termios
    import tty
    import select
except Exception:
    termios = None
    tty = None
    select = None

import threading
import queue
import time
from typing import Tuple

def kb_setup():
    """
    设置终端到 cbreak 模式，返回 (fd, old_settings)。
    若当前环境不支持（非类 Unix），返回 None。
    """
    if termios is None or tty is None:
        return None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return (fd, old_settings)


def kb_restore(kb_state):
    """
    恢复终端设置。
    """
    if kb_state is None or termios is None:
        return
    fd, old_settings = kb_state
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        pass


def kb_read_char():
    """
    非阻塞读取一个字符，如无输入返回 None。
    """
    if select is None:
        return None
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    except Exception:
        return None
    return None


def start_kb_listener() -> Tuple[threading.Thread, threading.Event, "queue.Queue[str]"]:
    """
    启动一个线程监听键盘，返回 (thread, stop_event, queue)。
    - queue: 按键字符队列，非阻塞读取
    - stop_event: 置位后线程会退出
    """
    stop_event = threading.Event()
    q: "queue.Queue[str]" = queue.Queue()

    def _worker():
        kb_state = kb_setup()
        try:
            while not stop_event.is_set():
                ch = kb_read_char()
                if ch:
                    q.put(ch)
                else:
                    time.sleep(0.001)
        finally:
            kb_restore(kb_state)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t, stop_event, q


