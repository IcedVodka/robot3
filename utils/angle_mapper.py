from typing import List, Optional
import atexit


class AngleLinearMapper:
    """
    线性映射工具：
    - 输入 `angles` 为 int 的列表
    - 将值从 [in_min, in_max] 线性映射到 [out_min, out_max]
    - 运行期间自动统计出现过的总体最小/最大值，并在进程结束时打印
    """

    def __init__(
        self,
        in_mins: List[int],
        in_maxs: List[int],
        out_mins: List[int],
        out_maxs: List[int],
        *,
        clip: bool = True,
        round_output: bool = True,
    ) -> None:
        if not (len(in_mins) == len(in_maxs) == len(out_mins) == len(out_maxs) == 6):
            raise ValueError("in_mins/in_maxs/out_mins/out_maxs 必须都是长度为 6 的列表")

        self.in_mins = list(in_mins)
        self.in_maxs = list(in_maxs)
        self.out_mins = list(out_mins)
        self.out_maxs = list(out_maxs)
        self.clip = clip
        self.round_output = round_output

        self._observed_min: List[Optional[int]] = [None] * 6
        self._observed_max: List[Optional[int]] = [None] * 6

        atexit.register(self._print_stats)

    def _update_observed(self, values: List[int]) -> None:
        if not values:
            return
        for i, v in enumerate(values[:6]):
            ov_min = self._observed_min[i]
            ov_max = self._observed_max[i]
            if ov_min is None or v < ov_min:
                self._observed_min[i] = v
            if ov_max is None or v > ov_max:
                self._observed_max[i] = v

    def _map_value(self, x: float, idx: int) -> float:
        in_min = self.in_mins[idx]
        in_max = self.in_maxs[idx]
        out_min = self.out_mins[idx]
        out_max = self.out_maxs[idx]

        if in_max == in_min:
            y = float(out_min)
        else:
            ratio = (x - in_min) / (in_max - in_min)
            y = out_min + ratio * (out_max - out_min)
        if self.clip:
            low = min(out_min, out_max)
            high = max(out_min, out_max)
            if y < low:
                y = low
            elif y > high:
                y = high
        return y

    def map_angles(self, angles: List[int]) -> List[int]:
        """对一组角度做线性映射（逐通道使用独立区间），并更新统计。返回 int 列表。"""
        self._update_observed(angles)
        mapped: List[int] = []
        for i, v in enumerate(angles[:6]):
            y = self._map_value(float(v), i)
            mapped.append(int(round(y)) if self.round_output else int(y))
        return mapped

    def report(self) -> None:
        """手动打印统计信息。"""
        self._print_stats()

    def _print_stats(self) -> None:
        any_observed = any(v is not None for v in self._observed_min) or any(
            v is not None for v in self._observed_max
        )
        if not any_observed:
            print("[AngleLinearMapper] 本次未收到任何 angles 数据。")
            return
        print("[AngleLinearMapper] 本次运行每通道观察到的最小/最大值：")
        for i in range(6):
            mn = self._observed_min[i]
            mx = self._observed_max[i]
            if mn is None or mx is None:
                print(f"  ch{i}: 无数据")
            else:
                print(f"  ch{i}: min={mn}, max={mx}")


__all__ = ["AngleLinearMapper"]




if __name__ == "__main__":
    # 简单示例：演示如何使用 AngleLinearMapper 进行逐通道映射
    in_mins = [0, 0, 0, 0, 0, 0]
    in_maxs = [1023, 1023, 1023, 1023, 1023, 1023]
    out_mins = [0, 0, 0, 0, 0, 0]
    out_maxs = [180, 180, 180, 180, 180, 180]

    mapper = AngleLinearMapper(in_mins, in_maxs, out_mins, out_maxs, clip=True, round_output=True)

    # 假设从某处读取到的六通道 angles（此处用固定示例代替）
    demo_angles_list = [
        [10, 200, 350, 500, 700, 900],
        [0, 1023, 512, 256, 768, 100],
        [123, 456, 789, 1000, 50, 1023],
    ]

    for angles in demo_angles_list:
        mapped = mapper.map_angles(angles)
        print(f"input={angles}  =>  mapped={mapped}")

    # 进程结束时会自动打印每通道最小/最大值；也可以手动：
    mapper.report()