from typing import List, Optional
import atexit
import numpy as np


class AngleLinearMapper:
    """
    线性映射工具：
    - 输入 `angles` 为 int 的列表
    - 将值从 [in_min, in_max] 线性映射到 [out_min, out_max]
    - 运行期间自动统计出现过的总体最小/最大值，并在进程结束时打印
    - 可选对输出做指数平滑以获得更平稳的控制
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
        reverse_mapping: List[bool] = None,
        smooth: bool = True,
        smooth_alpha: float = 0.2,
    ) -> None:
        if not (len(in_mins) == len(in_maxs) == len(out_mins) == len(out_maxs) == 6):
            raise ValueError("in_mins/in_maxs/out_mins/out_maxs 必须都是长度为 6 的列表")

        self.in_mins = list(in_mins)
        self.in_maxs = list(in_maxs)
        self.out_mins = list(out_mins)
        self.out_maxs = list(out_maxs)
        self.clip = clip
        self.round_output = round_output
        self.smooth = smooth
        self.smooth_alpha = float(smooth_alpha)
        
        # 处理 reverse_mapping 参数
        if reverse_mapping is None:
            self.reverse_mapping = [False] * 6  # 默认全部正向映射
        elif isinstance(reverse_mapping, bool):
            self.reverse_mapping = [reverse_mapping] * 6  # 单个布尔值应用到所有通道
        elif isinstance(reverse_mapping, list) and len(reverse_mapping) == 6:
            self.reverse_mapping = list(reverse_mapping)  # 6个布尔值的列表
        else:
            raise ValueError("reverse_mapping 必须是布尔值或长度为6的布尔列表")

        self._observed_min: List[Optional[int]] = [None] * 6
        self._observed_max: List[Optional[int]] = [None] * 6
        
        # 存储所有观察到的唯一值，用于计算去除极值后的范围
        self._unique_values: List[set] = [set() for _ in range(6)]

        # 平滑状态（每通道一个 EMA 状态）
        self._ema_values: List[Optional[float]] = [None] * 6

        # 参数校验
        if self.smooth:
            if not (0.0 < self.smooth_alpha <= 1.0):
                raise ValueError("smooth_alpha 必须在 (0, 1] 区间内")

        atexit.register(self._print_stats)

    def _update_observed(self, values: List[int]) -> None:
        if not values:
            return
        for i, v in enumerate(values[:6]):
            # 更新传统的最小最大值
            ov_min = self._observed_min[i]
            ov_max = self._observed_max[i]
            if ov_min is None or v < ov_min:
                self._observed_min[i] = v
            if ov_max is None or v > ov_max:
                self._observed_max[i] = v
            
            # 存储唯一值用于后续计算去除极值后的范围
            self._unique_values[i].add(v)

    def _map_value(self, x: float, idx: int) -> float:
        in_min = self.in_mins[idx]
        in_max = self.in_maxs[idx]
        out_min = self.out_mins[idx]
        out_max = self.out_maxs[idx]

        if in_max == in_min:
            y = float(out_min)
        else:
            ratio = (x - in_min) / (in_max - in_min)
            if self.reverse_mapping[idx]:
                # 反向映射：in_min -> out_max, in_max -> out_min
                y = out_max - ratio * (out_max - out_min)
            else:
                # 正向映射：in_min -> out_min, in_max -> out_max
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

            # 可选指数平滑（在输出空间进行）
            if self.smooth:
                prev = self._ema_values[i]
                if prev is None:
                    ema = y
                else:
                    a = self.smooth_alpha
                    ema = a * y + (1.0 - a) * prev
                self._ema_values[i] = ema
                y_out = ema
            else:
                y_out = y

            mapped.append(int(round(y_out)) if self.round_output else int(y_out))
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
        
        print("\n[AngleLinearMapper] 观察到的范围（去除最大10%和最小10%的唯一值）：")
        print("in_mins=[", end="")
        for i in range(6):
            if len(self._unique_values[i]) == 0:
                print("0", end="")
            else:
                # 将唯一值转换为排序后的数组
                unique_values = sorted(list(self._unique_values[i]))
                if len(unique_values) < 10:  # 数据太少时使用全部数据
                    mn = unique_values[0]
                else:
                    # 去除最大10%和最小10%的唯一值
                    remove_count = max(1, len(unique_values) // 10)  # 至少去除1个
                    filtered_values = unique_values[remove_count:-remove_count]
                    mn = filtered_values[0] if len(filtered_values) > 0 else unique_values[0]
                print(f"{mn}", end="")
            if i < 5:
                print(", ", end="")
        print("],")
        
        print("in_maxs=[", end="")
        for i in range(6):
            if len(self._unique_values[i]) == 0:
                print("1023", end="")
            else:
                # 将唯一值转换为排序后的数组
                unique_values = sorted(list(self._unique_values[i]))
                if len(unique_values) < 10:  # 数据太少时使用全部数据
                    mx = unique_values[-1]
                else:
                    # 去除最大10%和最小10%的唯一值
                    remove_count = max(1, len(unique_values) // 10)  # 至少去除1个
                    filtered_values = unique_values[remove_count:-remove_count]
                    mx = filtered_values[-1] if len(filtered_values) > 0 else unique_values[-1]
                print(f"{mx}", end="")
            if i < 5:
                print(", ", end="")
        print("],")
        
        print("out_mins=[", end="")
        for i in range(6):
            print(f"{self.out_mins[i]}", end="")
            if i < 5:
                print(", ", end="")
        print("],")
        
        print("out_maxs=[", end="")
        for i in range(6):
            print(f"{self.out_maxs[i]}", end="")
            if i < 5:
                print(", ", end="")
        print("],")


__all__ = ["AngleLinearMapper"]




if __name__ == "__main__":
    # 简单示例：演示如何使用 AngleLinearMapper 进行逐通道映射
    in_mins = [0, 0, 0, 0, 0, 0]
    in_maxs = [1023, 1023, 1023, 1023, 1023, 1023]
    out_mins = [0, 0, 0, 0, 0, 0]
    out_maxs = [180, 180, 180, 180, 180, 180]

    # 正向映射示例（所有通道）
    mapper = AngleLinearMapper(in_mins, in_maxs, out_mins, out_maxs, clip=True, round_output=True, reverse_mapping=False)
    
    print("正向映射示例 (0-1023 -> 0-180):")
    test_values = [0, 256, 512, 768, 1023]
    for val in test_values:
        mapped = mapper.map_angles([val] * 6)
        print(f"输入: {val} -> 输出: {mapped[0]}")
    
    print("\n反向映射示例 (0-1023 -> 180-0):")
    mapper_reverse = AngleLinearMapper(in_mins, in_maxs, out_mins, out_maxs, clip=True, round_output=True, reverse_mapping=True)
    for val in test_values:
        mapped = mapper_reverse.map_angles([val] * 6)
        print(f"输入: {val} -> 输出: {mapped[0]}")
    
    print("\n混合映射示例 (通道0,2,4反向，通道1,3,5正向):")
    mixed_reverse = [True, False, True, False, True, False]
    mapper_mixed = AngleLinearMapper(in_mins, in_maxs, out_mins, out_maxs, clip=True, round_output=True, reverse_mapping=mixed_reverse)
    test_input = [0, 256, 512, 768, 1023, 100]
    mapped = mapper_mixed.map_angles(test_input)
    print(f"输入: {test_input}")
    print(f"输出: {mapped}")
    print("通道映射方向:", mixed_reverse)
    
    print("\n" + "="*50)
    print("使用混合映射进行测试:")
    mapper = mapper_mixed

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