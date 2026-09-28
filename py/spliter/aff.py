"""Arcaea 谱面（``.aff``）的解析、坐标换算与回写。

本模块负责 AFF 特有的三件事：

1. **解析**：``timing`` / ``(t,lane)`` / ``hold(t1,t2,lane)`` / ``arc(...)`` / ``arctap(t)``；
2. **坐标统一**：地面音符的整数轨道换算成与 Arc 同域的浮点横坐标；
3. **回写**：给地面音符补一个手标记参数（``0`` = 左手 / ``1`` = 右手）。
   **arc 一律不修改** —— arc 的第 8 个参数 ``color`` 本身就是它的手标记。

解析结果放进与 :mod:`spliter.chart` 相同的物件模型（:class:`~spliter.chart.Note`
的 ``time`` / ``x`` / ``type`` / ``params`` / ``side`` 加 :class:`~spliter.chart.Chart`），
所以 :func:`spliter.split.split` 不需要知道谱面来自哪个格式。

坐标换算（依据 Arcaea 中文维基「坐标与布局规范」）
--------------------------------------------------
轨道编号 ``lane``（标准模式 1–4）与 Arc 横坐标 ``x`` 的关系为::

    x = LANE_ORIGIN + (lane - 1) * LANE_STEP
      = -0.25 + (lane - 1) * 0.5

即 ``lane 1 -> -0.25``、``lane 2 -> 0.25``、``lane 3 -> 0.75``、``lane 4 -> 1.25``，
**步长 0.5**，四条轨道关于判定面中线 ``x = 0.5`` 对称。

（历史坑：早期版本错写成 ``x = (lane-1)/4 + 0.25``，即 0.25/0.50/0.75/1.00、
步长 0.25、中线 0.625。那会让整块场地偏移并缩窄一半，是"整段左右翻转"那类
bug 的源头。以 :data:`LANE_ORIGIN` / :data:`LANE_STEP` 为准，不要照抄旧公式。）

Arc 横坐标的标准范围是 ``[-0.5, 1.5]``（判定面中线 ``0.5``）；天空音符可以
延伸到 ``0`` 以下和 ``1`` 以上（那是空中轨道），地面轨道只占中间一段。

纵坐标 ``y``：``0`` 是地面判定线，``1`` 是天空顶部。地面音符恒在 ``y = 0``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .chart import DEFAULT_META, Chart, Note, sort_notes

__all__ = [
    "AffArc",
    "AffArctap",
    "AffChart",
    "EASINGS",
    "LANE_CENTER",
    "SIX_LANE_SCALE_X",
    "SIX_LANE_SCALE_Y",
    "ScaleSchedule",
    "ScaleSegment",
    "SceneControl",
    "apply_arcaea_physics",
    "arc_x_at",
    "build_scale_schedule",
    "lane_to_x",
    "parse_aff",
    "serialize_aff",
    "x_to_lane",
]

#: 手标记取值：0 = 左手，1 = 右手（与 arc 的 color 参数同一套）。
HAND_LEFT = 0
HAND_RIGHT = 1

#: 地面轨道 -> Arc 横坐标。**每条轨道宽 0.5 个单位**，4 条轨道共 5 条边界，
#: 依次是 ``-0.5 / 0 / 0.5 / 1.0 / 1.5``；而 tap/hold 落在轨道**中点**上：
#:
#:     lane1 [-0.5, 0.0]  中点 -0.25
#:     lane2 [ 0.0, 0.5]  中点  0.25
#:     lane3 [ 0.5, 1.0]  中点  0.75
#:     lane4 [ 1.0, 1.5]  中点  1.25
#:
#: 即 ``x = -0.25 + (lane - 1) * 0.5``。
#:
#: **左右手拆分的中心是这套坐标下的 0.5**（2 轨与 3 轨之间那条边），**不是 0**。
#: 官方对 arctap 的判定区间 ``x∈[-0.5, 1.5]`` 正好就是这 4 条轨道的左右边界
#: （场地总宽 2.0 个单位）。
#:
#: 6k（``enwidenlanes`` 在 1 轨左侧加 0 轨、4 轨右侧加 5 轨）时轨道等宽 0.5，
#: 于是 7 条边界是 ``-1 / -0.5 / 0 / 0.5 / 1 / 1.5 / 2``，
#: 6 个中点是 ``-0.75 / -0.25 / 0.25 / 0.75 / 1.25 / 1.75``；
#: 因为 0/5 轨是左右对称加出去的，**中线仍然是 0.5**。
LANE_ORIGIN = -0.25         # lane 1 中点的 x
LANE_STEP = 0.5             # 轨道宽度（= 相邻轨道间距）
LANE_CENTER = 0.5           # 左右手拆分的中心（场地真中线）


def lane_to_x(lane: float) -> float:
    """轨道编号 -> Arc 横坐标（浮点）。``lane 1`` -> ``-0.25``，``lane 4`` -> ``1.25``。"""
    return LANE_ORIGIN + (float(lane) - 1.0) * LANE_STEP


def x_to_lane(x: float) -> float:
    """:func:`lane_to_x` 的逆运算。"""
    return 1.0 + (float(x) - LANE_ORIGIN) / LANE_STEP


# --------------------------------------------------------------------------- #
# 缓动曲线
# --------------------------------------------------------------------------- #

def _clamp01(t: float) -> float:
    return 0.0 if t < 0 else (1.0 if t > 1 else t)


def _si(t: float) -> float:      # Sine Out：先快后慢
    return 1.0 - (1.0 - t) ** 2


def _so(t: float) -> float:      # Sine In：先慢后快
    return t * t


def _axis_ease(kind: str, axis: int, t: float) -> float:
    """取缓动在某个轴上的分量：``axis=0`` 用前两字母，``axis=1`` 用后两字母。"""
    kind = (kind or "s").lower()
    if len(kind) == 4:
        part = kind[:2] if axis == 0 else kind[2:]
    elif len(kind) == 2:
        part = kind
    else:
        part = "s"
    lookup = {"b": lambda v: v, "s": lambda v: v, "si": _si, "so": _so}
    return lookup.get(part, lambda v: v)(_clamp01(t))


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class AffArc:
    """一条音弧。``color`` 同时就是它的手标记（0 = 左手 / 1 = 右手）。"""

    t1: int
    t2: int
    x1: float
    x2: float
    easing: str
    y1: float
    y2: float
    color: int
    hitsound: str = "none"
    arctype: str = "false"          # false = 实体音弧（有颜色） / true = 黑线
    smoothness: float | None = None
    arctaps: list[int] = field(default_factory=list)
    line_no: int = 0
    raw: str = ""

    @property
    def is_trace(self) -> bool:
        """黑线（``arctype != false``）：纯演出音轨，不产生判定、没有手标记语义。"""
        return self.arctype != "false"

    @property
    def is_solid(self) -> bool:
        """实体音弧：color 有效，且"区间内不得同手"的约束只对它成立。"""
        return self.arctype == "false"

    def x_at(self, time_ms: float) -> float:
        """这条弧在某个时刻的横坐标（``arctap`` 定位用）。"""
        return arc_x_at(self, time_ms)

    def y_at(self, time_ms: float) -> float:
        """这条弧在某个时刻的纵坐标。"""
        span = self.t2 - self.t1
        if span == 0:
            return self.y2
        t = (time_ms - self.t1) / span
        return self.y1 + (self.y2 - self.y1) * _axis_ease(self.easing, 1, t)


@dataclass
class AffArctap:
    """天空音符：挂在某条弧上的一个时间戳，坐标由所属弧算出。

    ``hand`` 是**已有的**手标记（``None`` 表示还没标）。标记写法是
    ``arctap(t,0)`` / ``arctap(t,1)``，与地面音符的 ``0/1`` 同一套。
    """

    time: int
    arc: AffArc
    line_no: int = 0
    hand: int | None = None
    raw_text: str = ""

    @property
    def is_marked(self) -> bool:
        return self.hand is not None


@dataclass
class SceneControl:
    """一条 ``scenecontrol`` 事件（只保留本项目关心的字段）。

    当前的 ``enwidencamera`` / ``enwidenlanes`` 用前两个参数决定
    "做什么" 与 "何时切换"；其余 scenecontrol 类型也一并解析出来备用
    （不影响任何计算）。
    """

    time: int
    kind: str                       # "camera" / "lanes" / 其它原始名
    enable: bool                    # param2：1 = 启用，0 = 关闭
    duration: float = 0.0           # param1：过渡时长（ms），当前不参与计算
    line: int = 0
    raw: str = ""


@dataclass
class AffChart:
    """一份解析后的 AFF 谱面。"""

    chart: Chart                                # 统一的物件模型（可直接喂给 split）
    arcs: list[AffArc] = field(default_factory=list)
    arctaps: list[AffArctap] = field(default_factory=list)
    timings: list[tuple[int, float, float]] = field(default_factory=list)
    audio_offset: int = 0
    density_factor: float = 1.0
    header: dict[str, str] = field(default_factory=dict)
    newline: str = "\n"
    lines: list[str] = field(default_factory=list)   # 原始行，回写用
    warnings: list[str] = field(default_factory=list)
    scenecontrols: list[SceneControl] = field(default_factory=list)

    @property
    def duration(self) -> int:
        return self.chart.duration

    def scale_schedule(self) -> ScaleSchedule:
        """这份谱面的分段刻度表（六轨模式）。"""
        return build_scale_schedule(self.scenecontrols, self.duration)

    def ground_notes(self) -> list[Note]:
        """地面物件（tap / hold）。"""
        return [n for n in self.chart.notes if n.type in ("tap", "hold")]

    def arctap_notes(self) -> list[Note]:
        """天空物件。"""
        return [n for n in self.chart.notes if n.type == "arctap"]

    def solid_arcs(self) -> list[AffArc]:
        """实体音弧（有颜色 = 有手标记）。"""
        return [a for a in self.arcs if a.is_solid]


#: 手标记在各类地面音符里的**参数下标**（0 基）。
_HAND_INDEX = {"tap": 2, "hold": 3}

#: 地面音符"还没有手标记"时的参数个数。
_EXPECTED_ARGS = {"tap": 2, "hold": 3}

#: 屏幕像素刻度（依据游戏内截图实测，见 tests/test_aff.py 的标定用例）。
#:
#: 参照物：判地面上「lane1 中点 ~ lane4 中点」的水平距离 = **1.5 个横坐标单位**
#: （-0.25 ~ 1.25），以及「y = 0 ~ y = 1」的竖直距离 = 1 个纵坐标单位。
#: 用像素量出这段矩形约为 804 × 326 px，于是::
#:
#:     x: 804 / 1.5 = 536 px/单位
#:     y: 326 / 1.00 = 326 px/单位
#:
#: **两个轴的刻度不同**（判定面是斜视的 3D 平面），所以位移必须分轴换算后
#: 再用勾股定理合成，不能只算一个轴。校验：相邻轨道相差 1 个轨道宽 = 0.5 单位
#: = 268px，与游戏内实测的轨道间距一致。
PIXELS_PER_UNIT_X = 536.0
PIXELS_PER_UNIT_Y = 326.0

#: 兼容旧名字（之前只用一个"单位=640px"的粗略换算）。
ARC_UNIT_IN_PX = PIXELS_PER_UNIT_X
# Arcaea 的实测取值已经**固化在 config.py 的默认值里**（标定理由见 README 参数表），
# 这里只留下与像素刻度相关的几何常量和一个设置函数。

#: 起手手位离中线的距离（px）。
HOME_OFFSET_PX = 80.0

#: 六轨模式（``enwidenlanes`` + ``enwidencamera``）维持期间的刻度缩放。
#:
#: 依据（用户口径 + 官方事件说明）：
#:
#: - ``enwidencamera`` 把 Sky Input 判定线抬到 ``y = 1.61`` —— 即**新的 y=1.61
#:   对应原来的 y=1.0**，所以 y 刻度乘 ``1 / 1.61``；
#: - ``enwidenlanes`` 在 1 轨左侧加 0 轨、4 轨右侧加 5 轨，轨道仍等宽 0.5。
#:   0 轨左边界 = 原 1 轨左边界（``-1.0`` ↔ ``-0.5``）、
#:   5 轨右边界 = 原 4 轨右边界（``+2.0`` ↔ ``+1.5``）——
#:   也就是**同样一块屏幕宽度现在要装下 3.0 个弧单位（原来 2.0 个）**，
#:   于是 x 刻度乘 ``2 / 3``（**位移一律缩小 1.5 倍**）。
#:
#: 轨道坐标本身**不用变**：:func:`lane_to_x` 的线性外推已经给出
#: ``0 轨 → -0.75``、``5 轨 → +1.75``，与实测谱面解析结果一致。
SIX_LANE_SCALE_X = 2.0 / 3.0
SIX_LANE_SCALE_Y = 1.0 / 1.61


@dataclass(frozen=True)
class ScaleSegment:
    """一段"维持某个像素刻度"的时间区间 ``[start, end)``。"""

    start: int
    end: int
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class ScaleSchedule:
    """按时间分段的**像素刻度倍率**（1.0 = 标准四轨）。

    六轨模式在同一条谱面里会来回切换，所以刻度不能再是全局常量。
    这里只记录**稳态**区间：过渡（淡入/淡出）期间刻度在插值，
    本实现按"切换时刻立即生效"的保守近似处理（见 :func:`build_scale_schedule`）。

    ``segments`` 按 ``start`` 升序、互不重叠。
    """

    segments: list[ScaleSegment] = field(default_factory=list)

    def at(self, time_ms: float) -> tuple[float, float]:
        """``time_ms`` 时刻的 ``(scale_x, scale_y)``；不在任何区间内返回 ``(1, 1)``。"""
        for segment in self.segments:          # 段数极少（通常 1~2），线性足够
            if segment.start <= time_ms < segment.end:
                return segment.scale_x, segment.scale_y
        return 1.0, 1.0

    def is_enabled(self, time_ms: float) -> bool:
        """该时刻是否处于扩展视野状态（用于报告/调试）。"""
        scale_x, scale_y = self.at(time_ms)
        return scale_x != 1.0 or scale_y != 1.0

    def __bool__(self) -> bool:
        return bool(self.segments)


def build_scale_schedule(events: Iterable["SceneControl"],
                         duration: int) -> ScaleSchedule:
    """把 ``enwidencamera`` / ``enwidenlanes`` 事件折成刻度区间表。

    事件语义（官方）：``param1`` = 过渡时长 ms，``param2`` = 1 启用 / 0 关闭。

    处理口径：

    - **只跟踪稳态**：``x`` 刻度由 ``enwidenlanes`` 控制、``y`` 刻度由
      ``enwidencamera`` 控制，两者**独立**（实测成对出现，但不假设一定成对）；
    - **过渡时长被忽略**（用户明确说明"扩展过程中的坐标变换太复杂，不需要计算"）：
      按事件时刻**瞬时切换**。真实谱面的维持段重发 ``param1 = 0.00``，瞬时切换
      本来就是精确的；只有淡入/淡出那一小段（实测 1333ms / 5333ms）不精确；
    - 结束时刻用**后续的关闭事件**；到谱面结束仍未关闭则延到 ``duration``。
    """
    cams = sorted((e for e in events if e.kind == "camera"), key=lambda e: e.time)
    lanes = sorted((e for e in events if e.kind == "lanes"), key=lambda e: e.time)

    def spans(items: list["SceneControl"]) -> list[tuple[int, int]]:
        """把 开/关 事件序列折成 ``[(start, end), ...]``（未闭合则延到 duration）。"""
        result: list[tuple[int, int]] = []
        opened_at: int | None = None
        for event in items:
            if event.enable:
                if opened_at is None:
                    opened_at = event.time
                # 已经在开启状态时的重复"开启"= 维持，不重开
            else:
                if opened_at is not None:
                    if event.time > opened_at:
                        result.append((opened_at, event.time))
                    opened_at = None
        if opened_at is not None:
            result.append((opened_at, max(duration, opened_at + 1)))
        return result

    camera_spans = spans(cams)
    lane_spans = spans(lanes)
    if not camera_spans and not lane_spans:
        return ScaleSchedule()

    # 把所有边界点收集起来切成互不重叠的小段，每段查一次"此刻哪个开关是开的"。
    bounds = {0, max(duration, 1)}
    for spans_ in (camera_spans, lane_spans):
        for start, end in spans_:
            bounds.add(start)
            bounds.add(end)
    ordered = sorted(bounds)

    segments: list[ScaleSegment] = []
    for start, end in zip(ordered, ordered[1:]):
        cam_on = any(lo <= start < hi for lo, hi in camera_spans)
        lane_on = any(lo <= start < hi for lo, hi in lane_spans)
        scale_x = SIX_LANE_SCALE_X if lane_on else 1.0
        scale_y = SIX_LANE_SCALE_Y if cam_on else 1.0
        if scale_x == 1.0 and scale_y == 1.0:
            continue                            # 标准段不记录（at() 默认即 1,1）
        # 与上一段刻度相同的直接合并，避免碎片
        if segments and (segments[-1].scale_x, segments[-1].scale_y) == (scale_x, scale_y) \
                and segments[-1].end == start:
            segments[-1] = ScaleSegment(segments[-1].start, end, scale_x, scale_y)
            continue
        segments.append(ScaleSegment(start, end, scale_x, scale_y))
    return ScaleSchedule(segments)


def apply_arcaea_physics(config, unit_px_x: float = PIXELS_PER_UNIT_X,
                         unit_px_y: float = PIXELS_PER_UNIT_Y,
                         schedule: "ScaleSchedule | None" = None):
    """把与**像素刻度**相关的参数设成当前谱面的口径。

    其余参数（速度阈值、代价权重、空间强度……）已经固化在 ``config.py`` 的默认值里，
    这里只写两个轴的刻度、由刻度推出的起手手位，以及可选的**分段刻度表**
    （六轨模式，见 :class:`ScaleSchedule`）。
    """
    config.physics.pixel_per_unit_x = float(unit_px_x)
    config.physics.pixel_per_unit_y = float(unit_px_y)
    config.physics.home_l = LANE_CENTER - HOME_OFFSET_PX / float(unit_px_x)
    config.physics.home_r = LANE_CENTER + HOME_OFFSET_PX / float(unit_px_x)
    config.physics.scale_schedule = schedule or ScaleSchedule()
    return config



def arc_x_at(arc: AffArc, time_ms: float) -> float:
    """弧在 ``time_ms`` 处的横坐标。

    ``x1/x2`` 是起止横坐标，中间按 X 轴缓动插值；``t2 < t1``（水平弧，仅在黑线上
    出现）按终止值处理。
    """
    span = arc.t2 - arc.t1
    if span == 0:
        # 水平弧：整条弧停在一个点上
        return arc.x1
    t = (time_ms - arc.t1) / span
    return arc.x1 + (arc.x2 - arc.x1) * _axis_ease(arc.easing, 0, t)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

_ARC_RE = re.compile(r"^arc\((?P<args>[^)]*)\)\s*(?P<tail>\[.*\])?\s*;?\s*$")
_ARCTAP_RE = re.compile(r"arctap\(\s*(\d+)\s*(?:,\s*([01])\s*)?\)")
_ARCTAP_ANY_RE = re.compile(r"arctap\(([^)]*)\)")
_TAP_RE = re.compile(
    r"^\(\s*(?P<t>\d+)\s*,\s*(?P<lane>[-\d.]+)\s*(?:,\s*(?P<hand>[-\d.]+)\s*)?\)\s*;?\s*$")
_HOLD_RE = re.compile(
    r"^hold\(\s*(?P<t1>\d+)\s*,\s*(?P<t2>\d+)\s*,\s*(?P<lane>[-\d.]+)"
    r"\s*(?:,\s*(?P<hand>[-\d.]+)\s*)?\)\s*;?\s*$")
_TIMING_RE = re.compile(r"^timing\(\s*(?P<t>\d+)\s*,\s*(?P<bpm>[-\d.]+)\s*,\s*(?P<beats>[-\d.]+)\s*\)\s*;?\s*$")
#: ``scenecontrol(t, 类型, param1, param2);``
#: 类型是裸标识符（``enwidencamera`` / ``enwidenlanes`` / ``trackshow`` ...）。
_SCENECONTROL_RE = re.compile(
    r"^scenecontrol\(\s*(?P<t>\d+)\s*,\s*(?P<kind>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*,\s*(?P<p1>[-\d.]+)\s*,\s*(?P<p2>[-\d.]+)\s*\)\s*;?\s*$")
_HEADER_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>.*)$")


def parse_aff(text: str) -> AffChart:
    """解析 AFF 谱面。

    容错策略：不认识的语句只记一条警告，不影响其余内容，也不改动那一行。
    分隔符 ``-`` 之前的行按头部标识处理。
    """
    source = str(text or "")
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        pass  # 最后一行没有换行符也照常处理

    # 音符宽度按 **AFF 坐标系**给：一个 tap/hold 就是一条轨道宽（0.5 单位 = 268px）。
    # 不能用像素口径的 120 —— 那放进 AFF 坐标会被当成 120 个单位，
    # 于是 jack 的空间衰减项又变回恒为 ~1 的失效状态。
    meta = dict(DEFAULT_META)
    meta["noteWidth"] = LANE_STEP
    chart = Chart(meta=meta, lines=lines, newline=newline)
    arcs: list[AffArc] = []
    arctaps: list[AffArctap] = []
    timings: list[tuple[int, float, float]] = []
    scenecontrols: list[SceneControl] = []
    warnings: list[str] = []
    header: dict[str, str] = {}
    audio_offset = 0
    density = 1.0
    in_body = False

    for index, raw in enumerate(lines):
        line_no = index + 1
        body = raw.rstrip("\r\n").strip()
        if not in_body:
            if body == "-":
                in_body = True
                continue
            if not body:
                continue
            match = _HEADER_RE.match(body)
            if match:
                key, value = match.group("key"), match.group("value").strip()
                header[key] = value
                if key == "AudioOffset":
                    try:
                        audio_offset = int(float(value))
                    except ValueError:
                        warnings.append(f"第 {line_no} 行：AudioOffset 不是数字（已按 0 处理）")
                elif key == "TimingPointDensityFactor":
                    try:
                        density = float(value)
                    except ValueError:
                        warnings.append(f"第 {line_no} 行：TimingPointDensityFactor 不是数字（已按 1 处理）")
            continue

        if not body:
            continue

        # ---- 音弧 ---- #
        arc_match = _ARC_RE.match(body)
        if arc_match:
            args = [part.strip() for part in arc_match.group("args").split(",")]
            if len(args) < 10:
                warnings.append(f"第 {line_no} 行：arc 参数不足（需要 10 个，已忽略）")
                continue
            try:
                arc = AffArc(
                    t1=int(float(args[0])), t2=int(float(args[1])),
                    x1=float(args[2]), x2=float(args[3]), easing=args[4],
                    y1=float(args[5]), y2=float(args[6]), color=int(float(args[7])),
                    hitsound=args[8], arctype=args[9],
                    smoothness=float(args[10]) if len(args) > 10 and args[10] else None,
                    line_no=line_no, raw=raw.rstrip("\r\n"),
                )
            except ValueError:
                warnings.append(f"第 {line_no} 行：arc 参数格式错误（已忽略）")
                continue
            tail = arc_match.group("tail") or ""
            parsed_arctaps: list[AffArctap] = []
            for match in _ARCTAP_RE.finditer(tail):
                parsed_arctaps.append(AffArctap(
                    time=int(match.group(1)),
                    arc=arc,
                    line_no=line_no,
                    hand=int(match.group(2)) if match.group(2) is not None else None,
                    raw_text=match.group(0),
                ))
            arc.arctaps = [arctap.time for arctap in parsed_arctaps]
            arcs.append(arc)
            for arctap in parsed_arctaps:
                tan = arctap.time
                if not (min(arc.t1, arc.t2) <= tan <= max(arc.t1, arc.t2)):
                    warnings.append(
                        f"第 {line_no} 行：arctap 时间 {tan} 不在弧区间 "
                        f"[{arc.t1},{arc.t2}] 内（仍按原样保留）"
                    )
                arctaps.append(arctap)
            continue

        # ---- 长音符 ---- #
        hold_match = _HOLD_RE.match(body)
        if hold_match:
            lane = float(hold_match.group("lane"))
            chart.notes.append(Note(
                time=int(hold_match.group("t1")),
                x=lane_to_x(lane),
                type="hold",
                params={"duration": int(hold_match.group("t2")) - int(hold_match.group("t1")),
                        "lane": lane, "end": int(hold_match.group("t2"))},
                line=line_no,
                raw_line=raw.rstrip("\r\n"),
            ))
            continue

        # ---- 地面音符 ---- #
        tap_match = _TAP_RE.match(body)
        if tap_match:
            lane = float(tap_match.group("lane"))
            chart.notes.append(Note(
                time=int(tap_match.group("t")),
                x=lane_to_x(lane),
                type="tap",
                params={"lane": lane},
                line=line_no,
                raw_line=raw.rstrip("\r\n"),
            ))
            continue

        # ---- 时间轴 ---- #
        timing_match = _TIMING_RE.match(body)
        if timing_match:
            timings.append((int(timing_match.group("t")),
                            float(timing_match.group("bpm")),
                            float(timing_match.group("beats"))))
            continue

        # ---- 场景控制（扩展视野 / 六轨）---- #
        scene_match = _SCENECONTROL_RE.match(body)
        if scene_match:
            kind = scene_match.group("kind")
            if kind in ("enwidencamera", "enwidenlanes"):
                scenecontrols.append(SceneControl(
                    time=int(scene_match.group("t")),
                    kind="camera" if kind == "enwidencamera" else "lanes",
                    enable=float(scene_match.group("p2")) != 0.0,
                    duration=float(scene_match.group("p1")),
                    line=line_no,
                    raw=raw.rstrip("\r\n"),
                ))
            continue

        # ---- 其它（camera / timinggroup / flick 等）---- #
        # timinggroup 的结构行（`timinggroup(noinput){` 与它的右括弧 `};` / `}`）
        # 是谱面的正常写法（Lamia 里就有很多），当前不实现分组语义、静默跳过即可，
        # **不该报警告** —— 那只是噪声。
        if body.startswith("timinggroup") or body in ("{", "}", "};") \
                or body.startswith("camera") or body.startswith("scenecontrol"):
            continue
        if body.startswith("flick"):
            warnings.append(f"第 {line_no} 行：flick 是废弃格式（已忽略，不做预留）")
            continue
        warnings.append(f"第 {line_no} 行：不认识的语句（已忽略）")

    # ---- arctap 变成统一物件 ---- #
    for arctap in arctaps:
        chart.notes.append(Note(
            time=arctap.time,
            x=arc_x_at(arctap.arc, arctap.time),
            y=arctap.arc.y_at(arctap.time),
            type="arctap",
            params={"y": arctap.arc.y_at(arctap.time),
                    "arc_color": arctap.arc.color,
                    "arc_line": arctap.arc.line_no},
            line=arctap.line_no,
            raw_line=arctap.arc.raw,
        ))

    # 地面音符行里已有的手标记（例如上次拆谱的结果）只当作"偏好"，不是硬锁定 ——
    # 只有 arc 的 color 是不可动的。这里把它记进 params 供回写与原位修改。
    for note in chart.notes:
        if note.type in ("tap", "hold"):
            existing = _existing_hand_of(note)
            if existing is not None:
                note.params["_hand"] = existing

    sort_notes(chart.notes)
    chart.warnings.extend(warnings)
    return AffChart(
        chart=chart, arcs=arcs, arctaps=arctaps, timings=timings,
        audio_offset=audio_offset, density_factor=density, header=header,
        newline=newline, lines=lines, warnings=warnings,
        scenecontrols=scenecontrols,
    )


# --------------------------------------------------------------------------- #
# 回写
# --------------------------------------------------------------------------- #

def _existing_hand_of(note: Note) -> int | None:
    """地面音符行里是否已经写了手标记（0/1）。有就返回它。"""
    body = note.raw_line.strip()
    open_paren = body.find("(")
    close_paren = body.rfind(")")
    if open_paren < 0 or close_paren <= open_paren:
        return None
    parts = [p.strip() for p in body[open_paren + 1: close_paren].split(",") if p.strip()]
    expected = _EXPECTED_ARGS.get(note.type, 0)
    if len(parts) <= expected:
        return None
    try:
        value = int(float(parts[expected]))
    except (ValueError, IndexError):
        return None
    return value if value in (HAND_LEFT, HAND_RIGHT) else None


def _insert_hand(args_text: str, hand: int) -> str:
    """把 ``0`` / ``1`` 作为最后一个参数插进括号内（保留原有空白风格）。"""
    stripped = args_text.rstrip()
    trailing = args_text[len(stripped):]
    return f"{stripped},{hand}{trailing}"


def serialize_aff(parsed: AffChart, hands: dict[Note, int] | None = None) -> str:
    """回写 AFF 谱面：给三类判定物件都补/换手标记，其它内容原样保留。

    - ``tap(t,lane)`` / ``hold(t1,t2,lane)``：在括号内末尾补一个 ``0`` / ``1``；
    - ``arctap(t)``：在括号内补 ``arctap(t,0)`` / ``arctap(t,1)``；
    - ``arc(...)``：**参数一律不动**（它的第 8 个参数 color 就是手标记）。

    :param hands: ``{物件: 0/1}``。已经写过标记的会**原位替换**，所以是幂等的。
    """
    lines = list(parsed.lines)

    # ---- arctap：按"行"聚合后整行重写（一条 arc 行上可能挂多个 arctap） ---- #
    if hands:
        arctap_hand: dict[int, int] = {}
        for note in parsed.chart.notes:
            if note.type != "arctap":
                continue
            value = hands.get(note)
            if value is not None:
                arctap_hand[note.line] = int(value)
        # 一行上有多个 arctap 时，用 (时间, 手标记) 逐个替换
        wanted: dict[int, list[tuple[int, int]]] = {}
        for note in parsed.chart.notes:
            if note.type != "arctap":
                continue
            value = hands.get(note)
            if value is None:
                continue
            wanted.setdefault(note.line, []).append((note.time, int(value)))
        for line_no, items in wanted.items():
            position = line_no - 1
            if position < 0 or position >= len(lines):
                continue
            raw = lines[position]
            eol = ""
            if raw.endswith("\r\n"):
                eol = "\r\n"
            elif raw.endswith("\n"):
                eol = "\n"
            body = raw[: len(raw) - len(eol)] if eol else raw
            by_time = {time: hand for time, hand in items}
            used: set[int] = set()

            def _replace(match: "re.Match[str]") -> str:
                inner = match.group(1)
                parsed_tap = _ARCTAP_RE.match(match.group(0))
                if parsed_tap is None:
                    return match.group(0)
                time = int(parsed_tap.group(1))
                if time in by_time and time not in used:
                    used.add(time)
                    return f"arctap({time},{by_time[time]})"
                return match.group(0)

            body = _ARCTAP_ANY_RE.sub(_replace, body)
            lines[position] = body + eol

    # ---- tap / hold ---- #
    for note in parsed.chart.notes:
        if note.type not in ("tap", "hold"):
            continue                      # arctap 在上面按行处理过了
        hand = (hands or {}).get(note) if hands else None
        if hand is None:
            continue
        position = note.line - 1
        if position < 0 or position >= len(lines):
            continue
        raw = lines[position]
        eol = ""
        if raw.endswith("\r\n"):
            eol = "\r\n"
        elif raw.endswith("\n"):
            eol = "\n"
        body = raw[: len(raw) - len(eol)] if eol else raw

        open_paren = body.find("(")
        close_paren = body.rfind(")")
        if open_paren < 0 or close_paren < open_paren:
            continue
        # 手标记的位次由格式决定：tap = 第 3 个参数（下标 2），hold = 第 4 个（下标 3）。
        # 不要用"当前行有几个参数"去推断 —— 对已经标过的行会算错位次、把整行写乱。
        hand_index = _HAND_INDEX[note.type]
        parts = body[open_paren + 1: close_paren].split(",")
        # 去掉纯空白造成的空片段影响：只按"非空参数"计数
        seen = 0
        replaced = False
        for part_index, part in enumerate(parts):
            if not part.strip():
                continue
            if seen == hand_index:
                parts[part_index] = str(int(hand))
                replaced = True
                break
            seen += 1
        if replaced:
            new_body = body[: open_paren + 1] + ",".join(parts) + body[close_paren:]
        else:
            new_body = (body[: open_paren + 1]
                        + _insert_hand(body[open_paren + 1: close_paren], int(hand))
                        + body[close_paren:])
        lines[position] = new_body + eol
    return "".join(lines)
