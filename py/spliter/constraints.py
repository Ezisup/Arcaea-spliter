"""Arcaea 谱面的硬约束：音弧颜色（= 手标记）与地面物件的互斥关系。

规则（来自谱面语义，不是偏好，所以做成**硬约束**而不是扣分）
----------------------------------------------------------
视频/游戏的记号约定：arc 的第 8 个参数 ``color`` 就是它的手标记
（``0`` = 左手 / ``1`` = 右手），而我们回写时给地面音符补的也是同一套 0/1。
于是：

1. **arc 自身的手标记不可修改** —— arc 的 color 是谱面既有的，程序只读；
2. **同刻同手**：若地面物件与某条实体音弧的**起始时间完全相同、而且落点也相同**，
   它应当与该弧同手（手指点下去正好接着按住，物理上说得通）。
   **位置必须对得上**：同刻但落在别的点上的键不是"同一次落指"，那根手指正在起弧，
   于是它落到规则 3 手里判成"异手"（谱面里 33 处同刻起弧的点全部属于这种，
   距离最近的一处也有 536px —— 只看时间不看位置会把左右手同时钉在两个点上）；
3. **区间异手**：实体音弧存在期间（``[t1, t2]``，含两端），其它物件原则上
   不得使用该弧的手 —— 否则同一根手指要在按住音弧的同时去点别的东西；
4. **多押例外**：同一时刻的多个物件若落在不同颜色的弧区间内，规则 3 不可能
   同时满足，此时允许特殊处理（这正是需求里说的"两种例外"之一）。
   另一个例外就是规则 2：同刻给同手，而不是异手。

黑线（``arctype != false``）没有颜色语义（color 对它无视觉效果），所以规则 2/3
只对实体音弧成立。挂在黑线上的 Arctap 是独立的判定物件，可以自由分手，
不对它施加 arc 约束。

实现方式
--------
:class:`ArcConstraints` 只回答两个问题：``force_hand`` / ``forbidden_hands``。
拆谱器（:class:`spliter.split.Scoring`）在生成候选和做可行性检查时会问它，
因此约束是**真正不可违反**的，而不是"罚得比较重"。
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field

from .aff import (HAND_LEFT, HAND_RIGHT, PIXELS_PER_UNIT_X, PIXELS_PER_UNIT_Y,
                  AffChart, AffArc)
from .chart import Note, isolated
from .split import HANDS, HOLD_ARC_OVERLAP, SAME_HAND_MARGIN_MS

__all__ = ["ArcConstraints", "build_arc_constraints"]

#: 手标记（整数） -> 拆谱器内部用的字母。
#: 官方 color 取值 0=蓝 / 1=红 / 2=绿（wiki:谱面格式）。目前只把 0/1 当手标记，
#: 其余颜色（绿=任一手可打）不产生约束，也绝不让程序因此崩掉。
INT_TO_HAND = {HAND_LEFT: "L", HAND_RIGHT: "R"}
HAND_TO_INT = {"L": HAND_LEFT, "R": HAND_RIGHT}


@dataclass
class ArcConstraints:
    """地面物件的可选手集合。

    - ``forced[index]``：被"同刻同手"钉死的手（字母 ``'L'``/``'R'``）；
    - ``forbidden[index]``：被"区间异手"排除的手；
    - ``notes``：不可行时用来报告行号。
    """

    forced: dict[int, str] = field(default_factory=dict)
    forbidden: dict[int, set[str]] = field(default_factory=dict)
    reasons: dict[int, list[str]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    arcs_by_hand: dict[str, list[AffArc]] = field(default_factory=dict, repr=False)

    def allowed_hands(self, index: int) -> tuple[str, ...]:
        """这个物件还能用哪些手。"""
        forced = self.forced.get(index)
        if forced:
            return (forced,)
        blocked = self.forbidden.get(index)
        if not blocked:
            return HANDS
        return tuple(hand for hand in HANDS if hand not in blocked)

    def force_hand(self, index: int) -> str | None:
        return self.forced.get(index)

    def forbidden_hands(self, index: int) -> set[str]:
        return self.forbidden.get(index, set())

    def explain(self, index: int) -> str:
        return "；".join(self.reasons.get(index, []))

    def hand_catch(self, hand: str, t_from: int, t_to: int):
        """这只手在 ``(t_from, t_to]`` 里要接住的**实体弧弧头**（没有返回 None）。

        用途：终点是音弧的位移，其到位难度明显轻于"点到点"（音弧只有最高一级判定）。
        """
        for arc in self.arcs_by_hand.get(hand, ()):   # 按 t2 有序，这里要按 t1 找
            if t_from < arc.t1 <= t_to:
                return (arc.t1, arc.x1, arc.y1)
        return None

    def hand_release(self, hand: str, t_from: int, t_to: int) -> tuple[int, float, float] | None:
        """这只手在 ``(t_from, t_to)`` 之间**被音弧尾巴钉住的最后一次落点**。

        音弧也是"按住不放"的：手从 ``t1`` 一路跟到 ``t2``，**结束那一刻停在
        ``(x2, y2)``**。所以上一个判定物件在弧上、下一个判定物件在弧结束之后时，
        位移要从尾巴算起、可用时间也只能从 ``t2`` 算起
        （否则会出现"12 秒扛着弧、跨 422px 却只给 175ms"那种错账）。

        返回 ``(时刻, x, y)``；这只手没有在途中收尾的音弧时返回 ``None``。
        要求 ``t_from < t2 < t_to`` —— 弧尾正好压在下一个物件上时那是"物件落在弧尾"，
        走原本的算法（也避免"同一时刻两个不同落点"这种模型矛盾被算成无穷大代价）。
        """
        best: tuple[int, float, float] | None = None
        for arc in self.arcs_by_hand.get(hand, ()):   # 按 t2 有序
            end = max(arc.t1, arc.t2)
            if end >= t_to:
                break
            # 弧必须在 (t_from, t_to) 之间收尾，而且不能在下一个键之后才开始。
            # 注意**不能**要求"弧在手上一键之前就已开始"：谱面里常见一整条
            # 首尾相接的弧链（例如每 100ms 一条），而那只手的上一个判定物件
            # 可能在很久以前（本谱 106300 处两者相差 1600ms），那样会把整条链
            # 全跳过 —— 手明明一路跟着走到弧尾，模型却当作"没有释放点"。
            if end <= t_from or arc.t1 >= t_to:
                continue                              # 没覆盖到这段
            if arc.t1 > t_from:
                continue                  # 弧要在上一键之前就已经在跟（严格归属）
            # 这条弧在 t_from 之后还有判定物件 -> 这只手当时跟的是别的东西
            # （同色弧区间重叠时按颜色归因会张冠李戴，曾把一处算成 1.8e6 的假代价），
            # 不能把它的尾巴当成这一手的释放点。
            if arc.arctaps and max(arc.arctaps) > t_from:
                continue
            if best is None or end > best[0]:
                best = (end, arc.x2, arc.y2)
        return best


#: 需要参与"同一只手不能兼顾"判断的判定物件类型。
#: arctap 与 tap 在游玩上完全等价（都是一次点击），**必须**一起参与 ——
#: 早期版本把它漏掉了，导致音弧区间内出现同一只手的 arctap。
JUDGEABLE_TYPES = ("tap", "hold", "arctap")


def _is_judgeable(note: Note) -> bool:
    return note.type in JUDGEABLE_TYPES


def _hold_windows(notes: list[Note]) -> list[tuple[int, int, str, int, int]]:
    """所有 hold 的占手区间：``(起, 止, 手, 物件下标, 隔离组号)``。

    hold 与音弧是同一条原则：手指按住期间不能去点别的键，
    所以区间内的其它判定物件应当换手。
    """
    windows = []
    for index, note in enumerate(notes):
        if note.type != "hold":
            continue
        if not note.side:
            continue                      # 没有手标记就不成约束（理论上不会发生）
        end = int(note.params.get("end", note.time + (note.params.get("duration") or 0)))
        windows.append((note.time, end, note.side, index, note.noinput))
    return windows


#: 判定"同刻起弧"时两点算同一个位置的屏幕像素容差（半条轨道 268/2 ≈ 134px）。
ARC_START_SAME_POS_PX = 134.0


def _same_point(note: Note, arc: AffArc, unit_px_x: float, unit_px_y: float) -> bool:
    """这个物件是不是落在音弧起点的"同一个点"上（分轴换算后勾股）。

    只看时间不看位置是不够的：手指点下去要能"正好接住"这条弧，两者必须在同一处。
    """
    dx = (note.x - arc.x1) * unit_px_x
    dy = (note.y - arc.y1) * unit_px_y
    return math.hypot(dx, dy) <= ARC_START_SAME_POS_PX


def _scale_at(schedule, time_ms: float,
              unit_px_x: float, unit_px_y: float) -> tuple[float, float]:
    """某一时刻的实际刻度（含六轨倍率）。``schedule`` 为空时退回标准四轨刻度。"""
    if schedule:
        scale_x, scale_y = schedule.at(time_ms)
        return unit_px_x * scale_x, unit_px_y * scale_y
    return unit_px_x, unit_px_y


def build_arc_constraints(parsed: AffChart, notes: list[Note],
                          unit_px_x: float = PIXELS_PER_UNIT_X,
                          unit_px_y: float = PIXELS_PER_UNIT_Y,
                          schedule=None) -> ArcConstraints:
    """从 AFF 谱面 + 统一物件列表生成硬约束。

    :param notes: 与 ``parsed.chart.notes`` 同一个列表（用 ``index`` 对齐）。
    :param schedule: 分段刻度表（六轨模式）；``None`` = 全程标准四轨。
    """
    constraints = ArcConstraints()
    # **双指**情况下所有实体弧（不分长短、动不动）都占手：color 就是手，
    # 区间内其它物件不得用那只手 —— 这条规则之前执行得很好，别动。
    # 多指段由拆谱核心另按"弧的采样点参与同 y 左右排序"处理
    # （见 split.Scoring._scan_multi_finger），所以这里不做任何长短区分。
    solid = [arc for arc in parsed.arcs
             if arc.is_solid and arc.color in INT_TO_HAND]
    holds = _hold_windows(notes)
    if not solid and not holds:
        constraints.stats = {"solid_arcs": 0, "holds": 0,
                             "forced": 0, "forbidden": 0, "chord_exceptions": 0}
        return constraints

    # 按起点建索引，便于第 2 条规则
    by_start: dict[int, list[AffArc]] = {}
    for arc in solid:
        by_start.setdefault(arc.t1, []).append(arc)
    # 按手建索引（按"结束时刻"有序），供 hand_release 查"手指停在弧尾的位置"
    for arc in solid:
        constraints.arcs_by_hand.setdefault(INT_TO_HAND[arc.color], []).append(arc)
    unknown_colors = sorted({arc.color for arc in parsed.arcs
                             if arc.is_solid and arc.color not in INT_TO_HAND})
    for arcs in constraints.arcs_by_hand.values():
        arcs.sort(key=lambda a: max(a.t1, a.t2))

    # 同一时刻的物件集合：多押要作为例外整体放行
    by_time: dict[int, list[int]] = {}
    for index, note in enumerate(notes):
        by_time.setdefault(note.time, []).append(index)

    # 每条弧**自己的** arctap 落在哪些点上：它们就在弧上、由同一根手指一并处理，
    # 所以不算"其他同高度的物件"，不受这条弧的颜色约束（用户口径：
    # "在同高度上，**其他** arctap 就不该有相同的颜色"）。
    own_spots: dict[int, set[tuple[int, float, float]]] = {}
    for arctap in parsed.arctaps:
        arc = arctap.arc
        x = arc.x_at(arctap.time)
        y = arc.y_at(arctap.time)
        own_spots.setdefault(id(arc), set()).add(
            (int(arctap.time), round(x, 3), round(y, 3)))

    forced = 0
    forbidden = 0
    chord_exceptions = 0

    for time, indexes in by_time.items():
        # ---- 规则 2：与弧起点同刻 -> 必须同手 ---- #
        # 起手时刻允许 SAME_HAND_MARGIN_MS 的对齐误差（谱面可能有浮点误差/失误）
        starts_here: list[AffArc] = []
        for offset in range(-SAME_HAND_MARGIN_MS, SAME_HAND_MARGIN_MS + 1):
            starts_here.extend(by_start.get(time + offset, ()))
        if starts_here:
            # **按点判**，不能按"这一时刻所有起弧的颜色"一刀切：
            # 谱面里常见"同一刻两条不同颜色的弧在不同位置起步 + 一个键落在其中
            # 一条的弧头上"，那种键必须与**它所在的那条弧**同手；只有同一个点上
            # 真的挤了两条不同颜色的弧头时才叫"多押例外"。
            for index in indexes:
                if not _is_judgeable(notes[index]):
                    continue
                note = notes[index]
                px_x, px_y = _scale_at(schedule, note.time, unit_px_x, unit_px_y)
                on_head = [arc for arc in starts_here
                           if not isolated(arc.noinput, note.noinput)
                           and _same_point(note, arc, px_x, px_y)]
                if not on_head:
                    continue              # 同刻但不同点：交给规则 3 判"异手"
                hands_here = {INT_TO_HAND[arc.color] for arc in on_head}
                if len(hands_here) > 1:
                    chord_exceptions += 1       # 同一点上两条异色弧头：真例外
                    continue
                hand = hands_here.pop()
                constraints.forced[index] = hand
                constraints.reasons.setdefault(index, []).append(
                    f"与音弧同刻起手、且落在弧头上（弧色 {on_head[0].color} -> {hand} 手）"
                )
                forced += 1

        # ---- 规则 3：落在实体弧区间内 -> 不得同手 ---- #
        for index in indexes:
            note = notes[index]
            if not _is_judgeable(note):
                continue
            if index in constraints.forced:
                continue                      # 规则 2 已经钉死了
            blocking: set[str] = set()
            reasons: list[str] = []
            # 长条要**整段**占着手：只看起点会漏掉"长条按到一半时另一条弧才起"
            # 这种组合（用户实测 halcyon 127535：左手按着 0.25 的长条、又去跟一条
            # 0 色弧 —— 那需要 3 指，必须筛掉）。所以按**区间重叠**判。
            note_end = note.time
            if HOLD_ARC_OVERLAP and note.type == "hold":
                try:
                    note_end = int(note.params.get("end")
                                   or note.time + (note.params.get("duration") or 0))
                except (TypeError, ValueError):
                    note_end = note.time
            for arc in solid:
                lo, hi = min(arc.t1, arc.t2), max(arc.t1, arc.t2)
                if note_end < lo - SAME_HAND_MARGIN_MS or note.time > hi + SAME_HAND_MARGIN_MS:
                    continue
                if isolated(arc.noinput, note.noinput):
                    continue              # 隔离组的弧不限制组外的键
                if _is_own_arctap(own_spots, arc, note):
                    continue                  # 挂在它自己身上的 arctap：同一根手指
                blocking.add(INT_TO_HAND[arc.color])
                reasons.append(f"音弧 color={arc.color}")
            # hold 的占手区间：区间内（不含尾指那一刻）不能同手
            for start, end, hand, hold_index, hold_group in holds:
                if hold_index == index:
                    continue
                if isolated(hold_group, note.noinput):
                    continue              # 隔离组的长条不挡组外的键
                if start < note.time < end:
                    blocking.add(hand)
                    reasons.append(f"长条 t=[{start},{end}] 占着 {hand} 手")
            # 规则 4：多押例外 —— 若这个时刻的其它物件是"被迫同手"的另一只手，
            # 那么两条规则会直接打架，此时不施加区间约束（并记账）。
            if len(indexes) > 1 and blocking:
                forced_here = {constraints.forced[i] for i in indexes
                               if i in constraints.forced}
                if forced_here and not forced_here <= blocking:
                    chord_exceptions += 1
                    continue
            if not blocking:
                continue
            if len(blocking) >= len(HANDS):
                # 两只手都被不同颜色的弧占用 -> 无解，只能放行
                chord_exceptions += 1
                continue
            constraints.forbidden[index] = blocking
            blocked_names = "/".join(sorted(blocking))
            constraints.reasons.setdefault(index, []).append(
                f"不得用 {blocked_names} 手（{'；'.join(reasons)}）"
            )
            forbidden += 1

    constraints.stats = {
        "solid_arcs": len(solid),
        "colors_without_hand": unknown_colors,
        "holds": len(holds),
        "forced": forced,
        "forbidden": forbidden,
        "chord_exceptions": chord_exceptions,
    }
    return constraints


def _is_own_arctap(own_spots: dict[int, set[tuple[int, float, float]]],
                   arc: AffArc, note: Note) -> bool:
    """这个物件是不是**就挂在这条弧上**的那个 arctap。

    是的话它与这条弧由同一根手指处理（"同刻同手"的推广），不构成"异手"。
    """
    spots = own_spots.get(id(arc))
    if not spots:
        return False
    return (int(note.time), round(note.x, 3), round(note.y, 3)) in spots


def describe_constraints(constraints: ArcConstraints, notes: list[Note],
                         limit: int = 10) -> list[str]:
    """把约束渲染成人能读的几行（``--show-constraints`` 用）。"""
    lines: list[str] = []
    stats = constraints.stats
    lines.append(
        f"实体音弧 {stats.get('solid_arcs', 0)} 条；长条占手 {stats.get('holds', 0)} 条；"
        f"同刻钉死 {stats.get('forced', 0)} 个地面物件；"
        f"区间限手 {stats.get('forbidden', 0)} 个；"
        f"多押例外 {stats.get('chord_exceptions', 0)} 处"
    )
    shown = 0
    for index in sorted(set(constraints.forced) | set(constraints.forbidden)):
        if shown >= limit:
            lines.append(f"  ...（还有更多，用 --explain 看逐个物件）")
            break
        note = notes[index]
        lines.append(f"  第 {note.line:>5} 行 t={note.time:<7} x={note.x:+.3f}  "
                     f"可用手={constraints.allowed_hands(index)}  ← {constraints.explain(index)}")
        shown += 1
    return lines
