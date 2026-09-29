"""拆谱质量评估：把一份"已经带标记"的谱面换算成可比数字。

用途有两个：

1. 反复调参时看指标变化（不用每次肉眼扫谱面）；
2. 多份拆谱结果对比（``--compare``）。

指标口径尽量按"人能看懂"来：

- **位移速度**：同一只手相邻两次按键的 ``位移 / 间隔``，取最大几个看瓶颈。
  这是"手要跑多远"，**不是**玩家口语里的手速（那个见下一条）。
- **手速榜**：同一只手相邻两次按键的**最小时间间隔**，与位移无关。
  位移**以"轨道"为单位**报告（1 轨道 = 0.5 个谱面坐标 = 相邻轨道的间距），
  因为像素对玩家毫无体感、且随设备分辨率变化，而轨道数是任何人在谱面上
  一眼就能数的。像素只在代码与标定处使用。
- **抗**：同手相邻且间隔 <= 抗窗口 的连续按键；同键位的单独计数。
- **交互率**：高密度相邻键里"左右手交替"的占比。
- **占手冲突**：按着长条还要去按别的键。
- **左右均衡**：L/R 数量差。

**统计口径：地面（tap/hold）与天空（arctap）一律合并统计。**
曾经只统计地面，那是错的 —— arctap 的负担同样是真实的，而且往往才是最
吃紧的部分（实测同一份谱面里天空的最大位移速度比地面高出 50%+）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from .chart import Note, isolated
from .config import Config
from .split import (HANDS, SAME_HAND_MARGIN_MS, SplitResult, build_groups,
                    scale_at, screen_distance, sequence_note)

#: 一个轨道的宽度（谱面坐标单位）。1 轨道 = 0.5。
#:
#: 这里**故意重复定义**而不从 ``aff`` 导入：``aff`` 会 import ``chart``，
#: 而 ``report`` 被 ``aff_cli`` 在很靠后的位置导入，跨模块取常量容易绕成环。
#: 数值本身是谱面格式的固定事实（见 ``aff.LANE_STEP``），自测里有断言对齐。
LANE_UNIT = 0.5

#: 满高（``y = 0`` 到 ``y = 1``）在"轨道"口径下的长度。
#:
#: 横向 1 轨道 = 268px、纵向满高 = 326px，所以整张判定面在斜视投影下
#: **横 4 轨 ≈ 纵 1.22 轨**。这个常数只用于把"人话"讲准，不参与任何判定。
FULL_HEIGHT_IN_LANES = 326.0 / (LANE_UNIT * 536.0)


def lanes(distance_px: float, physics, unit_px: float | None = None) -> float:
    """把屏幕像素距离换算成"横跨了几个轨道"。

    像素随设备分辨率变化、玩家毫无体感；轨道数是谱面上能直接数的东西，
    所以**给玩家看的数字一律用这个口径**。像素只在代码与标定处保留。

    :param unit_px: 该段落的横向像素刻度（``px / 坐标单位``）。
        **六轨段落的刻度更小，必须传当段刻度**，否则同样的几何位移会被报成
        不同的轨道数。省略时退回 ``physics.pixel_per_unit_x``（标准四轨）。
        传值请用 :func:`spliter.split.scale_at`，它已把六轨倍率算进去。
    """
    if unit_px is None:
        unit_px = getattr(physics, "pixel_per_unit_x", 536.0) or 536.0
    unit = LANE_UNIT * unit_px
    return distance_px / unit if unit else 0.0


@dataclass
class MoveRecord:
    """一次位移最快的移动（**位移速度**榜的条目）。"""

    from_line: int
    to_line: int
    hand: str
    delta_t: int
    delta_x: float
    speed: float
    from_x: float
    to_x: float
    from_y: float = 0.0
    to_y: float = 0.0
    #: 这次移动发生时的横向像素刻度（``px / 坐标单位``）。
    #: 六轨段落的刻度更小，所以**必须逐次记录**，不能全局取一个常量 ——
    #: 否则 6k 段落的位移会被报成不同的轨道数。
    unit_px_x: float | None = None
    #: 起点/终点物件的判定时刻（ms）。报告里用"时刻"而不是行号 ——
    #: 行号只在编辑器里才有意义，玩家是按时间找位置的（与手速榜同口径）。
    from_time: int = 0
    to_time: int = 0

    def describe(self, physics=None) -> str:
        """人类可读的一行。

        位移**分横向与纵向两个分量**报，不报勾股值：
        横向用"轨道"（谱面上能直接数），纵向用坐标高度
        （``0.00`` = 判定线、``1.00`` = 满高）。
        勾股值会把两个分量混在一起，"跨 1.57 轨道"到底是横向还是纵向完全看不出来。
        """
        horizontal = abs(self.to_x - self.from_x)
        vertical = abs(self.to_y - self.from_y)
        unit_px = self.unit_px_x
        if unit_px is None and physics is not None:
            unit_px = getattr(physics, "pixel_per_unit_x", None)
        if unit_px is None:
            # 没有刻度信息时退回像素口径（仅旧调用会走到）
            motion = f"位移 {self.delta_x:.0f}px"
            speed_text = f"{round(self.speed)} px/s"
        else:
            parts: list[str] = []
            if horizontal >= 1e-9:
                parts.append(f"横向 {lanes(horizontal * unit_px, physics, unit_px):.2f} 轨道")
            if vertical >= 1e-9:
                parts.append(f"纵向 {vertical:.2f}")
            motion = " + ".join(parts) if parts else "原地"
            # 速度仍用勾股（那才是手指真实的移动长度），单位是轨道/秒
            span = lanes(self.delta_x, physics, unit_px)
            speed_text = (f"{span / (self.delta_t / 1000):.2f} 轨道/秒"
                          if self.delta_t else "-")
        # 用"时刻"表示位置（与手速榜一致）：行号只在编辑器里有意义。
        where = (f"{self.from_time}-{self.to_time}" if self.to_time
                 else f"第 {self.from_line}->{self.to_line} 行")
        return (f"{self.hand}手  {self.delta_t}ms 内 {motion}  ({speed_text})  "
                f"{where}  "
                f"({self.from_x:+.2f},{self.from_y:.2f}) -> ({self.to_x:+.2f},{self.to_y:.2f})")


@dataclass
class GapRecord:
    """同手相邻两键的时间间隔（**手速榜**的条目）。

    与 :class:`MoveRecord` 的区别：那个衡量"手要跑多远"，这个只看
    "同一个手指两次落下之间剩多少时间"，**与位移无关**。
    玩家口语里的"手速"通常是后者。
    """

    gap_ms: int
    hand: str
    from_line: int
    to_line: int
    from_x: float
    to_x: float
    from_y: float
    to_y: float
    from_time: int
    to_time: int

    def describe(self, rank: int, tied: int = 1) -> str:
        """``第一名, 间隔xxms, 相当于bpm多少的8分/16分音符,  xx-yy等n组``。"""
        bpm_text = rhythm_label(self.gap_ms)
        # 并列时按用户口径："xx-yy等n组"，只拿一对举例
        pair = f"{self.from_time}-{self.to_time}"
        if tied > 1:
            pair += f"等 {tied} 组"
        return (f"第{_ordinals(rank)}名, 间隔 {self.gap_ms}ms, "
                f"{bpm_text}, {pair}")


#: 名词化序数（名次用）。
_ORDINAL_WORDS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")


def _ordinals(rank: int) -> str:
    if 1 <= rank <= len(_ORDINAL_WORDS):
        return _ORDINAL_WORDS[rank - 1]
    return str(rank)


def rhythm_label(gap_ms: int) -> str:
    """把毫秒间隔说成"相当于 BPM xx 的 8分/16分音符"。

    换算依据：``间隔 = 60000 / BPM × (4 / 音符数)``，于是

    - 8 分（每拍 2 个）：``间隔 = 30000 / BPM`` → ``BPM = 30000 / 间隔``
    - 16 分（每拍 4 个）：``间隔 = 15000 / BPM`` → ``BPM = 15000 / 间隔``

    口径（用户指定）：

    - 先用 **8 分**表示；
    - 若等效 BPM **超过 300**，换成 **16 分**；
    - **只有这两种选项**，16 分再超也不换 32 分。
    """
    if gap_ms <= 0:
        return "间隔为 0"
    bpm_eighth = 30000.0 / gap_ms
    if bpm_eighth <= 300.0:
        return f"相当于 BPM {round(bpm_eighth)} 的 8 分音符"
    bpm_sixteenth = 15000.0 / gap_ms
    return f"相当于 BPM {round(bpm_sixteenth)} 的 16 分音符"


@dataclass
class Spot:
    """报告里"可定位的一处"。

    `--stats` 的注意点凡是能明确列出位置的，都把位置一起给出来 ——
    只报一个计数（"有 2 处"）读者无法回到谱面里找。
    时刻用 ms（与位移榜/手速榜同口径），行号只在编辑器里有意义。
    """

    time: int
    #: 结束时刻（区间类问题用，例如"长条占手"的"下一个键在何时"）。
    end_time: int | None = None
    hand: str = ""
    text: str = ""

    def describe(self) -> str:
        return self.text


@dataclass
class Warning:
    """一条注意点：一句话 + 可定位的具体位置（可能为空）。"""

    text: str
    spots: list[Spot] = field(default_factory=list)
    #: 最多列几处（``None`` = 全部）。超出部分在渲染时折成"等 N 处"。
    limit: int | None = None

    def lines(self, indent: str = "  ") -> list[str]:
        """渲染成若干行：第一行是结论，后面缩进列出位置。"""
        rows = [self.text]
        spots = self.spots if self.limit is None else self.spots[:self.limit]
        rows.extend(f"{indent}- {spot.describe()}" for spot in spots)
        hidden = len(self.spots) - len(spots)
        if hidden > 0:
            rows.append(f"{indent}- （还有 {hidden} 处未列出）")
        return rows


@dataclass
class Stats:
    """一份谱面的拆谱指标。"""

    total: int = 0
    L: int = 0
    R: int = 0
    M: int = 0
    none: int = 0
    locked: int = 0
    free: int = 0
    holds: int = 0
    hold_blocked: int = 0
    chords: int = 0
    max_chord: int = 0
    same_hand_chord: int = 0
    min_gap: int | None = None
    median_gap: float | None = None
    same_hand_close: int = 0
    same_hand_same_x: int = 0
    over_comfy_speed: int = 0
    over_max_speed: int = 0
    max_speed_seen: float = 0
    #: 最大单手**位移速度**，以"轨道/秒"计（见 :func:`lanes`）。
    #: 注意这是**位移**速度，不是玩家口语里的"手速"（那个见 ``fastest_gaps``）。
    max_lane_speed: float = 0.0
    #: 舒适/极限**位移速度**阈值，同样换算到"轨道/秒"。
    comfy_lane_speed: float = 0.0
    max_lane_speed_limit: float = 0.0
    #: 本次统计里最大的**横向**单次位移（轨道数）。
    #: 只算横向：纵向不属于"轨道"这个概念，混进来会让数字无法解读。
    max_move_lanes: float = 0.0
    #: **手速榜**：同手相邻两键的时间间隔，从快到慢。
    #: 只看非 arc 物件（tap / arctap / hold 头），与位移无关。
    fastest_gaps: list[GapRecord] = field(default_factory=list)
    #: 手速榜里最快那一档的间隔（ms），和它的并列组数。
    fastest_gap_ms: int | None = None
    fastest_gap_tied: int = 0
    alternating: int = 0
    repeated: int = 0
    notes_per_second: float = 0
    balance: float = 0
    hard_moves: list[MoveRecord] = field(default_factory=list)
    per_second: list[tuple[int, int]] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)

    @property
    def alternation_ratio(self) -> float | None:
        """高密度相邻键里左右交替的占比（None = 没有高密度段）。"""
        total = self.alternating + self.repeated
        return round(self.alternating / total, 3) if total else None

    def summary_line(self) -> str:
        """一行紧凑摘要。"""
        parts = [
            f"物件 {self.total}",
            f"L/R/M/无 = {self.L}/{self.R}/{self.M}/{self.none}",
            f"和弦 {self.chords}",
            f"抗 {self.same_hand_close}",
        ]
        ratio = self.alternation_ratio
        if ratio is not None:
            parts.append(f"交互率 {ratio * 100:.0f}%")
        if self.min_gap is not None:
            parts.append(f"最小间隔 {self.min_gap}ms")
        return " | ".join(parts)


def analyze(notes: list[Note], config: Config) -> Stats:
    """把一份带标记的谱面换算成 :class:`Stats`。

    :param notes: ``parse_chart`` 的物件（``side`` 必须是最终标记）
    """
    jack_window = config.cost.jack_window
    dense_gap = config.output.report_dense_gap
    comfy_speed = config.physics.comfy_speed
    max_speed = config.physics.max_speed

    stats = Stats(total=len(notes))
    if not notes:
        return stats

    hands: list[str | None] = [note.side for note in notes]
    groups = build_groups(notes)

    # ---- 基础计数 ---- #
    for note in notes:
        if note.side == "L":
            stats.L += 1
        elif note.side == "R":
            stats.R += 1
        elif note.side == "M":
            stats.M += 1
        else:
            stats.none += 1
        if note.side:
            stats.locked += 1
        else:
            stats.free += 1
        if str(note.type).lower() == "hold":
            stats.holds += 1

    marked = stats.L + stats.R
    stats.balance = round((stats.L - stats.R) / marked, 3) if marked else 0.0

    # ---- 间隔 ---- #
    gaps = [notes[i].time - notes[i - 1].time for i in range(1, len(notes))]
    gaps = [gap for gap in gaps if gap > 0]  # 同一时刻（双押）不算间隔
    if gaps:
        stats.min_gap = min(gaps)
        stats.median_gap = median(gaps)

    span = (notes[-1].time - notes[0].time) / 1000
    stats.notes_per_second = round(len(notes) / span, 2) if span > 0 else 0.0

    first_time = notes[0].time
    buckets: dict[int, int] = {}
    for note in notes:
        second = int((note.time - first_time) // 1000)
        buckets[second] = buckets.get(second, 0) + 1
    stats.per_second = sorted(buckets.items())

    # ---- 逐键：移动 / 抗 / 占手 ---- #
    moves: list[MoveRecord] = []
    hold_spots: list[Spot] = []      # "长条占手冲突"的具体位置
    chord_spots: list[Spot] = []     # "同手覆盖和弦"的具体位置
    for group in groups:
        if len(group) > 1:
            stats.chords += 1
            stats.max_chord = max(stats.max_chord, len(group))
            used = {hands[index] for index in group if hands[index]}
            if len(used) < len(group):
                stats.same_hand_chord += 1
                group_time = notes[group[0]].time
                chord_spots.append(Spot(
                    time=group_time,
                    hand=sorted(h for h in used if h)[0] if used else "",
                    text=f"时刻 {group_time}（{len(group)} 个键，"
                         f"{len(used)} 只手承担）",
                ))
        for index in group:
            hand = hands[index]
            if hand is None or hand == "M":
                continue
            previous = sequence_note(_IndexView(notes), index, hand, hands)
            if previous < 0:
                continue
            delta_t = notes[index].time - notes[previous].time
            if delta_t <= SAME_HAND_MARGIN_MS:
                # 同一个和弦内的另一个键：没有"移动"这回事。
                # 口径与求解器一致（SAME_HAND_MARGIN_MS 内的键并成同一时刻），
                # 否则 1ms 错位和弦会报出 1216 轨道/秒 这种假读数。
                continue
            # 屏幕像素距离：两个轴刻度不同，必须分轴换算后勾股合成
            delta_x = screen_distance(notes[previous], notes[index], config.physics)
            speed = delta_x / (delta_t / 1000)
            stats.max_speed_seen = max(stats.max_speed_seen, speed)
            if speed > max_speed:
                stats.over_max_speed += 1
            elif speed > comfy_speed:
                stats.over_comfy_speed += 1
            # 该次移动发生时的横向刻度（六轨段落更小），逐次记录
            unit_px = scale_at(config.physics, notes[index].time)[0]
            moves.append(MoveRecord(
                from_line=notes[previous].line,
                to_line=notes[index].line,
                hand=hand,
                delta_t=delta_t,
                from_time=notes[previous].time,
                to_time=notes[index].time,
                delta_x=delta_x,
                speed=speed,
                from_x=notes[previous].x,
                to_x=notes[index].x,
                from_y=notes[previous].y,
                to_y=notes[index].y,
                unit_px_x=unit_px,
            ))
            if delta_t <= jack_window:
                stats.same_hand_close += 1
                # "同一个键位"是同一个**点**：同 x 但一个在地面一个在天空
                # （y 差 1 个满高 = 1.22 轨道）不是同一个键。
                if (notes[previous].x == notes[index].x
                        and notes[previous].y == notes[index].y):
                    stats.same_hand_same_x += 1
            # 留 SAME_HAND_MARGIN_MS 的容差：长条尾与下一个键相差 ≤2ms 时
            # 其实是同一个和弦时刻（谱面常写 1ms 错位），不算"还没松手就要按"。
            #
            # 隔离组（`timinggroup(noinput)`）的演出长条**不当占手**：求解器本来就
            # 允许组外的键与它同手，报告再报一次就是假警报（见 chart.isolated）。
            if _hold_duration(notes[previous]) > delta_t + SAME_HAND_MARGIN_MS \
                    and not isolated(notes[previous].noinput, notes[index].noinput):
                stats.hold_blocked += 1
                # 记下位置：报告里要指得出"哪一处"，只给一个数字没法定位。
                # 用"长条头 -> 下一个键"的两个时刻表示（与位移榜/手速榜同口径）。
                hold_time = notes[previous].time
                hold_tail = hold_time + int(_hold_duration(notes[previous]))
                # 注意 delta_t 是"长条头 -> 下一个键"，不是"长条尾 -> 下一个键"。
                # 真正有意义的是**重叠了多少**：下一个键落在按住期间内。
                overlap = hold_tail - notes[index].time
                hold_spots.append(Spot(
                    time=hold_time,
                    end_time=notes[index].time,
                    hand=hand,
                    text=f"{hand}手 长条 {hold_time}~{hold_tail}，"
                         f"但 {notes[index].time} 就有下一个键"
                         f"（提前 {overlap}ms，按住期间被要求再按）",
                ))

    stats.max_speed_seen = round(stats.max_speed_seen)
    # 汇总项一律**按每次移动自己的刻度**换算：六轨段落的刻度更小，
    # 用全局常量会把 6k 的位移算少。位移速度取所有移动里的最大者。
    lane_speeds = [
        lanes(m.delta_x, config.physics, m.unit_px_x) / (m.delta_t / 1000)
        for m in moves if m.delta_t > 0 and m.unit_px_x
    ]
    stats.max_lane_speed = round(max(lane_speeds, default=0.0), 2)
    # 阈值换算：用谱面里"最宽松"（最大）的刻度，避免六轨段落被误判为超速。
    # 六轨刻度更小 -> 同样像素对应的轨道数更少，取最大刻度是保守的一侧。
    unit_candidates = [m.unit_px_x for m in moves if m.unit_px_x]
    unit_px_for_limits = max(unit_candidates, default=config.physics.pixel_per_unit_x)
    stats.comfy_lane_speed = round(lanes(comfy_speed, config.physics, unit_px_for_limits), 2)
    stats.max_lane_speed_limit = round(lanes(max_speed, config.physics, unit_px_for_limits), 2)
    stats.max_move_lanes = round(max(
        (lanes(abs(m.to_x - m.from_x) * m.unit_px_x, config.physics, m.unit_px_x)
         for m in moves if m.unit_px_x), default=0.0), 2)
    stats.hard_moves = sorted(moves, key=lambda move: -move.speed)[:5]

    # ---- 手速榜：同手相邻两键的**最小间隔**（与位移无关）---- #
    # 口径（用户指定）：
    #   1. 只看非 arc 物件 —— tap / arctap / hold **头**（hold 尾不是一次"落下"）；
    #   2. 间隔按"头 -> 下一个键"算，不改成"尾 -> 下一个键"；
    #   3. 同手：相邻两键必须判给同一只手；
    #   4. 并列的间隔只拿一对举例，但报出总组数。
    stats.fastest_gaps = _fastest_gaps(notes, hands)
    if stats.fastest_gaps:
        top = stats.fastest_gaps[0]
        stats.fastest_gap_ms = top.gap_ms
        stats.fastest_gap_tied = sum(1 for g in stats.fastest_gaps
                                     if g.gap_ms == top.gap_ms)

    # ---- 交互率：只看间隔 <= dense_gap 的相邻键 ---- #
    for index in range(1, len(notes)):
        if notes[index].time - notes[index - 1].time > dense_gap:
            continue
        before, after = hands[index - 1], hands[index]
        if not before or not after or "M" in (before, after):
            continue
        if before == after:
            stats.repeated += 1
        else:
            stats.alternating += 1

    # ---- 结论性提示 ---- #
    # 只要能明确列出位置，就把位置一起给出来（见 Spot）：只报计数读者无法回查。
    if stats.none:
        stats.warnings.append(Warning(f"仍有 {stats.none} 个物件没有标记"))
    if stats.same_hand_chord:
        stats.warnings.append(Warning(
            f"有 {stats.same_hand_chord} 个和弦被同一只手覆盖"
            f"（可能存在跨度过大或原生标记冲突）",
            chord_spots, limit=WARNING_SPOT_LIMIT))
    if stats.hold_blocked:
        stats.warnings.append(Warning(
            f'有 {stats.hold_blocked} 处"长条还没松手就要按下一个键"',
            hold_spots, limit=WARNING_SPOT_LIMIT))
    if stats.over_max_speed:
        stats.warnings.append(Warning(
            f"有 {stats.over_max_speed} 处单手移速超过极限 "
            f"{stats.max_lane_speed_limit} 轨道/秒（物理上很勉强）",
            [Spot(time=move.from_time, end_time=move.to_time,
                  hand=move.hand, text=move.describe(config.physics))
             for move in sorted(moves, key=lambda item: -item.speed)
             if move.speed > max_speed],
            limit=WARNING_SPOT_LIMIT))
    if abs(stats.balance) > 0.25:
        stats.warnings.append(Warning(
            f"左右手使用量偏差 {stats.balance * 100:.0f}%（超过 25%）"))
    return stats


def _hold_duration(note: Note) -> float:
    """长条尾巴长度（没写 duration 就当 0，仅用于报告里的占手判断）。"""
    duration = note.duration
    return duration if duration is not None else 0.0


#: 注意点里最多列几处具体位置（多了会把报告淹掉）。``None`` = 全列。
WARNING_SPOT_LIMIT = 5

#: 手速榜只看这三类（"手要落下去按一下"的物件）。
#: **不含 hold 尾**，也不含音弧 —— 它们不是"再落一次"。
SPEED_NOTE_TYPES = ("tap", "hold", "arctap")


def _fastest_gaps(notes: list[Note], hands: list[str | None],
                  limit: int | None = None) -> list[GapRecord]:
    """同手相邻两键的时间间隔，从快到慢（**手速榜**）。

    只看 :data:`SPEED_NOTE_TYPES`（tap / arctap / hold **头**）：
    hold 参与排名时用的是它的**头**时间，间隔就是"头 -> 下一个键"。

    同一只手在同一时刻的两个键（双押）间隔为 0，**不算**手速（不进榜）——
    那不是"两次落下"，是同时按下去，所以只取 ``gap > 0``。

    :param hands: 每个物件的最终手标记（与 ``notes`` 同序）
    :param limit: 只留前多少条。**默认 ``None`` = 全部** ——
        截断会让"并列组数"少算（曾经截到 10 条，把 27 组的并列报成 10 组），
        所以统计一律用全量，只在显示时才截。
    """
    records: list[GapRecord] = []
    # 只保留参与排名的下标，然后**按手分组**取相邻 —— 这样"同手相邻"
    # 的定义与 sequence_note 一致：这只手上一次按的是什么，中间隔了几只手不管。
    for hand in HANDS:
        indexes = [i for i, note in enumerate(notes)
                   if note.type in SPEED_NOTE_TYPES and hands[i] == hand]
        for previous, current in zip(indexes, indexes[1:]):
            gap = notes[current].time - notes[previous].time
            if gap <= SAME_HAND_MARGIN_MS:
                # 同一个和弦（谱面常写 1ms 错位，如 53051 arctap + 53052 hold）：
                # 玩家是按一个和弦打下去的，不是"这个手指再落一次"。
                # 放进榜里会报出 BPM 15000 这种没有意义的读数。
                continue
            records.append(GapRecord(
                gap_ms=gap,
                hand=hand,
                from_line=notes[previous].line,
                to_line=notes[current].line,
                from_x=notes[previous].x,
                to_x=notes[current].x,
                from_y=notes[previous].y,
                to_y=notes[current].y,
                from_time=notes[previous].time,
                to_time=notes[current].time,
            ))
    records.sort(key=lambda r: r.gap_ms)
    return records if limit is None else records[:limit]


class _IndexView:
    """``sequence_note`` 只需要 ``notes``，这里做个最小适配。"""

    def __init__(self, notes: list[Note]) -> None:
        self.notes = notes
        self.n = len(notes)


# --------------------------------------------------------------------------- #
# 对比 / 解释
# --------------------------------------------------------------------------- #

def compare(before: Stats, after: Stats, before_label: str = "拆谱前",
            after_label: str = "拆谱后") -> str:
    """两份指标对比：返回可打印的多行文本。"""
    def ratio(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    rows = [
        ("物件数", before.total, after.total),
        ("L / R", f"{before.L} / {before.R}", f"{after.L} / {after.R}"),
        ("无标记", before.none, after.none),
        ("同手近距离连按（抗）", before.same_hand_close, after.same_hand_close),
        ("同键位连按", before.same_hand_same_x, after.same_hand_same_x),
        ("最大单手移速（轨道/秒）", before.max_lane_speed, after.max_lane_speed),
        ("最大横向位移（轨道）", before.max_move_lanes, after.max_move_lanes),
        ("超过舒适移速", before.over_comfy_speed, after.over_comfy_speed),
        ("超过极限移速", before.over_max_speed, after.over_max_speed),
        ("长条占手冲突", before.hold_blocked, after.hold_blocked),
        ("交互率", ratio(before.alternation_ratio), ratio(after.alternation_ratio)),
        ("左右偏差", ratio(before.balance), ratio(after.balance)),
    ]
    width = max(len(str(row[0])) for row in rows)
    cell = max(10, len(before_label) + 1, len(after_label) + 1)
    head = f"{'指标'.ljust(width)}  {before_label.ljust(cell)}{after_label}"
    lines = [head, "-" * len(head)]
    lines.extend(f"{str(name).ljust(width)}  {str(a).ljust(cell)}{b}" for name, a, b in rows)
    return "\n".join(lines)


@dataclass
class Explanation:
    """一个物件的判定依据。"""

    line: int
    time: int
    x: float
    y: float
    side: str
    locked: bool
    center: float
    bias: float
    reason: str

    def describe(self) -> str:
        star = "*" if self.locked else " "
        # x/y 都用**两位小数**：横坐标只在 0~2 之间，取整会让所有值变成 0/1。
        # 同时标出所在轨道（1~4 轨 / 0~5 轨，1 轨 = 0.5 坐标），方便肉眼定位。
        return (f"第 {self.line:>4} 行  {self.time:>6}ms  "
                f"x={self.x:+.2f} y={self.y:.2f}  "
                f"{self.side}{star}  {self.reason}")


def explain(result: SplitResult, limit: int | None = 40,
            start: int = 0, end: int | None = None) -> list[Explanation]:
    """逐键解释：每个物件为什么被判给这只手（``--explain`` 用它）。

    :param limit: 最多返回多少条（``None`` = 全部）
    """
    scoring = result.scoring
    if scoring is None:
        return []
    notes = scoring.notes
    hands = result.hands
    stop = len(notes) if end is None else min(end, len(notes))
    rows: list[Explanation] = []

    for index in range(max(0, start), stop):
        note = notes[index]
        hand = hands[index]
        if hand is None:
            continue
        center = scoring.centers[index]
        bias = scoring.bias[index]
        # 本键时刻的横向刻度（六轨段落更小），保证"轨道"换算与谱面一致
        unit_px = scale_at(scoring.physics, note.time)[0]
        reasons: list[str] = []

        if note.locked:
            reasons.append("原生 side 标记（锁定）")
        else:
            if abs(bias) > scoring.space.neutral_band:
                direction = "左" if bias < 0 else "右"
                # 偏差用"轨道"讲：像素没有体感，而玩家能直接数轨道
                reasons.append(
                    f"局部中心 {center:+.2f}，本键偏{direction} "
                    f"{lanes(abs(bias), scoring.physics, unit_px):.2f} 轨道")
            expect = "L" if note.x <= scoring.space.center_x else "R"
            if hand != expect:
                reasons.append(f"位置 {note.x:+.2f} 偏{'左' if expect == 'L' else '右'}（反手）")
            else:
                reasons.append(f"位置 {note.x:+.2f} 归{expect}手")

        previous = sequence_note(scoring, index, hand, hands)
        if previous >= 0:
            delta_t = note.time - notes[previous].time
            delta_x = screen_distance(notes[previous], note, scoring.physics)
            if delta_t <= scoring.cost.jack_window:
                reasons.append(
                    f"{delta_t}ms 前{hand}手还在 "
                    f"({notes[previous].x:+.2f},{notes[previous].y:.2f})"
                    f"（相隔 {lanes(delta_x, scoring.physics, unit_px):.2f} 轨道）")
        if scoring.hold_duration[index] > 0:
            reasons.append(f"长条占手 {round(scoring.hold_duration[index])}ms")

        rows.append(Explanation(
            line=note.line,
            time=note.time,
            x=note.x,
            y=note.y,
            side=hand,
            locked=note.locked,
            center=center,
            bias=bias,
            reason="；".join(reasons),
        ))

    return rows[:limit] if limit is not None else rows


