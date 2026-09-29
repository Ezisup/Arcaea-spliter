"""拆谱核心：物理/空间代价 + 束搜索 + 局部优化。

术语（沿用需求里的说法）：

- **拆**：把连续的键分给左右手交替
- **抗**：连续的键交给同一个手（jack）—— 有时必须、有时是设计
- **双押 / 和弦**：同一时刻的多个键 -> 分开给两只手

模型：把"拆谱"看成一个序列决策问题。按时间顺序扫过所有物件，
每一步决定每个物件归哪只手，代价由五块组成：

1. ``move``（物理）：这只手要移动到下一个键，速度越高越贵；
   长条期间手被占住，硬要去按别的键再罚一笔 ``busy``。
2. ``jack``（抗）：同一个手在很短时间里连按两下。**越接近同一个位置罚得越狠**，
   所以"同位置连按"很便宜，而低密度扫动交给同一只手也不贵。
3. ``hardness``（手感）：只看时间间隔。一根手指 150ms 内连点两下，
   和两根手指交替，难度不一样 —— 这是高密度必须拆交互的主要理由。
4. ``bias``（空间）：左边给左手、右边给右手（动态 + 静态两层叠加）。
5. ``switch`` / ``balance``：别每下都换手、别偏科。

硬约束：原生 ``side=`` 标记（锁定键）；同一只手在同一时刻不能覆盖超过
``max_reach`` 的两个键。长条占手是软约束（``busy`` 罚分）——
有解时不会选，没解时不会让程序崩掉。

求解：束搜索（beam search）拿到一个不错的解 -> 邻居翻转 / 交换 / 镜像 /
前缀翻转做局部优化。
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Iterator, Mapping, Sequence

from .chart import Note, isolated
from .config import Config, resolve_width_expr

HANDS: tuple[str, str] = ("L", "R")
OTHER: dict[str, str] = {"L": "R", "R": "L"}

DEFAULT_NOTE_WIDTH = 120.0


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #

@dataclass
class ScoreParts:
    """一份拆解的代价明细（便于调参时看是哪一项在起作用）。"""

    move: float = 0.0
    jack: float = 0.0
    bias: float = 0.0
    switch: float = 0.0
    cross: float = 0.0
    busy: float = 0.0
    balance: float = 0.0

    @property
    def total(self) -> float:
        return (self.move + self.jack + self.bias + self.switch + self.cross
                + self.busy + self.balance)

    def as_dict(self) -> dict[str, float]:
        return {
            "move": round(self.move, 3),
            "jack": round(self.jack, 3),
            "bias": round(self.bias, 3),
            "switch": round(self.switch, 3),
            "cross": round(self.cross, 3),
            "busy": round(self.busy, 3),
            "balance": round(self.balance, 3),
            "total": round(self.total, 3),
        }


@dataclass
class SplitResult:
    """拆谱结果。

    :attr:`hands` 按 ``note.index``（时间序）索引；:attr:`side_by_note` 按物件对象，
    回写谱面用它最稳 —— ``parse_chart`` 的排序会让 index 和原始文件顺序不一致。
    """

    hands: list[str | None] = field(default_factory=list)
    side_by_note: dict[Note, str | None] = field(default_factory=dict)
    score: ScoreParts = field(default_factory=ScoreParts)
    warnings: list[str] = field(default_factory=list)
    groups: list[list[int]] = field(default_factory=list)
    scoring: "Scoring | None" = None

    def text(self) -> str:
        """把标记拼成 ``'LRLR...'``（调试和测试用）。"""
        return "".join(hand or "?" for hand in self.hands)

    def counts(self) -> dict[str, int]:
        """统计标记数量。"""
        result = {"L": 0, "R": 0, "M": 0, "none": 0}
        for hand in self.hands:
            result[hand if hand in result else "none"] += 1
        return result


# --------------------------------------------------------------------------- #
# 分组
# --------------------------------------------------------------------------- #

def build_groups(notes: Sequence[Note]) -> list[list[int]]:
    """把物件按判定时间分组：同一时刻的算一组（双押/和弦）。"""
    groups: list[list[int]] = []
    for index, note in enumerate(notes):
        if groups and notes[groups[-1][-1]].time == note.time:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


# --------------------------------------------------------------------------- #
# 求解上下文
# --------------------------------------------------------------------------- #

class Scoring:
    """把「谱面 + 配置」编译成各种预计算量和代价函数。

    所有代价函数都是纯函数（只读 ``self``），方便单独拿出来实验。
    """

    def __init__(self, notes: Sequence[Note], config: Config, meta: dict | None = None,
                 constraints: object | None = None) -> None:
        """``constraints`` 是可选的**外部硬约束**（例如 Arcaea 的音弧颜色规则）。

        它只需要提供 ``allowed_hands(index) -> tuple[str, ...]``；
        传进来之后 :meth:`can_hand` 会把它和"锁定标记 / 不出界"一起判定，
        于是束搜索、局部优化、候选生成**全都**自动尊重它。
        这样拆谱核心不必知道任何 Arcaea 细节。
        """
        self.notes: list[Note] = list(notes)
        self.config = config
        self.constraints = constraints
        self._last_hands: list[str | None] | None = None
        self.physics = config.physics
        self.cost = config.cost
        self.space = config.space
        self.n = len(self.notes)
        # 每个物件所属的 ``timinggroup(noinput)`` 组号（0 = 普通）。
        # 隔离组只实现**单向**影响，见 :meth:`isolates`。
        self.noinput: list[int] = [note.noinput for note in self.notes]
        meta = meta or {}
        meta_width = meta.get("noteWidth")
        self.default_width = float(meta_width) if isinstance(meta_width, (int, float)) and meta_width else DEFAULT_NOTE_WIDTH

        self.sides: list[str | None] = [note.side for note in self.notes]
        self.widths: list[float] = [note.width or self.default_width for note in self.notes]
        self.locked_count = sum(1 for side in self.sides if side)
        self.free_count = sum(
            1 for index, side in enumerate(self.sides)
            if not side and not self._is_constraint_pinned(index)
        )
        # '0.2w' 这类写法按物件宽度换算；只在 Scoring 内部换算，不改配置对象
        self.jack_dead_zone = resolve_width_expr(config.cost.jack_min_distance, self.default_width)

        # 长条占手区间（按扫描顺序存），供 hold_conflict 快速查询
        # (起, 止, 扫描序号, 尾指 x, 尾指 y) —— 尾指位置是**一个点**，xy 都要记。
        # 这里是**硬约束**用的真实区间：Arcaea 的"异色原则"不变（长条区间内其它键
        # 必须换手）。"可提前松手"只体现在**权重**侧（hold_duration 收窄，
        # 见 physics.hold_tail_relax）。
        self._hold_spans: list[tuple[int, int, int, float, float]] = []
        self.hold_duration: list[float] = [0.0] * self.n
        self.hold_until: list[int] = [0] * self.n
        for index, note in enumerate(self.notes):
            duration = _hold_duration(note, self.physics)
            if duration > 0:
                # **权重侧**的占手时长：结尾可提前松手（Arcaea 无尾判），
                # 所以收窄 hold_tail_relax（不会早于起点）。硬约束（异色原则）
                # 仍按真实区间，见 _hold_spans。
                duration = max(0.0, duration - self.physics.hold_tail_relax)
            self.hold_duration[index] = duration
            self.hold_until[index] = note.time + int(duration)

        self.groups: list[list[int]] = build_groups(self.notes)
        self._order: list[int] = self.build_order()
        self._order_of: list[int] = [0] * self.n
        for position, index in enumerate(self._order):
            self._order_of[index] = position
        # `_order` 时刻表（升序）：`_feasible_region` 要按时间二分找"落在某条长条
        # 占手区间内"的物件，见 _hold_extended_stop。
        self._times: list[int] = [self.notes[index].time for index in self._order]
        # 给每个物件记下它所在的那个"同时刻组"，供可行性检查用
        self._group_of: list[list[int]] = [self.groups[0]] * self.n if self.groups else []
        for group in self.groups:
            for index in group:
                self._group_of[index] = group

        # "相对孤立"用的前后间隔（与时间上相邻物件的 ms 差）：bullet 的第三条口径
        by_time = sorted(range(self.n), key=lambda i: self.notes[i].time)
        far = 10 ** 9
        self._gap_before: list[int] = [far] * self.n
        self._gap_after: list[int] = [far] * self.n
        for pos, index in enumerate(by_time):
            if pos > 0:
                self._gap_before[index] = self.notes[index].time - self.notes[by_time[pos - 1]].time
            if pos + 1 < self.n:
                self._gap_after[index] = self.notes[by_time[pos + 1]].time - self.notes[index].time

        # 弧尾延续提示：这个键紧接在某条实体弧的尾巴上（同位置）时，记下那条弧
        # 的手（=它的 color）。bias_cost 会给这只手一点优惠。
        self.tail_hint: list[str | None] = [None] * self.n
        if _TAIL_HINT_BONUS > 0:
            # **只认时间上最近的那条弧尾**：同一位置常常有几条弧先后收尾
            # （例如一左一右两条天弧都停在同一点），谁先按字典序遍历到就听谁的
            # 会让结果依赖字典顺序、纯属噪声。用户参考谱的口径是"接住刚刚收尾
            # 的那条弧"（实测 aegleseeker 132307 / 136410 两簇）。
            nearest: list[float] = [float("inf")] * self.n
            for arcs in (getattr(self.constraints, "arcs_by_hand", None) or {}).values():
                for arc in arcs:
                    end = max(arc.t1, arc.t2)
                    hand = _COLOR_HAND.get(arc.color)
                    if hand is None:
                        continue
                    for index, note in enumerate(self.notes):
                        gap = note.time - end
                        if not 0 <= gap <= _TAIL_HINT_MS or gap >= nearest[index]:
                            continue
                        dx, dy = (note.x - arc.x2), (note.y - arc.y2)
                        if screen_span(dx, dy, self.physics, note.time) <= _TAIL_HINT_PX:
                            nearest[index] = gap
                            self.tail_hint[index] = hand

        # **同刻同 y、相差正好一个轨道**的一对键：两手必须分开，于是"谁左谁右"
        # 没有任何余地（用户口径：这种静态反手要狠狠罚）。
        self.lane_side: list[str | None] = [None] * self.n
        for index in range(self.n):
            note = self.notes[index]
            for member in self.group_of(index):
                if member == index:
                    continue
                mate = self.notes[member]
                if abs(mate.y - note.y) > _LANE_Y_EPS:
                    continue
                if _is_lane_pair(abs(mate.x - note.x)):
                    self.lane_side[index] = "L" if note.x < mate.x else "R"
                    break

        # 每个物件与**上一个判定物件**的间隔（"出张扣分"要看间隔够不够大）。
        self._step_gap: list[int] = [0] * self.n
        for position, index in enumerate(self._order):
            if position:
                self._step_gap[index] = (self.notes[index].time
                                         - self.notes[self._order[position - 1]].time)

        self.centers: list[float] = [0.0] * self.n
        self.bias: list[float] = [0.0] * self.n
        # 物件下标 -> 它作为长条的占手区间末端（非长条没有条目）
        self.hold_end: dict[int, int] = {}
        for position, index in enumerate(self._order):
            note = self.notes[index]
            if note.type != "hold":
                continue
            end = int(note.params.get("end", note.time + (note.params.get("duration") or 0)))
            self._hold_spans.append((note.time, end, position, note.x, note.y))
            self.hold_end[index] = end
        # 预先算好"每个物件落在哪些长条的占手窗口内"，避免每次评分都全表扫。
        # 窗口取**权重侧**的（含 hold_tail_relax 的提前松手），所以 100ms 那种
        # 短长条（本来就因 hold_min_duration 不算占手）不会进这张表。
        self._conflict_holds: list[list[int]] = [[] for _ in range(self.n)]
        for start, end, hold_position, hx, hy in self._hold_spans:
            hold_index = self._order[hold_position]
            occ_end = self.notes[hold_index].time + int(self.hold_duration[hold_index])
            for index, note in enumerate(self.notes):
                if index == hold_index:
                    continue
                if self.isolates(hold_index, index):
                    continue              # 隔离组的长条不对组外收占手代价
                # 与硬规则**同样的场合**：区间内（权重侧窗口，可提前 100ms 松手）
                # 前后各放宽 SAME_HAND_MARGIN_MS，以及"结尾同刻、落在别的位置"
                # （谱面会避免结尾同色，所以那时该换手）。
                if not (start - SAME_HAND_MARGIN_MS <= note.time
                        <= occ_end + SAME_HAND_MARGIN_MS):
                    continue
                if _same_pos(note.x, note.y, hx, hy) and (
                        abs(note.time - start) <= SAME_HAND_MARGIN_MS
                        or abs(note.time - end) <= 5):
                    continue                  # 起点/尾指本应对齐的键：同手才对
                self._conflict_holds[index].append(hold_index)

        self._build_centers()

        # ---- 多指段预扫描（在拆谱之前一次算清）---- #
        # 同一时刻需要的手指 = 该时刻的判定物件 + 仍在按住的长条 + 仍在跟的实体弧。
        # 达到 multi_finger_min 时，该时刻的 tap/arctap 进入"只守基本时空约束"模式：
        # 忽略 busy（手被长条占住）、长条头尾不能同色、长条区间异手这些**长条相关**限制，
        # 只剩"一键一手 / 原生标记 / 音弧颜色"这些最基本的。
        #
        # 理由：谱面里"2 条长条 + 1 个天键""1 条长条 + 1 条弧 + 2 个 tap"这类写法
        # 是**要玩家动第三根手指**的，逐条放宽约束只会头痛医头；一次扫清楚更省事。
        self.multi_finger: set[int] = set()
        #: 多指段里**直接定死**的手标记（见 _multi_finger_hands）
        self.multi_hand: dict[int, str] = {}
        self.multi_finger_times: list[tuple[int, int]] = []
        #: 多指时刻 -> 每只手的**弧触点数**（同一只手最多算一个；与判定物件同点的
        #: 不重复计）。_realizations 用它做"两手触点数差 <= 1"的硬约束。
        self.moment_arcs: dict[int, dict[str, int]] = {}
        #: 实在找不到合法手标记的物件（硬约束自相矛盾时才会出现，见 _realizations）
        self.unresolved: list[int] = []
        self._scan_multi_finger()

    # ---- 基本判断 ---- #

    def _is_constraint_pinned(self, index: int) -> bool:
        """被外部约束限制到只剩一只手 -> 它在权衡里等价于"已锁定"。"""
        if self.constraints is None:
            return False
        allowed = getattr(self.constraints, "allowed_hands", None)
        return allowed is not None and len(allowed(index)) <= 1

    def build_order(self) -> list[int]:
        """扫描顺序 = 按时间（组内按 x）排列的物件 index。"""
        return [index for group in self.groups for index in group]

    def px_to_units(self, value: float, time_ms: float | None = None) -> float:
        """把"按屏幕像素标定"的参数值换算成当前坐标刻度下的值。

        谱面坐标的刻度不一定是像素（Arcaea 是 1072 px/单位），凡是
        "多少像素算多远 / 多近"这类阈值，都必须过这一道换算，
        否则会出现"15 像素的中性带变成 15 个世界单位"那类错配。

        ``time_ms`` 给定时按**那一刻的**刻度换算（六轨模式下刻度更小，
        同样的像素阈值对应更大的坐标跨度）。
        """
        if time_ms is None:
            unit_x = self.physics.pixel_per_unit_x or 1.0
        else:
            unit_x = scale_at(self.physics, time_ms)[0]
        return float(value) / unit_x

    def scale_x_at(self, index: int) -> float:
        """第 ``index`` 个物件时刻的**横向**像素刻度（px/单位）。"""
        return scale_at(self.physics, self.notes[index].time)[0]

    def scale_y_at(self, index: int) -> float:
        """第 ``index`` 个物件时刻的**纵向**像素刻度（px/单位）。"""
        return scale_at(self.physics, self.notes[index].time)[1]

    def group_of(self, index: int) -> list[int]:
        """这个物件所在的"同时刻组"（同一判定时间的所有键）。"""
        return self._group_of[index]

    def is_locked(self, index: int) -> bool:
        """这个物件有没有原生标记（有 = 必须尊重）。"""
        return self.sides[index] is not None

    def can_hand(self, index: int, hand: str) -> bool:
        """这个物件能不能被这只手按到。

        判据只有两条：**原生锁定标记**、以及**外部硬约束**（如 Arcaea 的音弧颜色）。

        这里刻意**不做"位置是否出界"的判断**：谱面里能写出来的物件，游戏就认为
        它落在可判定范围内（Arcaea 的判定面本身也是斜视投影的，用"屏幕像素 ±640"
        去卡它只会误杀）。曾经加过这条检查，并且按像素刻度换算，结果把 `x=0.75`
        这类正常键的右手判成"越界"，导致每个键只剩一个候选、束搜索退化成单路径。
        """
        if hand not in HANDS:
            return False
        # 原生标记**永远优先**：它写在谱面里，改它就是改谱面
        locked = self.sides[index]
        if locked and locked != hand:
            return False
        if index in self.multi_finger:
            # 多指段：实体弧不再"占住整只手"，而是拿它的**采样点**去参与同 y 的
            # 左右排序（见 _multi_finger_hands）。所以双指那套"区间异手"在这里
            # 不适用 —— 一根手指在天上按着弧，同一只手的地面手指照样能点键。
            return True
        if self.constraints is not None and not _constraint_allows(self.constraints, index, hand):
            return False
        # 钉手本身**不在这里**判：它是"这个配置该这么分"的强烈偏好，
        # 真正落实的地方是 _realizations（候选只给钉手）与 _feasible_region
        # （不许改动钉手）。这样"钉手与硬约束打架"时不会两头都判非法。
        return True

    def isolates(self, source: int, target: int) -> bool:
        """``source`` 的约束是否**不该**作用于 ``target``。

        ``timinggroup(noinput)`` 是演出组：里面的物件不受玩家输入，但仍要**上手标记**
        （否则一堆拆好的键里冒出两个原色会很突兀）。所以取折中——

        - 组内物件**可以**受组外、以及**本组**的影响（正常参与约束）；
        - 组内物件**不能**影响组外。

        这一条正是"No input 的装饰长条把普通配置抬成多指段、于是整段无视硬约束"
        那个 bug 的修复点：noinput 的长条/弧只在本组内生效。

        没有 noinput 组时恒为 ``False``，所以普通谱面的行为**逐位不变**。
        """
        return isolated(self.noinput[source], self.noinput[target])

    def hold_tail_hand(self, index: int, hands: Sequence[str | None]) -> str | None:
        """这个键是否被"长条尾指"钉在某只手上（已经定了那只手时才返回它）。

        长条结束那一刻，手指停在**尾巴位置**；此时同一个位置的键交给同一只手
        才说得通（换手等于让手指先弹开再弹回来）。谱面里尾指与键常见 1~2ms
        的取整误差，所以时间给 ±5ms 的容差、位置要求严格相等。
        """
        if not self._hold_spans or index in self.multi_finger:
            return None                       # 多指段：只守基本时空约束
        note = self.notes[index]
        time, x, y = note.time, note.x, note.y
        position = self._order_of[index]
        for start, end, hold_position, tail_x, tail_y in self._hold_spans:
            if end < time:
                continue                      # 已经松手了
            if start >= time:
                break                         # 后面的长条都还没开始（按起点有序）
            if hold_position == position:
                continue
            if self.isolates(self._order[hold_position], index):
                continue                      # 隔离组的长条不钉组外的键
            if abs(time - end) <= 5 and _same_pos(x, y, tail_x, tail_y):
                hand = hands[self._order[hold_position]]
                if hand in HANDS:
                    return hand
        return None

    def is_tail_pair(self, first: int, second: int) -> bool:
        """两个物件是不是"长条 + 它尾指处的键"（同位置、时间差 <= 5ms，一个是 hold）。"""
        a, b = self.notes[first], self.notes[second]
        # 位置必须**整个点**都对上：地面长条的尾指停在 (x, 0)，正上方 (x, 1) 的
        # 天空键是另一个点，只比 x 会把它们误判成"同一根手指能同时按下"。
        if not _same_pos(a.x, a.y, b.x, b.y) or abs(a.time - b.time) > 5:
            return False
        return (a.type == "hold") != (b.type == "hold")

    def hold_conflict(self, index: int, hand: str, hands: Sequence[str | None]) -> bool:
        """``index`` 如果交给 ``hand``，会不会撞上"这只手正按着的长条"。

        长条与音弧是同一条原则：**手指按住期间不能去点别的键**。
        长条的手标记是拆谱时才定的，所以只能在拆到这一步时按已定的手检查。

        - 长条**起点**与其它键同刻：那条长条还没有"占用"，允许同手
          （手指点下去正好接住按住）；
        - 长条区间**之内**：不允许同手；
        - 长条**尾指那一刻且同一位置**：必须同手（见 :meth:`hold_tail_hand`），
          所以这里不算冲突。
        """
        if not self._hold_spans or index in self.multi_finger:
            return False                      # 多指段：只守基本时空约束
        time = self.notes[index].time
        position = self._order_of[index]
        tail_hand = self.hold_tail_hand(index, hands)
        note = self.notes[index]
        for start, end, hold_position, _tx, _ty in self._hold_spans:
            if end < time - 5:
                continue                      # 早就松手了
            if start > time + SAME_HAND_MARGIN_MS:
                break                         # 后面的长条都还没开始（按起点有序）
            if hold_position == position:
                continue
            other = self._order[hold_position]
            if self.isolates(other, index):
                continue                      # 隔离组的长条不挡组外的键
            if hands[other] != hand:
                continue
            if tail_hand == hand:
                continue                      # 尾指同手是应该的
            if abs(time - end) <= 5 and _same_pos(note.x, note.y, _tx, _ty):
                continue                      # 尾指那一刻的同位置键：同手才对
            if abs(time - start) <= SAME_HAND_MARGIN_MS \
                    and _same_pos(note.x, note.y, _tx, _ty):
                continue                      # 起点本应对齐的键：同手才对
            return True
        return False

    def _scan_multi_finger(self) -> None:
        """标记"同一时刻需要 >= multi_finger_min 根手指"的配置（见 __init__）。

        需求手指数 = 该时刻需要的**不同手指落点**个数：

        - 该时刻的每个判定物件算一个落点；
        - 那一刻仍在按住 / 跟住的非判定物件（长条、实体音弧）也算 —— **弧头、
          弧尾都在内**，因为弧头和弧尾都是要接住的判定点；
        - 但**落在同一个点**的只算一根手指："键压在弧头上""上一条弧的尾巴正好
          接下一条弧的头"都是一根手指顺下去的事（参考谱里这两种写法都很常见，
          重复计数会把普通配置误判成多指）。

        达到阈值时，该时刻的物件进入"只看空间"的多指模式。
        """
        minimum = int(self.physics.multi_finger_min)
        if minimum <= 0 or self.n == 0:
            return
        # 时刻 -> 该时刻的判定物件。**把 ±SAME_HAND_MARGIN_MS 内的键并成同一个
        # "时刻"**：谱面里常写 1ms 的错位和弦（例如 86004 的 tap 与 86005 的
        # hold），玩家是按一个和弦打的，左右要一起分；按精确时间戳拆开会让它们
        # 各自变成"孤立键"，既看不出多指、也分不出左右。
        at_time: dict[int, list[int]] = {}
        for index in sorted(range(self.n), key=lambda i: self.notes[i].time):
            time = self.notes[index].time
            for start in at_time:
                if time - start <= SAME_HAND_MARGIN_MS:
                    at_time[start].append(index)
                    break
            else:
                at_time[time] = [index]
        # 非判定物件：长条的落点固定；实体弧要按时刻求位置（弧可能在移动）
        holds: list[tuple[int, int, float, float, int]] = []
        for start, end, position, _tail_x, _tail_y in self._hold_spans:
            hold_index = self._order[position]
            hold = self.notes[hold_index]
            holds.append((start, end, hold.x, hold.y, self.noinput[hold_index]))
        arcs: list[object] = []
        arcs_by_hand = getattr(self.constraints, "arcs_by_hand", None) or {}
        for group in arcs_by_hand.values():
            arcs.extend(group)
        marked: dict[int, int] = {}
        for time, indexes in at_time.items():
            # 落点按**组**分桶（0 = 组外）。隔离组的物件只算给它自己和本组 ——
            # 组外的普通配置不该被装饰性的长条/弧抬成多指段（那会让整段无视硬约束）。
            spots_by_group: dict[int, set[tuple[float, float]]] = {}
            arcs_by_group: dict[int, dict[tuple[float, float], str]] = {}
            counted_by_group: dict[int, set[str]] = {}
            for index in indexes:
                spots_by_group.setdefault(self.noinput[index], set()).add(
                    _spot(self.notes[index].x, self.notes[index].y))
            for start, end, x, y, group in holds:
                if start <= time <= end:
                    spots_by_group.setdefault(group, set()).add(_spot(x, y))
            # 实体弧在**这一刻的采样点**（长度无关）：它同样是一个触点，
            # 并且带着自己**已经定好**的颜色（手）参与同 y 的左右排序。
            # 黑线（arctype=true）不参与：它的 color 只是占位符、没有任何语义，
            # 而挂在它上面的天键本来就会被正常流程算过一遍。
            for arc in arcs:
                lo, hi = min(arc.t1, arc.t2), max(arc.t1, arc.t2)
                if not (lo <= time <= hi):
                    continue
                group = getattr(arc, "noinput", 0)
                spot = _spot(arc.x_at(time), arc.y_at(time))
                hand = _COLOR_HAND.get(arc.color)
                points = arcs_by_group.setdefault(group, {})
                if hand and spot not in points:
                    points[spot] = hand
                # **同一只手的多条弧只算一个触点**：谱面允许同一时刻出现多条
                # 同色弧（新版本能解析），但判定上"当成一条"就够 —— 否则两条
                # 同色弧会被数成两根手指，把普通配置误抬成多指段，而多指段会
                # 整段无视"有弧时不得同色"这条硬约束（用户报的就是这个 bug）。
                counted = counted_by_group.setdefault(group, set())
                if hand is not None and hand in counted:
                    continue
                spots_by_group.setdefault(group, set()).add(spot)
                if hand is not None:
                    counted.add(hand)
            outer_spots = spots_by_group.get(0, set())
            outer_points = arcs_by_group.get(0, {})
            for group in sorted({self.noinput[i] for i in indexes}):
                hits = [i for i in indexes if self.noinput[i] == group]
                if not hits:
                    continue
                # 组内：组外落点 + **本组**落点。组外物件只看组外落点。
                merged = outer_spots | spots_by_group.get(group, set())
                if len(merged) < minimum:
                    continue
                # 组内**每个物件**都要定手（含长条）：多指配置里长条也是
                # 一根手指，同样"只看空间"。
                self.multi_finger.update(hits)
                # 手标记**直接定死**：弧的采样点 + 各物件的落点一起参与
                # "同 y 从左到右分左右"，不同键之间的前后关系一律不看。
                points = outer_points if group == 0 else {
                    **outer_points, **arcs_by_group.get(group, {})}
                if self.physics.multi_hand_pin:
                    self.multi_hand.update(self._multi_finger_hands(hits, points))
                else:
                    # 只保留两类**合法硬钉**：原生 side 标记，以及"骑在实体弧
                    # 采样点上的键听弧色"（弧色是谱面写死的）。几何分半一律取消，
                    # 自由键交给代价模型 + "±1 平衡"硬约束。
                    decided = self._multi_finger_hands(hits, points)
                    for index in hits:
                        x, y = self.notes[index].x, self.notes[index].y
                        if self.sides[index] or _spot(x, y) in points:
                            self.multi_hand[index] = decided[index]
                # 该时刻每只手的**弧触点数**（同一只手最多算一个；与判定物件同点的
                # 不重复计）—— _realizations 的 ±1 平衡硬约束要用。
                arcs_here = {"L": 0, "R": 0}
                counted = set()
                for spot, hand in points.items():
                    if hand in counted or spot in merged:
                        continue
                    counted.add(hand)
                    arcs_here[hand] += 1
                self.moment_arcs[time] = arcs_here
                marked[time] = max(marked.get(time, 0), len(merged))
        self.multi_finger_times = sorted(marked.items())

    def _multi_finger_hands(self, indexes: list[int],
                            arc_points: Mapping[tuple[float, float], str] | None = None
                            ) -> dict[int, str]:
        """多指配置的手标记：**只看空间**。

        规则（用户口径）：
        - **原生 ``side`` 标记优先**，绝不覆盖（改它就是改谱面）；
        - 每个物件按**绝对左右**定基准（x < 中线 -> 左，否则右）；
        - **同一个 y 上**有多个判定物件时，按 x 从左到右分成左右两半（左半 L、右半 R）。
          **实体弧的采样点不参与这个排序** —— 它只用来数"要几根手指"，以及给
          骑在它身上的键定弧色；让弧占"槽位"会把分半边界顶偏，把明显偏右的键
          判给左手（用户实测 99666 的 x=0.75 就是这么变成 L 的）；
        - 该 y 上只有一个触点、而这个触点上有实体弧时，直接听弧的颜色 ——
          没有左右可排，弧色是唯一的信息。

        "撞成同一只手"在多指配置里**不是错误** —— 一只手可以有几根手指
        （按住弧/长条的同时用别的指点键），所以这里不去强行拆开。
        """
        center = self.space.center_x
        arc_points = arc_points or {}
        out: dict[int, str] = {}
        by_y: dict[float, list[int]] = {}
        for index in indexes:
            locked = self.sides[index]
            if locked:
                out[index] = locked            # 原生标记优先
                continue
            by_y.setdefault(round(self.notes[index].y, 6), []).append(index)
        for y, group in by_y.items():
            # 1) **骑在实体弧采样点上的键**（天键基本都是这种）：手由弧色定死，
            #    不能再按空间左右去改 —— 弧的颜色是谱面写死的、改不了。
            #    多条同色弧也只是同一条（用户口径）。
            points: list[tuple[float, str | None, list[int]]] = []
            for index in group:
                x = self.notes[index].x
                arc_hand = arc_points.get(_spot(x, y))
                if arc_hand:
                    out[index] = arc_hand
                    continue
                points.append((x, None, [index]))
            # 2) **弧的采样点不参与"分左右"**。它们只负责两件事：数"这一刻要
            #    几根手指"（在 _scan_multi_finger 里）和"骑在弧上的键听弧色"
            #    （上面第 1 步）。曾经把它们也塞进下面的排序当"占位槽"，结果槽位
            #    序号被顶偏：y=0 上 [L弧 0.25, 键 0.75, R弧 1.25] 三槽、half=1.5，
            #    x=0.75 的键落在"槽 1"被判成左手 —— 与绝对/相对位置、手数平衡
            #    三种判据全都矛盾（用户指出"无论按哪个思路都说不通"）。
            points.sort(key=lambda item: item[0])
            if not points:
                continue
            if len(points) == 1:
                # 该 y 上没有左右可排 -> 有弧就听弧的颜色（弧色是唯一信息），
                # 否则按绝对左右。
                x, arc_hand, members = points[0]
                hand = arc_hand or ("L" if x < center else "R")
                for index in members:
                    out[index] = hand
                continue
            # 有左右可排：**只看判定物件**，按 x 严格分左右两半。
            half = len(points) / 2.0
            for slot, (_x, _arc_hand, members) in enumerate(points):
                hand = "L" if slot < half else "R"
                for index in members:
                    out[index] = hand
        # 兜底：同刻两个键被绝对规则判成同一只手 -> 按 x 左右分开。
        # 但若其中一个键带着**弧的颜色锚点**，就听弧的（弧的颜色改不了）。
        if len(indexes) == 2:
            first, second = indexes
            anchored = any(arc_points.get(_spot(self.notes[i].x, self.notes[i].y))
                           for i in indexes)
            if not anchored and out.get(first) == out.get(second):
                left, right = sorted(indexes, key=lambda i: self.notes[i].x)
                out[left], out[right] = "L", "R"
        self._balance_multi_hands(out, indexes, arc_points)
        return out

    def _balance_multi_hands(self, out: dict[int, str], indexes: list[int],
                             arc_points: Mapping[tuple[float, float], str]) -> None:
        """多指段里两只手的**触点数**之差不得超过 1（用户口径，就地改 ``out``）。

        "一边 3 个一边 1 个"是打不出来的 —— 那要求一只手同一刻按住 3 个点。
        这条规则的优先级**高于**"同 y 从左到右分左右"：空间排布再好看，
        只要数量差到 2 就必须挪。

        只挪**自由键**（没有原生标记的）；原生标记是谱面写死的，
        宁可数量不齐也不动它。
        """
        free = [i for i in indexes if not self.sides[i]]
        if len(free) < 2:
            return
        counts = self._multi_hand_counts(out, indexes, arc_points)
        # 挪"最靠近另一只手"的那些：按到对侧手的距离升序，先挪代价最小的。
        other = {"L": "R", "R": "L"}
        while abs(counts["L"] - counts["R"]) > 1:
            major = "L" if counts["L"] > counts["R"] else "R"
            movable = [i for i in free if out.get(i) == major]
            if not movable:
                return
            # 挑位置最靠对手侧的：major 为 L 时取 x 最大的（最靠右）
            pick = max(movable, key=lambda i: self.notes[i].x) if major == "L" \
                else min(movable, key=lambda i: self.notes[i].x)
            out[pick] = other[major]
            counts[major] -= 1
            counts[other[major]] += 1
            free.remove(pick)

    def _multi_hand_counts(self, out: Mapping[int, str], indexes: list[int],
                           arc_points: Mapping[tuple[float, float], str]
                           ) -> dict[str, int]:
        """这一刻两只手各要落几个触点（判定物件 + 实体弧，同点不重复计）。"""
        counts = {"L": 0, "R": 0}
        occupied: set[tuple[float, float]] = set()
        for index in indexes:
            hand = out.get(index)
            if hand not in counts:
                continue
            counts[hand] += 1
            occupied.add(_spot(self.notes[index].x, self.notes[index].y))
        # 实体弧每个手最多算一个触点（多条同色弧在判定上是"一条"），
        # 而且与判定物件落在一起的是同一根手指，不重复计。
        counted: set[str] = set()
        for spot, hand in arc_points.items():
            if hand not in counts or hand in counted or spot in occupied:
                continue
            counted.add(hand)
            counts[hand] += 1
        return counts

    def cross_cost(self, previous: int, previous_hand: str | None,
                   index: int, hand: str | None) -> float:
        """"挤在一起时的反手"加成：两键交错且距离落在"挤"的区间里时加罚。

        手指有宽度 —— 0.33 的反手比 1.0 的反手更难打。距离用**真实屏幕距离**
        （xy 分轴换算后勾股，不是只看 Δx），形状是**抛物线**：
        ``cross_near``（太近算同点）与 ``cross_far``（太远不算挤）处为 0、
        中点为峰值 ``cost.cross``。同手不算反手（那是"抗"的事）。"""
        if self.cost.cross <= 0 or hand not in HANDS or previous_hand not in HANDS:
            return 0.0
        if hand == previous_hand:
            return 0.0
        note, prev = self.notes[index], self.notes[previous]
        # 用**真实距离**（xy 分轴换算后勾股），不是只看 Δx
        scale = self.scale_x_at(index)
        distance = screen_span(note.x - prev.x, note.y - prev.y, self.physics, note.time)
        near = self.cost.cross_near * scale
        far = self.cost.cross_far * scale
        if not (near < distance < far):
            return 0.0                      # 太近算同点、太远不算挤
        # 反手：左边那个键给了右手、右边那个给了左手（左右按 x 判断）
        dx = note.x - prev.x
        crossed = (dx > 0 and previous_hand == "R") or (dx < 0 and previous_hand == "L")
        if not crossed:
            return 0.0
        # 抛物线：两端 0、中点峰值
        mid = (near + far) / 2.0
        half = max((far - near) / 2.0, 1e-9)
        shape = 1.0 - ((distance - mid) / half) ** 2
        return self.cost.cross * max(0.0, shape)

    def hand_cross_cost(self, index: int, hand: str, hand_of) -> float:
        """分给这只手之后会不会**落在另一只手的另一侧**（双手交叉）。

        左手在右、右手在左就是交叉。用户的口径：交叉比"多跑一段"更难受，
        所以宁可同手硬抗也不该交叉。另一只手还没落过键时不算。"""
        if self.cost.cross_hand <= 0 or hand not in HANDS:
            return 0.0
        other = "R" if hand == "L" else "L"
        other_index = hand_of(other)
        if other_index is None:
            return 0.0
        dx = self.notes[index].x - self.notes[other_index].x
        crossed = (hand == "L" and dx > 0) or (hand == "R" and dx < 0)
        if not crossed:
            return 0.0
        # 两只手都在动（另一只手刚落下不久）-> 真正的"两手交叉"，更重；
        # 另一只手短时间没动（这只手单方面伸过去）-> 轻一些。
        gap = self.notes[index].time - self.notes[other_index].time
        factor = _CROSS_MUTUAL if gap <= _CROSS_MUTUAL_MS else _CROSS_STILL
        return self.cost.cross_hand * factor

    def hand_catch(self, hand: str, t_from: int, t_to: int):
        """这只手在这段时间里需要"接住"的音弧弧头（没有就返回 None）。

        拆谱核心不关心那是音弧，只问外部约束"这只手是不是要赶去接一个长按类目标"。
        """
        catch = getattr(self.constraints, "hand_catch", None)
        if catch is None:
            return None
        return catch(hand, t_from, t_to)

    def hold_conflict_cost(self, index: int, hand: str, hand_of) -> float:
        """这只手此刻**正按着别处的长条**时，为"同时按两处"收的代价。

        代价随"被占住的位置到这个键的距离"增长 —— 手被钉在长条上，越远越勉强。

        但**多指时刻**例外：那一刻手本来就有多根手指可用（拇指按长条、食指打键），
        收这笔"同时按两处"的账既不符合打法，也会把正确的分法顶掉（用户口径：
        被"锁"的那只手不能影响后续的自由键）。
        这样"哪只手的长条更近"就成为一个可比较的空间信号（用户人工核对过的
        chaos 谱里，多指段正是按"更近的那条 hold"分的）。

        必须**按物件统一收**，不能像以前那样只在"上一键恰好是那条 hold"时
        才收 —— 否则"让同一只手连着打几个键"就完全免掉了这笔账，而"换手"
        反而因为新手的上一键是 hold 被罚，方向整个反了。
        """
        if index in self.multi_finger:
            return 0.0

        if index in self.multi_finger:
            return 0.0                        # 多指段：只手被占住这件事不算数
        holds = self._conflict_holds[index]
        if not holds:
            return 0.0
        tail_hand = self.hold_tail_hand(index, _HandView(self, hand_of))
        total = 0.0
        for hold_index in holds:
            if hand_of(hold_index) != hand:
                continue
            if tail_hand == hand:
                continue                      # 尾指同手是应该的，不算冲突
            # 平罚：**不再**按"离那条已染色弧/长条多远"定位置 —— 用户指出这个
            # 定位是错的（多指段现在只看绝对左右 + 同 y 的左右关系，见
            # _scan_multi_finger），这里只负责"手被占住"这件事本身。
            total += self.cost.busy
        return total

    def can_fill_group(self, group: list[int]) -> bool:
        """这个时刻组能不能"一键一手"分完（一只手同一时刻只能按一个键）。

        双押总是可以（除非两个键被锁成同一只手）；**同一时刻 3 个以上的键不行** ——
        两只手按不下三个同时的键。这类组会被排除在硬约束检查之外：它本来就
        没有合法解，不该让整条谱面的可行性检查全部失败。
        """
        reach = self.px_to_units(self.physics.max_reach)
        used: dict[str, float] = {}          # 手 -> 这只手已覆盖的跨度范围
        for index in group:
            x = self.notes[index].x
            for hand in HANDS:
                if not self.can_hand(index, hand):
                    continue
                if hand in used and x - used[hand] > reach:
                    continue                     # 这只手够不到两个键
                used[hand] = max(used.get(hand, x), x) if hand in used else x
                break
            else:
                return False
        return True

    def may_jack(self, first: int, second: int) -> bool:
        """只有两个自由键之间的连续同手才算"抗"（锁定的键是既定事实，不该被罚）。"""
        return not self.is_locked(first) and not self.is_locked(second)

    # ---- 局部中心（只用于报告/观察，不参与代价） ---- #

    def _build_centers(self) -> None:
        """算每个物件"前后 window 内邻居的加权中心"（带时间衰减）。

        这个量描述的是"这一段整体偏哪边"，供 ``--explain`` 和调参观察用。
        它**不参与代价**，原因见 README 第 2.3 节（扫动折返点会翻车）。
        """
        notes = self.notes
        window = max(self.space.window, 1)
        for index, note in enumerate(notes):
            total = 0.0
            weight = 0.0
            for other in notes:
                delta = abs(other.time - note.time)
                if delta > window:
                    continue
                decay = math.exp(-delta / 300)
                total += other.x * decay
                weight += decay
            center = total / weight if weight > 0 else note.x
            self.centers[index] = center
            self.bias[index] = note.x - center

    # ---- 代价函数 ---- #

    def screen_distance(self, first: int, second: int) -> float:
        """两个物件在**屏幕像素**上的实际距离。"""
        return screen_distance(self.notes[first], self.notes[second], self.physics)

    def home_distance(self, index: int, hand: str, home: Mapping[str, float]) -> float:
        """起手手位到某个物件的屏幕距离。

        起手位是"手停在判定面近端"的位置，即 ``y = 0``；所以从起手位到天空物件的
        位移同样有 y 分量，必须一起算（默认的移动距离就是这条 xy 勾股长度）。
        """
        note = self.notes[index]
        return screen_span(note.x - home[hand], note.y, self.physics, note.time)

    def hand_release(self, hand: str, t_from: int, t_to: int) -> tuple[int, float, float] | None:
        """这只手在 ``(t_from, t_to)`` 之间被**非判定物件**钉住的最后一次落点。

        目前只有音弧会用到它：弧和长条一样是"按住不放"，手跟到 ``t2``，
        **结束那一刻停在尾巴 ``(x2, y2)``**。拆谱核心不关心那是音弧还是别的什么，
        只问外部约束"你有没有把这只手钉在某个位置上"（没传约束时永远返回 None）。
        """
        release = getattr(self.constraints, "hand_release", None)
        if release is None:
            return None
        return release(hand, t_from, t_to)

    def relationship_previous(self, previous: int | None, current: int,
                              hand: str) -> int | None:
        """把"被长条 / 音弧占住的上一个键"视同**不存在**（用户口径）。

        拇指先按住长条、或者这只手刚开始追一条音弧，食指来接后面的自由键 ——
        这一下没有"相互关系"：``move`` / ``jack`` / ``hardness`` 全部归零，
        那只手只由**这个键自己的起始位置**决定。之后的自由键则在这条"食指链"
        上自己结算（前一个自由键就是它的 previous）。

        - 长条：上一个键是**仍按着**的长条（``hold_until >= current``）；
        - 音弧：这只手在 ``(prev, current]`` 之间**新开始**追一条弧（弧的起点落在
          区间内）—— 只在"刚进入"这一下生效，弧中间的后续自由键照常结算。
        """
        if previous is None:
            return None
        if self.hold_until[previous] >= self.notes[current].time:
            return None
        arcs = getattr(self.constraints, "arcs_by_hand", None) or {}
        t_prev, t_now = self.notes[previous].time, self.notes[current].time
        for arc in arcs.get(hand, ()):
            if t_prev < min(arc.t1, arc.t2) <= t_now:
                return None
        return previous

    def move_cost(self, first: int, second: int, hand: str) -> float:
        """从 ``first`` 移到 ``second`` 的代价（这只手）。"""
        notes = self.notes
        delta_t = notes[second].time - notes[first].time
        # 同一时刻（双押/和弦）：手是"已经就位"的两个落点，没有移动过程
        if delta_t <= 0:
            return 0.0
        physics = self.physics
        cost = 0.0

        # ---- 手最早能开始移动的时刻，以及那一刻手在哪个位置 ---- #
        start_x, start_y = notes[first].x, notes[first].y
        # 隔离组的长条**不当占手**：它的占手效应不得外溢到组外的键（见 isolates）。
        holds_here = self.hold_duration[first] > 0 and not self.isolates(first, second)
        if holds_here:
            tail = notes[first].time + self.hold_duration[first]
            # 按住期间手被钉在长条上：这段时间不能算作"可以用来移动的时间"。
            # 手指在长条**结束那一刻停在尾巴位置**，所以从松手到下一个键
            # 只有 (dt - 按住时长) 的时间可用。
            if tail >= notes[second].time:
                # 手还按在长条上（下一个键落在按住期间**或正好同刻**）：
                # 手指被钉在长条上、根本没在移动，所以**不做位移速度判定**
                # （否则 travel_time=0 会让速度项炸成天文数字）。这笔冲突的代价
                # 统一由 hold_conflict_cost() 收，见那里。
                #
                # 用 >= 而不是 >：**尾指同刻**必须和"按住期间"同样处理。
                # 尾指同刻只有两种可能，都不是"移动"：
                #   - 落在原地（arc 尾原地 tap 那种）：手就在那儿，松手即按下，
                #     合法且不该罚（hold_tail_hand 也正要求它同手）；
                #   - 落在别处：手要同时在两处，本来就不合理，由 busy 收代价。
                # 曾经写成 >，于是恰好同刻时落到下面"释放点压在键上"的分支，
                # 白收 40.0，而且和 busy 重复收费（同手 total 64 而非 20）。
                return 0.0
            release_time = tail
        else:
            release_time = notes[first].time
        # **不再**把"这只手正扛着音弧"当成位移起点/时间起点（用户口径）：
        # 拇指追弧，食指照样打键，食指这条链的位移必须从**它自己上一个自由键**
        # 算起。旧写法把起点强行挪到弧尾，于是正确分法凭空背上巨额 move
        # （cc_3 90k~110k 实测 10 笔 40 分封顶罚、合计 +391，相位被整体镜像）。
        # "刚进入弧/长条"那一下由 relationship_previous 归零。
        travel_time = max(notes[second].time - release_time, 0)

        # 位移一律用 xy 勾股算屏幕实际距离，不能只看 x。
        # 注意：max_reach 是"同一时刻单手能覆盖多大跨度"（和弦用的），
        # 不能拿来卡"前后两个键之间的位移" —— 手可以在两拍之间慢慢挪很远。
        # 相邻键之间只看速度（像素/秒），上限由 comfy_speed / max_speed 表达。
        delta = screen_span(notes[second].x - start_x, notes[second].y - start_y, physics)
        speed = delta / (max(travel_time, 1) / 1000)
        # 允许的舒适移速随位移衰减（走越远越吃力），免费区也跟着一起缩。
        comfy = physics.comfy_speed
        if speed > 0:
            ratio = speed / max(comfy, 1e-9)
            # 终点是长按类（hold，或路上要接一条音弧）：只有最高一级判定，
            # 到位难度明显低于"点→点 / 按→点"，所以惩罚显著更轻。
            # 只对"**移动到音弧**"减轻（终点要接住一条实体弧弧头）：
            # 音弧只有最高一级判定，赶去接它的到位要求比点类松。
            # **移动到 hold 暂不减轻**（用户要求先保持原样）。
            arrival = 1.0
            if notes[first].type != "hold" and \
                    self.hand_catch(hand, notes[first].time, notes[second].time) is not None:
                arrival = self.cost.arc_arrival
            power = self.cost.move_speed_power
            shaped = ratio * ratio if power == 2.0 else ratio ** power
            cost += self.cost.move * shaped * (0.5 if ratio < 1 else 1.0) * arrival
        # 峰值上限不衰减：那是手指本身的极限速度，与走了多远无关。
        if speed > physics.max_speed:
            ratio = speed / physics.max_speed - 1
            cost += 40 * ratio * ratio
        # 一次移动的代价封顶：travel_time 趋近 0 时速度项会平方级炸开
        # （实测能到 25 万），把整份谱的判断全压掉。再离谱也只是"这一下做不到"，
        # 不该让它主导全局排序。
        return min(cost, _MOVE_COST_MAX)

    def is_bullet(self, first: int, second: int) -> bool:
        """这两个键是不是"子弹"（原地弹一下）。

        用户的**三条口径缺一不可**：

        1. 只有两个且相邻（调用方已经保证了"同手相邻"）；
        2. 完全相同的 xy（放宽到 ``jack_min_distance``，默认 0.1 单位）；
        3. **相对孤立** —— ``bullet_isolation`` 窗口内没有别的判定物件。

        第 3 条最容易被漏掉：密集段（例如 16 分交互串）里"同位置连着按"仍然是
        硬抗，必须罚；漏了它整串都会被当成 bullet 免罚而判反。

        ``bullet_isolation <= 0`` 表示**没有配置孤立性窗口**（退回"只看位置"的老口径），
        此时不检查第 3 条（保持旧行为：同位置连按就是 bullet）；配置了窗口
        （Arcaea 150ms）才会把密集段里的同键连按排除在 bullet 之外。
        """
        if self.jack_dead_zone <= 0:
            return False
        px = self.scale_x_at(second)
        distance = screen_span(self.notes[second].x - self.notes[first].x,
                               self.notes[second].y - self.notes[first].y,
                               self.physics, self.notes[second].time)
        if distance > self.jack_dead_zone * px:
            return False
        # 3'. **间隔够大的同位置二连**：手指完全来得及再落一次，不必满足孤立性
        #     （用户实测 gengaozo：98ms 连续 16 分里的同位置连按就是子弹；
        #      而 75ms 那档仍按硬抗罚）。
        delta_t = self.notes[second].time - self.notes[first].time
        if self.cost.bullet_min_gap > 0 and delta_t >= self.cost.bullet_min_gap:
            tol = self.cost.bullet_stream_tol
            if tol > 0:
                # 只在**节奏规整的连续二连**里免孤立：前后相邻间隔与这一对相当
                lo, hi = delta_t * (1 - tol), delta_t * (1 + tol)
                for gap in (self._gap_before[first], self._gap_after[second]):
                    if gap and not lo <= gap <= hi:
                        break
                else:
                    return True
            else:
                return True
        margin = self.cost.bullet_isolation
        if margin > 0 and (self._gap_before[first] < margin
                           or self._gap_after[second] < margin):
            return False
        return True

    def jack_cost(self, first: int, second: int, hand: str) -> float:
        """抗（同手连续两下）的代价 —— 拆谱里最核心的一项。

        判据是"再落一次指有多难受"，由两件事决定：

        - **间隔**：越短越难受（时间因子）。超过 ``jack_window`` 就不算抗。
        - **速度**：这一下要求手移动多远 / 有多快。

          - 位置几乎没动（<= 死区）：原地起落，很轻松，默认不罚；
          - 速度超过舒适移速：已经是"硬抗"了，加一笔超速加成；
          - 速度远小于舒适移速：手刚摆好就要再按，越近越难受。

        于是同一套公式能同时说清：80ms 跨 200px 的左右跳必须交互、
        100ms 跨 100px 的扫动可以一只手连着扫、同键位 250ms 连按随意。

        "同键位连按"有两种，必须分开（用户口径）：

        - **bullet**（同位置 + 相对孤立）-> 上面直接返回 0，轻松；
        - **密集流里的同键连按** -> 不是 bullet，按 ``jack_time`` 收一笔
          （``spatial`` 在死区内是 0，于是强度就是 ``jack_time``）。
        """

        delta_t = self.notes[second].time - self.notes[first].time
        if delta_t <= 0 or delta_t > self.cost.jack_window:
            return 0.0
        taper = _taper_factor(delta_t, _TAPER_JACK_END)
        # 全部换算到**屏幕像素**口径再比：距离走 xy 勾股（不能只看 x），
        # 而 width / 死区 / jack_decay 本来就是像素量 —— 之前 `distance` 是世界
        # 单位（<= 1.4）却拿去和 width*jack_decay (=168) 比，空间衰减项恒为 ~1，
        # 等于"抗的空间衰减"整个失效。见 README 第 2.3 节。
        px = self.scale_x_at(second)
        width = min(self.widths[first], self.widths[second], self.default_width)
        distance = self.screen_distance(first, second)
        width_px = width * px
        dead = self.jack_dead_zone * px
        time_factor = (self.cost.jack_window - delta_t) / self.cost.jack_window

        if self.is_bullet(first, second):
            # bullet（同位置 [+ 相对孤立]）：原地弹一下，**按 jack_time 折价** ——
            # Arcaea 取 0.2（五分之一，轻松很多）；取 0 就是完全免费。
            # 密集流里的同键连按（未通过孤立性）不走这里：它们的 spatial 在死区内
            # 是 0，于是强度落到 ``(1 - spatial) * jack_time``，也就是同一笔价 ——
            # 关键是 jack_time **不能是 0**，否则两者一起免费（整串 16 分交互
            # 就是这么被判反的）。
            return self.cost.jack * self.cost.jack_time * time_factor * taper

        speed = distance / (delta_t / 1000)
        # "抗"的空间权重：**越远越重**。
        #   d <= 死区(0.1 单位)  -> bullet，上面已经免了；
        #   0.1 -> 0.5 单位      -> 线性变重（0.1 最轻、0.5 最重）；
        #   >= 0.5 单位          -> 保持最重，再远交给 move 惩罚继续涨。
        near = dead
        far = max(width_px * self.cost.jack_decay, near + 1e-9)
        # 小间隔额外加重（大间隔不变）：乘一个衰减很快的加成项。空间衰减照旧
        # 由下面的 spatial 负责，这里只按**时间**加重。
        # 只作用在**同位置死区**内：那正是"16 分同键连打 / 均匀纵连"。全区间加成
        # 在 dt=15ms 处就把抗放大到 2.6 倍，实测会把整份谱打崩（strife 12 -> 220）。
        if self.cost.jack_boost > 0 and distance <= near:
            time_factor *= 1.0 + self.cost.jack_boost * time_factor ** self.cost.jack_boost_power
        if _DENSE_TIME_CONVEX and distance <= near:
            time_factor = time_factor * time_factor
        if self.cost.jack_uphill:
            if distance <= near:
                # 死区内的**非 bullet**对 = 密集段里的同键连按。它和 bullet 是
                # 同一个动作，但"相对孤立"那条口径不成立 —— 用户口径是"这就是硬抗，
                # 必须罚"，而且是硬抗里最难受的一种（16 分同键连按要拆成两手交互）。
                # bullet 已经在上面按 jack_time 轻罚过了，两者由此分开。
                spatial = 1.0
            else:
                spatial = min(1.0, max(0.0, (distance - near) / (far - near)))
        else:                              # 旧行为：越近越像抗（到窗末归零）
            spatial = max(0.0, 1.0 - (distance - near) / (far - near))
        too_fast = max(0.0, (speed - self.physics.comfy_speed) / self.physics.comfy_speed)
        strength = spatial + too_fast * self.cost.jack_hard_boost + (1 - spatial) * self.cost.jack_time
        return (self.cost.jack * strength * time_factor * taper
                * (width / self.default_width))

    def hardness_cost(self, first: int, second: int, hand: str) -> float:
        """同手连打的"手感硬度"：只看时间间隔（不看位置）。

        调用方只在"这只手刚按过"时才问它，所以这里不用再判手。

        **bullet（子弹）**：两个键落在**同一个位置**时是"原地起落"（手指弹一下），
        比跑到别处再落一次轻松得多；这里只看时间间隔、**不做** bullet 折扣
        （实测打折会让密集段变差，见 README）。
        """

        if self.cost.hardness <= 0:
            return 0.0
        delta_t = self.notes[second].time - self.notes[first].time
        if delta_t <= 0 or delta_t > self.cost.hardness_window:
            return 0.0
        width_ratio = min(self.widths[first], self.widths[second]) / self.default_width
        # 和抗同一套收尾：<=100ms 原样，100~120 连续收到 0；
        # 判定为 bullet 的一对再打一个折扣（用户口径："hardness 也收一点"）。
        factor = _taper_factor(delta_t, _TAPER_HARD_END)
        if self.is_bullet(first, second):
            factor *= self.cost.bullet_hardness
        return (self.cost.hardness * (1 - delta_t / self.cost.hardness_window)
                * width_ratio * factor)

    def group_switch_cost(self, previous_hand: str | None,
                          hands_in_group: set[str], dt: float | None = None) -> float:
        """一整个**同时刻组**的换手代价（与组内顺序无关）。

        同一时刻的几个键是**同时**按下去的：两只手都已经落在键上，谈不上
        "从这只手换到那只手"。所以：

        - **组内不记换手**；
        - 进入这一组时，只要上一组收尾的那只手这一组也在用，就算延续，不记；
        - 离开这一组时若两只手都用过，则下一组用哪只都不算换手
          （状态记成 ``None``，见 :func:`_group_exit`）。

        早期是按扫描顺序**逐键**记换手的，而组内顺序又是按 x 排的 —— 等于让
        "哪个键在最左边"决定谁算入口/出口，凭空造出一笔与位置无关的偏好。
        实谱 23720 处就是这样：空间项本来更想让 arctap(x=0.00) 走左手
        （差 0.038），却被这笔 0.3 的假换手压过去，判成了反的。
        """
        if self.cost.switch <= 0 or not hands_in_group:
            return 0.0
        if previous_hand is None or previous_hand in hands_in_group:
            return 0.0
        if (_STREAM_ALT_BONUS > 0 and dt is not None
                and 0 < dt <= _STREAM_ALT_MS and len(hands_in_group) == 1):
            # 快速串：换手是"两手交替"的技巧，给一笔负代价（净效果是奖励交替）
            return self.cost.switch * self.switch_decay(dt) - _STREAM_ALT_BONUS
        return self.cost.switch * self.switch_decay(dt)

    def switch_decay(self, dt: float | None) -> float:
        """换手代价随"与上一组的间隔"衰减。

        手有充裕时间跑位时，"同一只手还是换手"根本不重要（真实代价由 move 表达），
        所以间隔超过 ``cost.switch_window`` 就不算换手。0 = 不衰减。
        """
        window = self.cost.switch_window
        if window <= 0 or dt is None:
            return 1.0
        if dt >= window:
            return 0.0
        return 1.0 - dt / window

    def bias_cost(self, index: int, hand: str) -> float:
        """空间偏好：自由键被分到"不该给"的那只手时的代价。

        两条证据叠加：

        - **动态**（``weight``）：按偏离局部中心多少增长，有 ``neutral_band`` 阈值；
        - **静态**（``home``）：这个键在左半屏还是右半屏，线性、无阈值。

        注意 ``neutral_band`` / ``strong_bias`` 这两个参数是按**屏幕像素**标定的
        （它们描述"离中心多近算中性"），而谱面坐标的刻度可能不是像素
        （Arcaea 是 1072 px/单位）。所以这里要按 ``pixel_per_unit_x`` 把它们换算到
        当前坐标刻度 —— 否则"15 像素的中性带"会变成"15 个世界单位"，
        比整个判定面还宽两倍，空间规则会被整段关掉，左右就靠平局乱分了。
        """
        hint = self.tail_hint[index]
        bonus = -self.cost.move * _TAIL_HINT_BONUS if hint is not None and hand == hint else 0.0
        if self.is_locked(index) or self.free_count == 0 or self.space.weight == 0:
            return bonus
        note = self.notes[index]
        scale_x = self.scale_x_at(index)
        neutral_band = self.space.neutral_band / scale_x     # 像素 -> 当前坐标单位
        strong_bias = max(self.space.strong_bias / scale_x, 1e-9)
        base = self.cost.move * (self.widths[index] / self.default_width)
        cost = 0.0

        if self.space.weight > 0:
            bias = self.bias[index]
            if hand != ("L" if bias <= 0 else "R"):
                over = abs(bias) - neutral_band
                if over > 0:
                    strength = min(1.0, over / strong_bias)
                    cost += base * self.space.weight * strength * 1.5

        if (self.space.lane_reverse > 0 and self.lane_side[index] is not None
                and hand != self.lane_side[index]):
            scale = 1.0
            if _STATIC_DURATION_SCALE and self.hold_duration[index] > 0:
                scale = 1.0 + self.hold_duration[index] / 200.0
            cost += base * self.space.lane_reverse * scale
        if self.space.reach > 0 and self._step_gap[index] >= _REACH_MS:
            offset = note.x - self.space.center_x
            wrong_side = max(0.0, offset) if hand == "L" else max(0.0, -offset)
            if wrong_side > 0:
                cost += base * self.space.reach
        if self.space.home > 0:
            # 中线由 space.center_x 给出：Arcaea 是 0.5，不是 0 —— 拿 0 当中心的话，
            # 整个 4 条轨道（-0.25 ~ 1.25）会被判成"全在右边"。
            offset = note.x - self.space.center_x
            wrong = max(0.0, offset) if hand == "L" else max(0.0, -offset)
            if wrong > 0:
                scale = 1.0
                if _STATIC_DURATION_SCALE and self.hold_duration[index] > 0:
                    scale = 1.0 + self.hold_duration[index] / 200.0
                cost += base * self.space.home * 3.0 * (wrong / (640.0 / scale_x)) * scale

        return cost + bonus

    def balance_cost(self, counts: dict[str, int]) -> float:
        """左右手使用量不平衡的代价。

        既要"别偏科"，又不能"为了凑平衡把该抗的一串键硬拆开"，
        所以是**相对偏差的 1.5 次方**：差 1~2 个键几乎不罚，差一半以上才明显罚。
        """
        total = counts["L"] + counts["R"]
        if self.free_count == 0 or total == 0:
            return 0.0
        deviation = (counts["L"] - counts["R"]) / total
        return self.cost.balance * abs(deviation) ** 1.5

    # ---- 整份拆解的代价 ---- #

    def _dense_hands(self, hands: Sequence[str | None]) -> list[str | None]:
        """把按 index 给出的手标记铺成"按扫描顺序"的数组（局部优化要顺序访问）。"""
        dense: list[str | None] = [None] * self.n
        for order, index in enumerate(self._order):
            dense[order] = hands[index]
        return dense

    def remember(self, hands: Sequence[str | None]) -> None:
        """记下当前手标记，供下一次 :func:`delta` 算"旧段之和"。"""
        self._last_hands = list(hands)

    def evaluate(self, hands: Sequence[str | None]) -> ScoreParts:
        """对一份完整拆解打分（和束搜索同一个代价函数，局部优化才好比较）。

        实现：把整份谱面当成"一个从 0 到 n-1 的段"交给 :func:`_segment_sum`，
        这样 :func:`delta` 与全量评分**共用同一套逐物件代价**，不会出现
        "两处各算一遍、结果对不上"的问题。
        """
        dense = self._dense_hands(hands)
        parts = _segment_sum(self, dense, 0, self.n - 1)
        counts = {"L": 0, "R": 0}
        for position, index in enumerate(self._order):
            hand = dense[position]
            if hand in ("L", "R") and not self.is_locked(index):
                counts[hand] += 1
        parts.balance = self.balance_cost(counts)
        return parts


#: 单次移动代价的上限（见 :meth:`Scoring.move_cost`）。
_MOVE_COST_MAX = 40.0

#: "长条/音弧区间内不许同手"这条判定的**额外容差**（ms）。
#: 谱面手写或生成时可能有浮点误差/失误，让本应对齐的时刻差上一两毫秒；
#: 判定区间前后各放宽这么多，并对"起点/尾指本应对齐"的键仍按同手处理。
#: 对正常谱面几乎没有影响。
SAME_HAND_MARGIN_MS = 2

#: 判定"同一个位置"的坐标容差（谱面坐标，1e-9 足够区分 0.25 的轨道间隔）。
_POS_EPS = 1e-9


def _same_pos(x1: float, y1: float, x2: float, y2: float) -> bool:
    """两个点是不是**同一个位置** —— xy 都要比，不能只看 x。"""
    return abs(x1 - x2) <= _POS_EPS and abs(y1 - y2) <= _POS_EPS


#: 弧颜色 -> 手（Arcaea：0 = 左手 / 1 = 右手）。放在这里是因为多指段要按
#: 弧的采样点做左右排序；`constraints.INT_TO_HAND` 是同一套映射。
_COLOR_HAND = {0: "L", 1: "R"}

#: 判定"是不是同一高度"（y 相差多少以内算同一层，多指段排序用）。
_LEVEL_EPS = 0.25

#: **音弧尾巴的延续偏好**（软）：音弧结束之后 ``_TAIL_HINT_MS`` 以内、落在弧尾
#: 同一个点上的键，交给弧那只手会便宜 ``cost.move × _TAIL_HINT_BONUS``。
#: 用户口径是"arc 的延续稍微长一点"—— 手指沿弧走到尾部刚松手，同位置的下一个键
#: 由它接着按最自然。用**软偏好**而不是硬钉：硬钉在别处会把该换手的键也钉住。
_TAIL_HINT_MS = 250.0
_TAIL_HINT_PX = 44.0
_TAIL_HINT_BONUS = 0.35

#: **快速串里的交替奖励**：与上一组的间隔不超过 ``_STREAM_ALT_MS`` 时，换手不再
#: 收费、反而给一笔负数（奖励严格交替）。依据：用户参考谱在密集串里写的是
#: **严格两手交替**（不看空间），而模型原来只在"同手连打更贵"时才会交替，
#: 相位/是否插入同手对就成了平局噪声。0 = 关闭。
_STREAM_ALT_MS = 130.0
_STREAM_ALT_BONUS = 0.3

#: 长条与实体弧的冲突是否按**区间重叠**判（True）还是只看长条起点（False）。
#: 用户实测 halcyon 127535：左手按着 0.25 的长条、又去跟一条 0 色弧（要 3 指），
#: 应当筛掉；但实测 aegleseeker 参考里这种重叠大量存在（硬筛会让它 14 -> 28），
#: 所以留成开关，取值以两侧参考的总体表现决定。
HOLD_ARC_OVERLAP = True



#: "出张扣分"（``space.reach``）的间隔阈值（ms）：**间隔够大**才收这笔账。
#: 用户口径就是这个 100ms：时间足够时手来得及换到正确的一侧，出张没有理由；
#: 间隔很短时不收（快速交互里两手本来就要来回跨）。
_REACH_MS = 100.0

#: 时间类惩罚的**连续收尾**（ms）：100ms 以内原样，100~120 之间梯度更大、
#: 线性收到 0。用户口径："不要直接截断到 0（不连续），100 以上梯度更大、
#: 到 120 收到 0；jack 与 hardness 都按这个思路"。
_TAPER_START_MS = 100.0
#: 收尾终点（ms）。等于/小于 ``_TAPER_START_MS`` 表示这一项**不设收尾**。
#: jack 与 hardness 分开：实测共同收尾会让 infinitestrife 12 -> 20（100~120ms
#: 那一档的硬度是它需要的），而 jack 单独收尾是净收益。
_TAPER_JACK_END = 120.0
#: hardness 的收尾终点要更缓：也收到 120 会让 infinitestrife 12 -> 20（100~120ms
#: 那一档的硬度是它需要的），收到 180 则既保住 strife 12、又把 aegleseeker 从
#: 26 降到 20（四张谱合计 43 -> 37，无回归）。整谱实测（非切片）。
_TAPER_HARD_END = 180.0


def _taper_factor(delta_t: float, end: float) -> float:
    """间隔接近窗口末端时的**连续收尾系数**（1 -> 0，无跳变）。

    这样 <=100ms 的行为一字不改（78/92-93ms 那几档不受影响），
    ``start``~``end`` 之间平滑归零（128ms 那种"为躲抗而反手出张"的平局被解开）。
    """
    if end <= _TAPER_START_MS:
        return 1.0
    if delta_t <= _TAPER_START_MS:
        return 1.0
    if delta_t >= end:
        return 0.0
    return (end - delta_t) / (end - _TAPER_START_MS)

#: **动态反手分两种**（用户口径）：两只手都在移动（另一只手在 ``_CROSS_MUTUAL_MS``
#: 之内刚落过键，说明它也在参与这一段）比"一只手动、另一只手短时间没动"更难受。
#: 用两个倍率分开：前者 ``_CROSS_MUTUAL`` 倍、后者 ``_CROSS_STILL`` 倍。
_CROSS_MUTUAL_MS = 250.0
_CROSS_MUTUAL = 1.0
_CROSS_STILL = 1.0

#: "同刻同 y、相差**整数个轨道**"的一对键（见 ``SpaceConfig.lane_reverse``）。
#: 用户口径：这种静态反手要狠狠罚，而且**不限于相邻的那一个轨道** ——
#: 1、2、3 个轨道（0.5 / 1.0 / 1.5）的间距都算。差值不是整数轨道
#: （例如弧线经过的任意位置）时不适用，所以判据仍然收得很紧。
_LANE_PAIR_SPACINGS = (0.5, 1.0, 1.5)
_LANE_PAIR_TOL = 0.05
_LANE_Y_EPS = 1e-3


def _is_lane_pair(dx: float) -> bool:
    """``|Δx|`` 是不是"1~3 个整数轨道"（见 ``_LANE_PAIR_SPACINGS``）。"""
    return any(abs(dx - spacing) <= _LANE_PAIR_TOL for spacing in _LANE_PAIR_SPACINGS)

#: 同位置连按（死区内、非 bullet）的**时间因子加速**（平方）：间隔越短，
#: "抬指再落"的难度是加速上升的 —— 46ms 的同键连按远比两次 93ms 交替难，
#: 线性因子下这两者几乎等价（实测 AD 的均匀纵连因此被拆出同手对）。
_DENSE_TIME_CONVEX = True


#: 静态空间罚按**占手时长**放大：长条按住的这段时间里，手一直被锁在错的一侧，
#: 比"单个键偏侧"严重得多。``1 + hold_duration/200``（400ms 长条 = 3 倍）。
_STATIC_DURATION_SCALE = True


def _spot(x: float, y: float) -> tuple[float, float]:
    """把一个落点化成可去重的键（多指段数"需要几根手指"用）。"""
    return (round(x, 3), round(y, 3))


def scale_at(physics, time_ms: float) -> tuple[float, float]:
    """``time_ms`` 时刻的**实际像素刻度** ``(px_per_unit_x, px_per_unit_y)``。

    六轨模式（``enwidenlanes`` / ``enwidencamera``）维持期间，同一块屏幕要装下
    更宽的弧坐标范围，所以刻度按倍率缩小（x 乘 2/3、y 乘 1/1.61，见
    :class:`spliter.aff.ScaleSchedule`）。**所有"多少个坐标单位 = 多少像素"
    的换算都必须过这里**，否则六轨段落的位移会被高估 1.5 倍。
    """
    base_x = getattr(physics, "pixel_per_unit_x", 1.0) or 1.0
    base_y = getattr(physics, "pixel_per_unit_y", 1.0) or 1.0
    schedule = getattr(physics, "scale_schedule", None)
    if schedule:
        scale_x, scale_y = schedule.at(time_ms)
        return base_x * scale_x, base_y * scale_y
    return base_x, base_y


def screen_span(dx: float, dy: float, physics, time_ms: float | None = None) -> float:
    """一段位移在**屏幕像素**上的实际长度：|Δx|、|Δy| 各自换算后勾股合成。

    **任何情况下都不能只算一个轴** —— 判定面是斜视的 3D 平面，Arcaea 里
    x 是 536 px/单位而 y 只有 326 px/单位；只看 x 会把"同一横坐标、从天到地"
    的位移当成 0。

    ``time_ms`` 给定时按六轨刻度换算；不给则用标准四轨刻度
    （向后兼容旧的调用与测试）。
    """
    if time_ms is None:
        unit_x = getattr(physics, "pixel_per_unit_x", 1.0)
        unit_y = getattr(physics, "pixel_per_unit_y", 1.0)
    else:
        unit_x, unit_y = scale_at(physics, time_ms)
    return math.hypot(dx * unit_x, dy * unit_y)


def screen_distance(first: Note, second: Note, physics) -> float:
    """两个物件在**屏幕像素**上的实际距离（按较晚那个物件的时刻取刻度）。"""
    return screen_span(second.x - first.x, second.y - first.y, physics,
                       max(first.time, second.time))


def _constraint_allows(constraints: object, index: int, hand: str) -> bool:
    """问外部约束"这个物件能不能用这只手"（容错：约束没有该方法时视为放行）。"""
    allowed = getattr(constraints, "allowed_hands", None)
    if allowed is None:
        return True
    return hand in allowed(index)


def _hold_duration(note: Note, physics) -> float:
    """这个长条是否真的"占手"，返回占手时长（0 = 不占手）。

    AFF 的长条**总是**带结束时间，不需要"按到下一个键的比例"去估长度
    （按物件长度估算的那条路径已经删掉）。
    """
    if str(note.type).lower() != "hold":
        return 0.0
    duration = note.duration
    if duration is None:
        return 0.0
    return float(duration) if duration >= physics.hold_min_duration else 0.0


def sequence_note(scoring: Scoring, index: int, hand: str, hands: Sequence[str | None]) -> int:
    """找 ``hand`` 这只手在按 ``index`` 之前按下的最后一个键（同组内更早的优先）。"""
    notes = scoring.notes
    time = notes[index].time
    for previous in range(index - 1, -1, -1):
        if notes[previous].time < time:
            for earlier in range(previous, -1, -1):
                if hands[earlier] == hand:
                    return earlier
            return -1
        if hands[previous] == hand:
            return previous
    return -1


def delta(scoring: "Scoring", hands: list[str | None], changed: Sequence[int],
          current: "ScoreParts",
          base_dense: list[str | None] | None = None,
          old_prefix: "PrefixCache | None" = None) -> "ScoreParts":
    """改动 ``changed`` 之后的总分。

    做法刻意保持简单、好验证：**总分 = 所有物件的局部代价之和**，
    所以只要重扫"受影响的那一段"，用"新段之和 − 旧段之和"修正总分即可。
    段的左右边界取"足够大"（多算几个物件不影响正确性，只影响一点速度），
    正确性由 :func:`_segment_sum` 保证 —— 它按扫描顺序重建，
    每个位置的分项与 :meth:`Scoring.evaluate` 完全一致。
    """
    # 新状态：从 base_dense 复制后只改动 changed 那几个位置（O(n) 的 C 级拷贝 + O(变化数)）
    if base_dense is not None:
        dense_new = list(base_dense)
        order_of_local = scoring._order_of
        for index in changed:
            dense_new[order_of_local[index]] = hands[index]
    else:
        dense_new = scoring._dense_hands(hands)
    dense_old = scoring._dense_hands(list(scoring._last_hands)) if scoring._last_hands else None
    order_of = scoring._order_of
    n = scoring.n
    if n == 0:
        return ScoreParts()

    lo, hi = n, -1
    for index in changed:
        position = order_of[index]
        lo = min(lo, position)
        hi = max(hi, position)
        for member in scoring.group_of(index):
            position_m = order_of[member]
            lo = min(lo, position_m)
            hi = max(hi, position_m)
        # 这只手在这之前的那个键：它本身不变，但"以它为 previous"的键变了，
        # 所以它的代价也要重算 —— 别忘了它。
        hand = dense_new[position]
        if hand in ("L", "R"):
            for earlier in range(position - 1, -1, -1):
                if dense_new[earlier] == hand:
                    lo = min(lo, earlier)
                    break
        # 同组成员（同刻的另一个键）之间的先后关系也可能变
        for member in scoring.group_of(index):
            position_m = order_of[member]
            for earlier in range(position_m - 1, -1, -1):
                if dense_new[earlier] == dense_new[position_m]:
                    lo = min(lo, earlier)
                    break
    if hi < 0:
        return current
    lo = max(0, lo)
    # 往右扩到"两只手都各自又出现过"之后的一小段。
    # **必须从 hi + 1 起算**：被改的那个键自己也算"这只手出现了一次"，
    # 从 hi 起算会让"两手都又出现"提前一条满足，段尾漏掉"同一只手的下一个键"
    # —— 而那条键的 move 正是以被改的键为 previous，实测能差出 39.9（封顶罚）。
    seen: set[str] = set()
    hi2 = hi
    for position in range(hi + 1, n):
        hand = dense_new[position] if dense_new[position] not in (None,) else None
        if hand and hand != "M":
            seen.add(hand)
        hi2 = position
        if len(seen) >= 2 and position > hi + 1:
            break
    hi = min(n - 1, hi2)

    old_start = old_prefix.state_at(lo) if old_prefix is not None else None
    old_sum = _segment_sum(scoring, dense_old, lo, hi, old_start) if dense_old else None
    new_parts = _segment_sum(scoring, dense_new, lo, hi)
    if old_sum is None:
        return scoring.evaluate(hands)
    parts = ScoreParts(
        move=current.move + new_parts.move - old_sum.move,
        jack=current.jack + new_parts.jack - old_sum.jack,
        bias=current.bias + new_parts.bias - old_sum.bias,
        switch=current.switch + new_parts.switch - old_sum.switch,
        cross=current.cross + new_parts.cross - old_sum.cross,
        busy=current.busy + new_parts.busy - old_sum.busy,
        balance=current.balance,
    )
    # 均衡项只依赖两只手的数量，整体重算（O(n) 但很便宜）
    counts = {"L": 0, "R": 0}
    for position, index in enumerate(scoring._order):
        hand = dense_new[position]
        if hand in ("L", "R") and not scoring.is_locked(index):
            counts[hand] += 1
    parts.balance = scoring.balance_cost(counts)
    return parts


class _HandView:
    """把"按物件索引取手"的回调包成 :meth:`Scoring.hold_tail_hand` 能用的序列。

    手标记有时按"扫描顺序"存（`_segment_sum` 的 dense）、有时按物件索引存
    （束搜索的状态），用一个视图把两种都接进来，避免为了取一只手而整表重建。
    """

    __slots__ = ("_scoring", "_hand_of")

    def __init__(self, scoring: "Scoring", hand_of) -> None:
        self._scoring = scoring
        self._hand_of = hand_of

    def __getitem__(self, order_position: int) -> str | None:
        return self._hand_of(self._scoring._order[order_position])

    def __len__(self) -> int:
        return self._scoring.n


def _group_bounds(scoring: "Scoring", position: int) -> tuple[int, int]:
    """``position`` 所在的同时刻组在**扫描顺序**里的起止位置（闭区间）。"""
    time = scoring.notes[scoring._order[position]].time
    start = position
    while start > 0 and scoring.notes[scoring._order[start - 1]].time == time:
        start -= 1
    end = position
    last = scoring.n - 1
    while end < last and scoring.notes[scoring._order[end + 1]].time == time:
        end += 1
    return start, end


def _iter_groups(scoring: "Scoring", lo: int, hi: int):
    """把扫描顺序的 ``[lo, hi]`` 按同时刻切成整组（要求 ``lo`` 组对齐）。"""
    position = lo
    while position <= hi:
        start, end = _group_bounds(scoring, position)
        yield start, end
        position = end + 1


def _group_exit(previous_hand: str | None, hands_in_group: set[str],
                has_m: bool) -> str | None:
    """扫完一个同时刻组之后，"上一只手"应该记成什么。

    两只手都用过 -> 两只手都已就位，下一组用哪只都不算换手 -> ``None``；
    只用了一只 -> 就是那只；一个都没用 -> 沿用原状态。
    组里有 ``M``（居中、不参与决策）时按老规矩把状态清掉。
    """
    if has_m:
        return None
    if len(hands_in_group) >= 2:
        return None
    if len(hands_in_group) == 1:
        return next(iter(hands_in_group))
    return previous_hand


class PrefixCache:
    """惰性前缀状态缓存：``state_at(i)`` 给出"扫完前 i 个位置"的状态。

    某一轮局部优化里，所有候选都从同一份 ``current`` 出发，所以前缀状态只在
    **被改动的那些位置之前**才会不同。这个类按需计算并记住结果：
    第一次问某个位置要扫一遍，之后直接命中。相比"每个候选重建整张表"，
    实测把真实谱面（630 物件）的局部优化从 18 秒压回个位数。
    """

    def __init__(self, scoring: "Scoring", dense: list[str | None]) -> None:
        self.scoring = scoring
        self.dense = dense
        self._cache: dict[int, tuple[dict[str, int], str | None]] = {0: ({}, None)}
        self._filled = 0

    def state_at(self, position: int) -> tuple[dict[str, int], str | None]:
        # 换手按整组记，所以状态只能停在"组的边界"上；先对齐到组首
        if self.scoring.n == 0:
            return ({}, None)
        position = min(position, self.scoring.n - 1)
        position, _ = _group_bounds(self.scoring, position)
        if position in self._cache:
            return self._cache[position]
        # 从已算到的位置继续往前补（缓存里存的都是组首）
        start = max(key for key in self._cache if key <= position)
        last_by_hand, previous_hand = self._cache[start]
        last_by_hand = dict(last_by_hand)
        for g_start, g_end in _iter_groups(self.scoring, start, position - 1):
            hands_in_group: set[str] = set()
            has_m = False
            for offset in range(g_start, g_end + 1):
                hand = self.dense[offset]
                if hand == "M":
                    has_m = True
                elif hand in ("L", "R"):
                    hands_in_group.add(hand)
                    last_by_hand[hand] = self.scoring._order[offset]
            previous_hand = _group_exit(previous_hand, hands_in_group, has_m)
        self._cache[position] = (last_by_hand, previous_hand)
        return self._cache[position]


def _segment_sum(scoring: "Scoring", dense: list[str | None], lo: int, hi: int,
                 start_state: tuple[dict[str, int], str | None] | None = None) -> "ScoreParts":
    """重扫 [lo, hi] 这一段，返回这一段贡献的分项（不含均衡项）。

    ``start_state`` 是"扫到 lo 之前那一瞬间"的状态 ``(每只手最近的键, 上一个手)``。
    局部优化里成百上千个候选的前缀是同一个（都从 ``current`` 出发），
    所以这个状态可以在每轮**预计算一次**，不必每个候选重建 —— 实测这是
    真实谱面上最大的单笔开销。
    """
    parts = ScoreParts()
    if hi < lo:
        return parts
    # 换手按"整组"记（组内不记），所以段边界必须先扩到整组，
    # 否则从组中间起扫会把这一组的换手记成两半。
    lo, _ = _group_bounds(scoring, lo)
    _, hi = _group_bounds(scoring, hi)

    def group_hands(g_start: int, g_end: int) -> tuple[set[str], bool]:
        """这一组用了哪些手；以及组里有没有 ``M``（居中，不参与决策）。"""
        used: set[str] = set()
        has_m = False
        for position in range(g_start, g_end + 1):
            hand = dense[position]
            if hand == "M":
                has_m = True
            elif hand in ("L", "R"):
                used.add(hand)
        return used, has_m

    # "上一个（不同时刻的）有手物件"：反手加成要跟它比。**必须从段前接上**，
    # 否则 delta 重扫一段时会漏掉段首那一笔，与 evaluate 对不上。
    previous_time_index: int | None = None
    if start_state is not None:
        last_by_hand = dict(start_state[0])
        previous_hand = start_state[1]
        for position in range(lo - 1, -1, -1):
            if dense[position] in ("L", "R"):
                previous_time_index = scoring._order[position]
                break
    else:
        last_by_hand = {}
        previous_hand = None
        for g_start, g_end in _iter_groups(scoring, 0, lo - 1):
            used, has_m = group_hands(g_start, g_end)
            for position in range(g_start, g_end + 1):
                hand = dense[position]
                if hand in ("L", "R"):
                    last_by_hand[hand] = scoring._order[position]
                    previous_time_index = scoring._order[position]
            previous_hand = _group_exit(previous_hand, used, has_m)
    # 扫目标段
    for g_start, g_end in _iter_groups(scoring, lo, hi):
        used, has_m = group_hands(g_start, g_end)
        # 与上一组的间隔（组对齐扫描，所以往前一格就是上一组的最后一个物件）
        dt = None
        if g_start > 0:
            dt = scoring.notes[scoring._order[g_start]].time                 - scoring.notes[scoring._order[g_start - 1]].time
        parts.switch += scoring.group_switch_cost(previous_hand, used, dt)
        # **同刻组内不算"交叉"**：进组前给每只手的最近键拍快照。
        group_last = dict(last_by_hand)
        for position in range(g_start, g_end + 1):
            index = scoring._order[position]
            hand = dense[position]
            if hand is None or hand == "M":
                continue
            parts.bias += scoring.bias_cost(index, hand)
            parts.busy += scoring.hold_conflict_cost(
                index, hand, lambda i: dense[scoring._order_of[i]])
            if previous_time_index is not None:
                parts.cross += scoring.cross_cost(
                    previous_time_index, dense[scoring._order_of[previous_time_index]],
                    index, hand)
            parts.cross += scoring.hand_cross_cost(
                index, hand, lambda h: group_last.get(h))
            previous = scoring.relationship_previous(
                last_by_hand.get(hand), index, hand)
            if previous is not None:
                parts.move += scoring.move_cost(previous, index, hand)
                if scoring.may_jack(previous, index):
                    parts.jack += scoring.jack_cost(previous, index, hand)
                if not scoring.is_locked(index):
                    parts.jack += scoring.hardness_cost(previous, index, hand)
            last_by_hand[hand] = index
        previous_hand = _group_exit(previous_hand, used, has_m)
        # 记住本组最后一个有手的物件，供下一组的"反手"判定使用
        for position in range(g_end, g_start - 1, -1):
            if dense[position] in ("L", "R"):
                previous_time_index = scoring._order[position]
                break
    return parts


def evaluate(notes: Sequence[Note], config: Config, hands: Sequence[str | None],
             meta: dict | None = None) -> ScoreParts:
    """便捷入口：给一份谱面 + 标记，直接算代价。"""
    return Scoring(notes, config, meta).evaluate(hands)


# --------------------------------------------------------------------------- #
# 求解器
# --------------------------------------------------------------------------- #

@dataclass
class _State:
    """束搜索的一个状态：这次分配之后每只手"最近按过的键"和占手情况。"""

    cost: float
    key: str
    hands: list[str | None]
    last: dict[str, int | None]      # 每只手最近按过的物件序号
    hold_until: dict[str, int]       # 每只手的长条按到什么时候
    counts: dict[str, int]


def split(notes: Sequence[Note], config: Config, meta: dict | None = None,
          constraints: object | None = None) -> SplitResult:
    """拆一份谱面。

    :param constraints: 可选的外部硬约束（例如音弧颜色），见 :class:`Scoring`
    :returns: :class:`SplitResult`（``hands`` 按 index、``side_by_note`` 按物件）
    """
    notes = list(notes)
    scoring = Scoring(notes, config, meta, constraints)
    groups = build_groups(notes)
    warnings: list[str] = []

    if not notes:
        return SplitResult(hands=[], side_by_note={}, score=ScoreParts(),
                           warnings=warnings, groups=[], scoring=scoring)

    _check_lock_conflicts(scoring, groups, warnings)
    _check_chord_feasibility(scoring, groups, warnings)
    hands = _beam_search(scoring, groups, config, warnings)
    hands = _refine(scoring, hands, config)
    # 最后再强制一遍"长条尾指同手"：这条规则语义上必须成立，而它在求解过程里
    # 只是候选生成/可行性检查的一个条件，被平局或局部优化绕过的可能性不值得去赌。
    hands = _enforce_hold_tail(scoring, hands)
    # 硬约束自相矛盾的地方（极少数）必须**显式暴露**，不能悄悄给个手标记了事
    if scoring.unresolved:
        lines = [str(scoring.notes[index].line) for index in scoring.unresolved[:6]]
        more = "…" if len(scoring.unresolved) > len(lines) else ""
        warnings.append(
            f"第 {'/'.join(lines)}{more} 行：这些物件没有任何合法手标记"
            "（原生标记与音弧颜色/多指空间规则互相矛盾），已按「原生标记 → 多指钉手 "
            "→ 绝对左右」的顺序取一个，请人工复核"
        )
    score = scoring.evaluate(hands)

    side_by_note = {note: hands[index] for index, note in enumerate(notes)}
    return SplitResult(
        hands=hands,
        side_by_note=side_by_note,
        score=score,
        warnings=warnings,
        groups=groups,
        scoring=scoring,
    )


def _enforce_hold_tail(scoring: Scoring, hands: list[str | None]) -> list[str | None]:
    """强制"长条尾指处的键与长条同手"。

    长条结束时手指停在尾巴上，同一个位置的键交给同一只手才说得通。
    这里放在最后统一收口：规则是语义要求，不该由代价和平局去决定。
    """
    if not scoring._hold_spans:
        return hands
    result = list(hands)
    for start, end, hold_position, tail_x, tail_y in scoring._hold_spans:
        hold_index = scoring._order[hold_position]
        hand = result[hold_index]
        if hand not in HANDS:
            continue
        for other, note in enumerate(scoring.notes):
            if other == hold_index or other in scoring.multi_finger:
                continue                      # 多指段不参与尾指同手
            if abs(note.time - end) <= 5 and _same_pos(note.x, note.y, tail_x, tail_y):
                if not scoring.can_hand(other, hand):
                    continue          # 有原生锁定/外部约束时以那边为准
                result[other] = hand
    return result


def _check_lock_conflicts(scoring: Scoring, groups: list[list[int]], warnings: list[str]) -> None:
    """原生 side 冲突检测：同一时刻同一只手被要求按跨度很大的两个键。"""
    for group in groups:
        locked: dict[str, list[int]] = {"L": [], "R": []}
        for index in group:
            side = scoring.sides[index]
            if side:
                locked[side].append(index)
        for hand, indexes in locked.items():
            if len(indexes) < 2:
                continue
            span_px = abs(scoring.notes[indexes[-1]].x - scoring.notes[indexes[0]].x) \
                * scoring.scale_x_at(indexes[0])
            if span_px > scoring.physics.max_reach:
                first, second = scoring.notes[indexes[0]], scoring.notes[indexes[1]]
                warnings.append(
                    f"第 {first.line} / {second.line} 行：同一时刻有 {len(indexes)} 个键都标了 "
                    f"side={hand}，跨度约 {round(span_px)}px 超过单手可及范围 "
                    f"（{scoring.physics.max_reach}px），谱面本身可能有误"
                )


def _check_chord_feasibility(scoring: Scoring, groups: list[list[int]],
                             warnings: list[str]) -> None:
    """和弦可行性检查：同一时刻的键能不能真的分给两只手。

    一只手在同一时刻只能按一个键，所以

    - 同一时刻 2 个键：只要不是两个键都被锁成同一只手，就能一左一右；
    - 同一时刻 3 个及以上：**没有任何合法分配**，只能退化处理并警告。

    三押在"两根手指游玩"的设定下本来就不可能，所以这里把它报出来，
    而不是默默输出一串同色标记。

    **多指配置除外**：那里一只手可以有几根手指（按住的弧/长条 + 别的指点键），
    同刻多个键、甚至同一只手拿两个键都是常态，不是谱面错误。
    """
    for group in groups:
        if len(group) < 2:
            continue
        if any(index in scoring.multi_finger for index in group):
            continue                          # 多指配置：一只手可以有几根手指
        notes = [scoring.notes[index] for index in group]
        lines = "/".join(str(note.line) for note in notes)
        if len(group) > 2:
            warnings.append(
                f"第 {lines} 行：同一时刻有 {len(group)} 个键，但只有两只手 —— "
                "物理上无法全部按下（已尽量摊到两只手上，建议检查谱面）"
            )
            continue
        first, second = group
        both_left = scoring.can_hand(first, "L") and scoring.can_hand(second, "L")
        both_right = scoring.can_hand(first, "R") and scoring.can_hand(second, "R")
        if (both_left and not (scoring.can_hand(first, "R") or scoring.can_hand(second, "R"))) or \
           (both_right and not (scoring.can_hand(first, "L") or scoring.can_hand(second, "L"))):
            warnings.append(
                f"第 {lines} 行：同一时刻的两个键都被锁定给同一只手，"
                "物理上无法同时按下（已按原生标记保留）"
            )


def _beam_search(scoring: Scoring, groups: list[list[int]], config: Config,
                 warnings: list[str]) -> list[str | None]:
    """束搜索：按时间顺序推进，每步只保留 beam_width 个最优状态。"""
    beam_width = config.search.beam_width
    start_time = scoring.notes[0].time - scoring.physics.home_ready
    home = {"L": scoring.physics.home_l, "R": scoring.physics.home_r}

    beam: list[_State] = [_State(
        cost=0.0,
        key=f"|{home['L']}|{home['R']}||",
        hands=[None] * scoring.n,
        last={"L": None, "R": None},
        hold_until={"L": 0, "R": 0},
        counts={"L": 0, "R": 0},
    )]

    # 只有在"这个时刻组落在某条长条区间内"时，候选才依赖当前手标记；
    # 其余情况（绝大多数）候选是固定的，算一次就够。
    hold_spans = [
        (note.time, int(note.params.get("end", note.time + (note.params.get("duration") or 0))))
        for note in scoring.notes if note.type == "hold"
    ]

    for group in groups:
        group_time = scoring.notes[group[0]].time
        inside_hold = any(start < group_time < end for start, end in hold_spans)
        fixed_candidates = None if inside_hold else _realizations(scoring, group)
        # 同一个 key 只留最优状态 —— 这是让束宽可控的关键剪枝
        best: dict[str, _State] = {}
        for state in beam:
            candidates = fixed_candidates if fixed_candidates is not None \
                else _realizations(scoring, group, state.hands)
            for assignment in candidates:
                expanded = _expand(scoring, state, group, assignment, start_time, home)
                if expanded is None:
                    continue
                existing = best.get(expanded.key)
                if existing is None or expanded.cost < existing.cost:
                    best[expanded.key] = expanded
        # 平局时用"极小的均衡代价 + key"打破 —— 只写 key 的话，
        # 字典序会让"整段都给左手"这类解系统性胜出（像素刻度下大量转移代价相同，
        # 这个偏置会真的改变结果）。
        states = sorted(best.values(),
                        key=lambda item: (item.cost + _tie_break(scoring, item), item.key))

        if not states:
            # 兜底：这一组在所有状态下都无解。最常见的原因是**两条长条同时占着
            # 两只手**，而谱面又在它们的区间里放了天键 —— 那是要玩家动第三根手指，
            # 而"一键一手"的模型只有两只手。
            #
            # 关键：**绝不能把物件留成"没有手"**（回写出去就是个空标记）。
            # 所以这里放弃"长条占手"这一条硬约束，按最小代价挑一个分配。
            # 其余硬约束（原生标记 / 音弧颜色 / 一键一手）仍然照旧。
            relaxed = _realizations(scoring, group)
            states = []
            for state in beam:
                for assignment in relaxed:
                    expanded = _expand(scoring, state, group, assignment, start_time, home,
                                       ignore_hold_conflict=True)
                    if expanded is not None:
                        states.append(expanded)
            if states:
                states.sort(key=lambda item: (item.cost, item.key))
                warnings.append(
                    f"第 {scoring.notes[group[0]].line} 行附近：本组没有合法分配"
                    "（多半是两条长条同时占着两只手，谱面实际要求第三根手指），"
                    "已忽略「长条占手」按最小代价分配"
                )
            else:
                states = beam
                warnings.append(
                    f"第 {scoring.notes[group[0]].line} 行附近：本组物件无法分配左右手"
                    "（原生标记可能自相矛盾），已沿用上一状态"
                )
        beam = states[:beam_width]

    beam.sort(key=lambda item: (item.cost + scoring.balance_cost(item.counts), item.key))
    return list(beam[0].hands)


def _tie_break(scoring: Scoring, state: "_State") -> float:
    """纯平局打破用的一点点均衡倾向（量级远小于真实代价差）。"""
    return scoring.balance_cost(state.counts) * 1e-3


def _realizations(scoring: Scoring, group: list[int],
                  hands: Sequence[str | None] | None = None) -> list[list[str]]:
    """同一时刻的物件 -> 所有合法的"一键一手"分配。

    **硬约束**：一只手在同一时刻只能按一个键。因此

    - 双押（同一时刻两个键）必须一左一右 —— 同手分配根本不生成；
    - 三押及以上**没有任何合法分配**（两只手按不下三个同时的键）。

    这一点必须做成硬约束而不是扣分：和弦同色对玩家是"物理上做不到"，
    靠加大罚分永远避免不干净（罚得再重，总会在某个上下文里显得"划算"）。
    真的遇到三押时，程序会退化处理并在 :attr:`SplitResult.warnings` 里说明。
    """
    results: list[list[str]] = []
    current: list[str | None] = [None] * len(group)

    def walk(slot: int, used: frozenset[str]) -> None:
        if slot == len(group):
            # 多指时刻：**两手触点数差不得超过 1**（用户口径的硬限制）。
            # 计数 = 本组判定物件的手 + 该时刻每只手的弧触点数。
            arcs_here = scoring.moment_arcs.get(scoring.notes[group[0]].time)
            if arcs_here:
                counts = dict(arcs_here)
                for hand in current:
                    if hand in counts:
                        counts[hand] += 1
                if abs(counts.get("L", 0) - counts.get("R", 0)) > 1:
                    return
            results.append([hand for hand in current])  # type: ignore[misc]
            return
        index = group[slot]
        for hand in HANDS:
            if hand in used or not scoring.can_hand(index, hand):
                continue
            # 同组内的"长条 + 尾指键"必须同手（手指停在尾巴上）
            if any(current[earlier] is not None
                   and scoring.is_tail_pair(group[earlier], index)
                   and current[earlier] != hand
                   for earlier in range(slot)):
                continue
            if hands is not None:
                if scoring.hold_conflict(index, hand, hands):
                    continue
                required = scoring.hold_tail_hand(index, hands)
                if required is not None and hand != required:
                    continue
            current[slot] = hand
            walk(slot + 1, used | {hand})
        current[slot] = None

    # ---- 多指配置：一只手可以有几根手指 ---- #
    # "一键一手"在这里不成立（典型：一只手按住音弧/长条，同一刻另一个手指
    # 去点别的键 —— 参考谱面里这种配置的同手是**常态**）。手标记由空间钉死，
    # 所以候选就是"钉手优先、其余任选"的组合；长条头尾那些约束也一并放宽。
    # 钉手若与硬约束（原生标记 / 音弧颜色）冲突，就**听硬约束的**（用户口径里
    # "只看空间"是对前后关系的取舍，不是允许违反音弧颜色）。
    if any(index in scoring.multi_finger for index in group):
        options: list[list[str]] = []
        for index in group:
            want = scoring.sides[index] or scoring.multi_hand.get(index)
            if want and scoring.can_hand(index, want):
                options.append([want])
            else:
                options.append([hand for hand in HANDS if scoring.can_hand(index, hand)])
        if all(options):
            arcs_here = scoring.moment_arcs.get(scoring.notes[group[0]].time)
            reach = scoring.px_to_units(scoring.physics.max_reach)
            combos: list[list[str]] = []
            for combo in product(*options):
                # ① 两手触点数差 <= 1（用户口径的硬限制；含弧触点数）
                if arcs_here:
                    counts = dict(arcs_here)
                    for hand in combo:
                        if hand in counts:
                            counts[hand] += 1
                    if abs(counts.get("L", 0) - counts.get("R", 0)) > 1:
                        continue
                # ② 同一只手在同一刻覆盖的跨度不能超过可及范围
                ok = True
                for hand in HANDS:
                    pts = [(scoring.notes[group[slot]].x, scoring.notes[group[slot]].y)
                           for slot in range(len(group)) if combo[slot] == hand]
                    if len(pts) > 1:
                        span = max(screen_span(px - qx, py - qy, scoring.physics)
                                   for px, py in pts for qx, qy in pts)
                        if scoring.px_to_units(span) > reach:
                            ok = False
                            break
                if ok:
                    combos.append(list(combo))
            if combos:
                return combos

    walk(0, frozenset())

    if not results:
        # 两只手按不下（典型是同一时刻 3 个以上的键），或者锁定标记自相矛盾。
        # 退化处理：一定有某只手要兼顾两个键，那就**枚举所有分配、挑"最大的同手跨度"
        # 最小的那个**（等于让最勉强的那只手尽量轻松），并在上层给出警告。
        # 和弦的键数很少，枚举完全可行。
        reach = scoring.px_to_units(scoring.physics.max_reach)
        best: list[str] | None = None
        best_score: tuple[float, float] | None = None
        for mask in range(1 << len(group)):
            assignment = [HANDS[(mask >> slot) & 1] for slot in range(len(group))]
            if not all(scoring.can_hand(index, hand)
                       for index, hand in zip(group, assignment)):
                continue
            widest = 0.0
            for hand in HANDS:
                pts = [(scoring.notes[group[slot]].x, scoring.notes[group[slot]].y)
                       for slot in range(len(group)) if assignment[slot] == hand]
                if len(pts) > 1:
                    span = max(screen_span(px - qx, py - qy, scoring.physics)
                               for px, py in pts for qx, qy in pts)
                    widest = max(widest, scoring.px_to_units(span))
            # 先看"有没有满足可及范围的"，再看最大同手跨度；最后偏好两手都用上
            feasible = 0.0 if widest <= reach else 1.0
            spread = 0.0
            for hand in HANDS:
                xs = [scoring.notes[group[slot]].x
                      for slot in range(len(group)) if assignment[slot] == hand]
                ys = [scoring.notes[group[slot]].y
                      for slot in range(len(group)) if assignment[slot] == hand]
                if len(xs) > 1:
                    spread += scoring.px_to_units(max(
                        screen_span(px - qx, py - qy, scoring.physics)
                        for px, py in zip(xs, ys) for qx, qy in zip(xs, ys)))
            score = (feasible, round(widest, 6), round(spread, 6))
            if best_score is None or score < best_score:
                best, best_score = assignment, score
        if best is None:
            # 连"枚举所有分配"都找不到合法手标记：硬约束自己矛盾了。
            # 这里**绝不静默**用 slot 奇偶凑数（那会把非法手标记直接写进谱面）：
            # 优先原生标记 / 多指钉手，再退到"绝对左右"；真的一个合法手都没有
            # 就记进 ``scoring.unresolved``，上层据此给警告。
            best = []
            for index in group:
                spatial = "L" if scoring.notes[index].x < scoring.space.center_x else "R"
                for hand in (scoring.sides[index], scoring.multi_hand.get(index),
                             spatial, "L", "R"):
                    if hand and scoring.can_hand(index, hand):
                        best.append(hand)
                        break
                else:
                    best.append(scoring.sides[index]
                                or scoring.multi_hand.get(index) or spatial)
                    scoring.unresolved.append(index)
        results.append(best)
    return results


def _expand(scoring: Scoring, state: _State, group: list[int], assignment: list[str],
            start_time: int, home: dict[str, float],
            ignore_hold_conflict: bool = False) -> _State | None:
    """状态转移：加上本组的代价，推出新状态。

    ``ignore_hold_conflict`` 只给"整组无解"时的兜底用：谱面里两条长条同时占着
    两只手、而在它们区间内又来了天键时，按"一键一手"是无解的（实际要第三根手指）。
    这时宁可放弃"长条占手"这一条，也不能把物件留成没有手。
    """
    hands = list(state.hands)
    last = dict(state.last)
    hold_until = dict(state.hold_until)
    counts = dict(state.counts)
    cost = state.cost

    group_last = dict(last)      # 同刻组内不算"交叉"（与 _segment_sum 一致）
    for slot, index in enumerate(group):
        hand = assignment[slot]
        if not ignore_hold_conflict and scoring.hold_conflict(index, hand, hands):
            return None                       # 撞上正在按住的长条：这条路不成立
        if not ignore_hold_conflict:
            required = scoring.hold_tail_hand(index, hands)
            if required is not None and hand != required:
                return None                   # 长条尾指处必须同手
        # 换手按"整组"记：组内是同刻按下的，不算换手（见 group_switch_cost）。
        # 束搜索的代价只是排序用的前缀量，这里与 evaluate 保持一致即可。
        previous = scoring.relationship_previous(last[hand], index, hand)

        if previous is None:
            # 这只手第一次出手：从起手手位走过去（也要时间，但不算太严格）
            # 位移按 xy 勾股算屏幕距离 —— 天空物件有 y 分量，只看 x 会漏掉它。
            delta_t = max(scoring.notes[index].time - start_time, 1)
            distance = scoring.home_distance(index, hand, home)
            speed = distance / (delta_t / 1000)
            if speed > scoring.physics.max_speed * 3:
                cost += 20
            elif speed > scoring.physics.comfy_speed:
                ratio = speed / scoring.physics.comfy_speed - 1
                cost += scoring.cost.move * ratio * ratio * 0.5
        else:
            cost += scoring.move_cost(previous, index, hand)
            if scoring.may_jack(previous, index):
                cost += scoring.jack_cost(previous, index, hand)
            if not scoring.is_locked(index):
                cost += scoring.hardness_cost(previous, index, hand)

        cost += scoring.bias_cost(index, hand)
        cost += scoring.hold_conflict_cost(index, hand, lambda i: hands[i])
        cost += scoring.hand_cross_cost(index, hand, lambda h: group_last.get(h))
        if not scoring.is_locked(index):
            counts[hand] += 1
        last[hand] = index
        # 占手状态取"结束更晚"的那个（同组内同一只手只可能有一个键）
        if scoring.hold_duration[index] > 0:
            hold_until[hand] = max(hold_until[hand], scoring.hold_until[index])
        hands[index] = hand

    key = "|".join((
        str(last["L"]) if last["L"] is not None else "-",
        str(last["R"]) if last["R"] is not None else "-",
        str(hold_until["L"] or 0),
        str(hold_until["R"] or 0),
        str(counts["L"]),
        str(counts["R"]),
    ))
    return _State(cost=cost, key=key, hands=hands, last=last,
                  hold_until=hold_until, counts=counts)


# --------------------------------------------------------------------------- #
# 局部优化
# --------------------------------------------------------------------------- #

def _refine(scoring: Scoring, hands: list[str | None], config: Config) -> list[str | None]:
    """贪心局部搜索：单键翻转 / 邻近交换 / 前缀翻转 / 整段镜像。

    每一轮把邻域**全部**评一遍，取最好的那个接受（steepest descent）。
    曾经为了提速改成"发现更好就立刻接受"，结果在"每 100ms 挪 100px"这类
    平坦但需要整体协调的段落上会卡在次优解（实测比最优贵 15%）。
    现在这个粒度（每轮约 O(n) 个候选、每个候选 O(n) 打分）已经够快，
    换来的是"输出确实是这套代价下的局部最优"。
    """
    if config.search.refine_passes <= 0 or scoring.n == 0:
        return list(hands)

    current = list(hands)
    current_parts = scoring.evaluate(current)
    best = current_parts.total

    for _ in range(config.search.refine_passes):
        best_candidate: list[str | None] | None = None
        best_parts = current_parts
        best_score = best
        scoring.remember(current)
        # 这一轮的所有候选都从 current 出发：前缀状态缓存与 base_dense 可以共用
        base_dense = scoring._dense_hands(current)
        old_prefix = PrefixCache(scoring, base_dense)
        for candidate in _neighbours(scoring, current, config):
            changed = _changed_indexes(scoring, current, candidate)
            # 只重算受影响的几个键（见 delta 的说明），而不是整份谱面
            parts = delta(scoring, candidate, changed, current_parts,
                          base_dense=base_dense, old_prefix=old_prefix)
            if parts.total < best_score - 1e-9:
                best_score = parts.total
                best_candidate = candidate
                best_parts = parts
        if best_candidate is None:
            break
        current = best_candidate
        current_parts = best_parts
        best = best_score

    return current


def _changed_indexes(scoring: "Scoring", before: Sequence[str | None],
                     after: Sequence[str | None]) -> list[int]:
    """两次手标记之间变了的物件 index（局部优化每次只改少数几个）。"""
    return [index for index in scoring._order if before[index] != after[index]]


def _neighbours(scoring: Scoring, hands: list[str | None], config: Config) -> Iterator[list[str | None]]:
    """生成邻域候选（翻转 / 交换 / 前缀翻转 / 镜像）。

    每次都 ``yield`` 一个新列表，调用方拿去打分；改进就整体替换。
    写成生成器是为了让"尝试哪些改动"一眼能看清。
    """
    n = scoring.n

    # 1) 单键翻转
    for index in range(n):
        if scoring.is_locked(index):
            continue
        candidate = list(hands)
        candidate[index] = OTHER[candidate[index] or "L"]
        if _feasible(scoring, candidate, index):
            yield candidate

    # 2a) 窗口内同标记交换：在一个很短的窗口里把两个"相同标记"的位置对调。
    #     专门用来修"整段该换个起手"的等代价平局解：例如把 RLRLRLRLRL**RR**L
    #     调整成严格交替 —— 单键翻转会被同刻约束挡住，窗口重排则能过去。
    window = max(1, config.search.refine_radius)
    for start in range(n):
        stop = min(n, start + window + 2)
        for first in range(start, stop):
            if scoring.is_locked(first):
                continue
            for second in range(first + 1, stop):
                if scoring.is_locked(second) or hands[first] != hands[second]:
                    continue
                candidate = list(hands)
                candidate[first] = "R" if hands[first] == "L" else "L"
                candidate[second] = candidate[first]
                if _feasible_region(scoring, candidate, first, second + 1):
                    yield candidate

    # 2b) 邻近交换（把"抗"的边界往旁边挪一格）
    radius = config.search.refine_radius
    for index in range(n):
        if scoring.is_locked(index):
            continue
        for offset in range(1, radius + 1):
            other = index + offset
            if other >= n:
                break
            if scoring.is_locked(other) or hands[index] == hands[other]:
                continue
            if scoring.notes[other].time - scoring.notes[index].time > 900:
                break
            candidate = list(hands)
            candidate[index], candidate[other] = candidate[other], candidate[index]
            if _feasible(scoring, candidate, index) and _feasible(scoring, candidate, other):
                yield candidate

    # 3) 前缀翻转：翻转从某个组边界往后的所有标记，用来切换"这一段从哪只手起"
    for group in scoring.groups:
        start = group[0]
        if start == 0:
            continue
        if any(scoring.is_locked(index) for index in range(start, n)):
            continue
        candidate = list(hands)
        for index in range(start, n):
            candidate[index] = OTHER[candidate[index] or "L"]
        if _feasible_region(scoring, candidate, start, n):
            yield candidate

    # 4) 整段镜像（把一段交互整体左右互换，内部结构不动）
    #    长度上限可配（见 search.mirror_max）
    if config.search.refine_mirror:
        for start in range(n):
            for length in range(2, max(3, config.search.mirror_max + 1)):
                end = start + length - 1
                if end >= n:
                    break
                if scoring.notes[end].time - scoring.notes[start].time > 2600:
                    break
                if any(scoring.is_locked(index) for index in range(start, end + 1)):
                    continue
                candidate = list(hands)
                for index in range(start, end + 1):
                    candidate[index] = OTHER[candidate[index] or "L"]
                if _feasible_region(scoring, candidate, start, end + 1):
                    yield candidate


def _feasible(scoring: Scoring, hands: list[str | None], index: int) -> bool:
    """改动 ``index`` 之后，它**所在的那个时刻组**是否仍然满足硬约束。

    关键是"整组"而不是"这一个键"：把双押的左边那个键翻到和右边同一只手，
    只检查被翻的那一个键是查不出来的。
    """
    group = scoring.group_of(index)
    return _feasible_region(scoring, hands, group[0], group[-1] + 1)


def _hold_extended_stop(scoring: Scoring, start: int, stop: int) -> int:
    """把 ``[start, stop)`` 里**长条**的占手区间并进来，返回扩展后的 ``stop``。

    长条换手会牵动它**整段**：区间里的键必须换手，尾指同刻的键也算。局部优化
    只查"被翻物件自己那一刻的组"是不够的 —— 把长条翻到与区间**后面**某个键同手
    时查不出来（实测 ``cc_test.txt`` 109833 / 113166）。所以可行性检查要按扩展后
    的区间来做。

    只有**长条**需要扩展：翻一个非长条的键不会改变任何长条的占手关系，也就
    不会在区间外造出新违规。
    """
    end_slots = scoring.hold_end
    if not end_slots:
        return stop
    times = scoring._times
    order = scoring._order
    extended = stop
    slot = start
    while slot < extended:                 # 扩展段里可能还有长条，边扫边扩
        end = end_slots.get(order[slot])
        if end is not None:
            extended = max(extended, bisect_right(times, end + 5))
        slot += 1
    return min(extended, scoring.n)


def _feasible_region(scoring: Scoring, hands: list[str | None],
                     start: int, stop: int) -> bool:
    """``[start, stop)`` 这段（以及它们各自的同刻伙伴）是否满足硬约束。

    局部优化的"整段镜像 / 前缀翻转"会一次改很多键，所以必须按区域检查。
    因为同一时刻的键之间没有先后，检查某个区间时也要把它涉及到的
    **整个时刻组**都算进来，否则会漏掉"改了半组"造成同手和弦的情况。
    """
    if start < 0 or stop > scoring.n or start >= stop:
        return False
    # 区域内有长条时，把它的占手区间一起查（否则长条换手造成的区间外违规查不出）
    stop = _hold_extended_stop(scoring, start, stop)
    for index in range(start, stop):
        if not hands[index] or not scoring.can_hand(index, hands[index]):
            return False
    # 同一时刻：这只手只能出现一次（把区间涉及的整组都算上）
    for group in scoring.groups:
        if group[-1] < start or group[0] >= stop:
            continue
        if any(index in scoring.multi_finger for index in group):
            # 多指配置：一只手可以有几根手指，因此**允许同手**；但手标记必须
            # 与空间钉手（或原生标记）一致 —— 局部优化不许改这些键。
            # 这里要覆盖到整组，因为区间可能只切到了组的一半。
            for index in group:
                hand = hands[index]
                if not hand or not scoring.can_hand(index, hand):
                    return False
                want = scoring.sides[index] or scoring.multi_hand.get(index)
                if want and scoring.can_hand(index, want) and hand != want:
                    return False
            continue
        if scoring.can_fill_group(group):
            # 有合法分配：那就必须一键一手，并且不能撞上正在按住的长条
            seen: set[str] = set()
            for index in group:
                hand = hands[index]
                if not hand:
                    return False
                if hand in seen:
                    return False
                if scoring.hold_conflict(index, hand, hands):
                    return False
                required = scoring.hold_tail_hand(index, hands)
                if required is not None and hand != required:
                    return False
                for other in group:
                    if other != index and scoring.is_tail_pair(other, index) \
                            and hands[other] != hand:
                        return False
                seen.add(hand)
            continue
        # 三押这类"两只手按不下"的组：没有合法解，但仍然要求**尽量摊平** ——
        # 不能退化成"三个键全给一只手"，也不能出现空标记。
        # 允许的手数上限 = ceil(n/2)（两只手平分）。
        allowed = -(-len(group) // len(HANDS))
        used: dict[str, int] = {hand: 0 for hand in HANDS}
        for index in group:
            hand = hands[index]
            if hand not in HANDS:
                return False
            used[hand] += 1
        if used["L"] > allowed or used["R"] > allowed:
            return False
    return True


# --------------------------------------------------------------------------- #
# 便捷查询
# --------------------------------------------------------------------------- #

def hand_of(result: SplitResult, note: Note) -> str | None:
    """从结果里取某个物件的标记（按物件对象查，不受排序影响）。"""
    return result.side_by_note.get(note)


