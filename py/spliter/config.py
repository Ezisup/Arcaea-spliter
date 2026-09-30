"""拆谱参数：**Arcaea 专用，已固化**；对外只留 ``jack`` / ``move`` 两个可调项。

固定值
------
``PhysicsConfig`` / ``SpaceConfig`` / ``SearchConfig`` 的默认值就是 Arcaea 的实测
取值，``CostConfig`` 里除 ``jack`` / ``move`` 之外也全部固化（标定理由见
``spliter/README.md`` 的参数表）。它们**不再暴露成命令行参数** —— 换音游要改的
是整套模型，不是调几个数。

可调项
------
- ``cost.jack``：抗（同手连打）的权重。调大 -> 更愿意拆交互、更不愿同手连打；
- ``cost.move``：移动/速度的权重。调大 -> 更"偷懒"、更倾向就近按键。

另外保留 ``output.*``（输出文件写法与 info 报告格式），与拆谱本身无关。

已随机制一起删掉的旧参数
------------------------
为早期通用格式服务的那套参数；位移衰减 ``move_decay_*``；
move 免费区 ``move_soft_ratio``；bullet 硬度折扣 ``bullet_factor``；基础代价
``base``；原生标记的软拉动 ``lock_influence`` / ``lock_pull``；``M`` 标记输出
``allow_m`` / ``neutral_action``；宽松弧释放 ``arc_release_loose``；长条长度估算
``hold_ratio``。这些在 Arcaea 单一分支下都不起作用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Config",
    "PhysicsConfig",
    "CostConfig",
    "SpaceConfig",
    "SearchConfig",
    "OutputConfig",
    "resolve_width_expr",
    "validate",
]


@dataclass
class PhysicsConfig:
    """物理模型（Arcaea 实测标定，固化）。"""

    comfy_speed: float = 3120.0
    """单手舒适移速（px/s）：超出后按平方曲线涨价（标定见 README）。"""

    max_speed: float = 9360.0
    """单手极限移速（px/s）：超过再追加峰值罚。"""

    home_l: float = 0.35075
    """起手手位：开场时左手待的位置（中线 - 80px）。"""

    home_r: float = 0.64925
    """起手手位：右手（中线 + 80px）。"""

    home_ready: float = 350.0
    """起手准备时间（ms）：第一个物件之前算手已就位。"""

    pixel_per_unit_x: float = 536.0
    """横坐标 1 个单位 = 多少像素（游戏内截图标定）。"""

    pixel_per_unit_y: float = 326.0
    """纵坐标 1 个单位 = 多少像素（两轴刻度不同，位移必须分轴换算）。"""

    scale_schedule: Any = None
    """分段像素刻度（六轨模式）；``None`` 或空 = 全程标准四轨。

    由 :func:`spliter.aff.apply_arcaea_physics` 按谱面里的
    ``enwidencamera`` / ``enwidencamera`` 事件填好，见
    :class:`spliter.aff.ScaleSchedule`。**不要在代价函数里直接读
    ``pixel_per_unit_*``** —— 六轨段落要乘这里的倍率。
    """

    max_reach: float = 620.0
    """单手同一时刻能覆盖的最大横向跨度（px）：和弦可行性与跨度判据。"""

    hold_min_duration: float = 130.0
    """长条 duration 超过它才真的"占手"（太短的尾巴可以忽略）。

    只影响**权重侧**：低于它的长条 ``hold_duration`` 归零，不再收"抗/位移"的
    占手代价（硬规则"区间异手"始终用真实区间，不受这里影响，见 §4.6）。
    它**不是**"多短的长条可以忽略"的物理判据，纯粹是权重侧的取舍，
    所以取值只需要落在"明显按不住"的量级上。
    """

    hold_tail_relax: float = 50.0

    #: 音弧的"提前松手"实验开关（毫秒）。``< 0`` = 关闭（默认，弧不参与位移结算）；
    #: ``>= 0`` = 把弧尾当作位移/计时的释放点，并可前移这么多毫秒（与长条的
    #: ``hold_tail_relax`` 同一思路）。
    arc_release: float = -1.0
    """占手窗口在结尾**提前**多少 ms 结束（0 = 一直算到长条真结束）。

    这是"权重侧"的松动：硬规则（长条区间异手）始终用真实区间，这里只影响
    "这只手还被按着"的那笔代价。原值 100ms 会让长条在模型里提前"松手"，
    从而丢掉它对后续键的**引导**（用户口径：相位平局时应当优先看长条/音弧
    锁住的那只手）。实测三张参考谱在 0 处同时最好（44 处 vs 55 处）。"""

    multi_finger_min: int = 3

    #: 多指段的手标记策略。``True`` = 硬钉（历史口径：只看绝对空间 + 同 y 左右分半）；
    #: ``False`` = 只钉**原生 side 标记**，其余自由键交给代价模型（双指那套空间罚分）
    #: 决定，并用"两手触点数差 <= 1"作为硬约束兜底（用户口径）。
    multi_hand_pin: bool = False
    """多指段阈值：同一时刻需要的**触点**数达到它就走"只看空间"的分法。"""


@dataclass
class CostConfig:
    """代价权重。**除前两项外全部固化**（Arcaea 实测取值）。"""

    # ---- 唯二的可调项 ---- #
    jack: float = 6.0
    """★ 抗（同手连打）的权重：这是"拆 vs 抗"最主要的旋钮。"""

    move: float = 1.0
    """★ 移动/速度的权重：调大 -> 手更"偷懒"，更倾向就近按键。"""

    # ---- 以下固化 ---- #
    jack_window: float = 150.0
    """抗的判定窗口（ms）：只对间隔不超过它的连续同手按键罚分。

    试过收到 **120**（用户建议）：128ms 那一档不再收抗，aegleseeker 84358/84486
    那两处"为躲抗而反手出张"确实解开了（26 -> 24），但 halcyon 5 -> 8、
    AD 0 -> 6 —— 那些地方的 128ms 同手本来就该收抗。合计 43 -> 50，故退回 150。
    想再试：改这个值即可（配合 ``space.reach`` 一起看）。
    """

    jack_min_distance: float | str = 0.1
    """抗的位置死区（谱面坐标单位）：0.1 以内算"原地弹一下"。"""

    jack_boost: float = 0.3
    """**小间隔的额外抗罚**（大间隔基本不变）。

    ``jack`` 的时间因子是 ``f = (窗口 - 间隔)/窗口``；这里把它改成
    ``f * (1 + jack_boost * f ** jack_boost_power)`` —— 间隔越小加成越强，
    接近窗口时 ``f`` 本身就趋 0，所以"较大间隔"与原来几乎一样。
    用户口径："jack 的惩罚再调大一点，但依旧随着空间递减，较大间隔时和现在一样"。
    """
    jack_boost_power: float = 2.0
    """上面那个加成的衰减指数（越大越只在极短间隔生效）。"""
    move_speed_power: float = 0.3
    """``move`` 速度项指数：``ratio ** move_speed_power``。

    原值 2（平方）等于把"速度"当主导：同一段位移，间隔减半就贵 4 倍，于是密集串里
    "让手跑过去换手"被算得太贵，模型宁可插同手对。压到 0.3 后位移代价几乎与速度
    无关（只剩"低于舒适移速打五折"和 ``max_speed`` 硬上限），但**保留**了远近差异。

    实测（三张参考谱的四处关切区间）合计 44 -> 26 处：AD 两段全部归零、
    aegleseeker 26 -> 14，infinitestrife 12 不变。``move`` 仍然是用户可调权重。
    """
    jack_decay: float = 1.0
    """抗的空间上坡边界 = 物件宽 × 它（1.0 = 一个轨道宽处到达最重）。"""

    jack_uphill: bool = True
    """抗的空间方向：True = 越远越重（0.1 最轻、0.5 最重）。"""

    jack_hard_boost: float = 1.0
    """超速加成：同手这一下移速超过舒适移速时，额外算一份抗。"""

    jack_time: float = 0.2
    """同键位连按的分量：bullet 按它折价，密集同键连按的底价（取 0 会漏判）。"""

    bullet_stream_tol: float = 0.0
    """**"连续二连"的规整性容差**（实测**没能区分** gengaozo 与 aegleseeker，
    两边都是 ~100ms 的规整流，故保持 0 = 不检查）：免孤立的同位置二连还要满足"前后相邻间隔与这一对
    的间隔相当"（相对差不超过它），即一串节奏规整的二连（用户说的"连续二连"）。
    0 = 不检查规整性。实测 gengaozo 前后都是 98ms（规整 ✓），aegleseeker 那几处不规整。
    """

    bullet_hardness: float = 0.0
    """判定为 bullet 的那一对，hardness（同手连打"手感硬度"）的**折扣系数**。

    用户口径："hardness 也收一点"。实测 gengaozo 的二连处：参考版只多付 0.77，
    而每处二连的 hardness 就是 0.66 —— 不打折就始终被对侧手压住。
    1.0 = 不打折。
    """

    bullet_min_gap: float = 90.0
    """**同位置"二连"不必孤立**的间隔下限（ms）：同位置的同手两下只要间隔不小于它，
    就算 bullet（不必满足孤立性）。用户口径：98ms 这种连续 16 分里的同位置连按，
    手指完全来得及再落一次，玩家眼里就是子弹；而 75ms 那档仍是硬抗，必须罚。
    0 = 关闭（只按孤立性判 bullet）。
    """

    bullet_isolation: float = 150.0
    """bullet 的"相对孤立"窗口（ms）：窗口内有别的键就不算 bullet。"""

    hardness: float = 1.3
    """同手连打的"手感硬度"权重（只看时间间隔）。

    与抗用同一套**连续收尾**（``_taper_factor``：100ms 以内原样、100~120 平滑到 0），
    权重由用户要求"也收一点"从 1.5 调到 1.3（实测 1.1~1.5 对这四张参考无差别，
    只是留出余量）。"""
    """同手连打的"手感硬度"（只看时间，不看位置）。"""

    hardness_window: float = 200.0
    """手感硬度的判定窗口（ms）。"""

    cross: float = 0.6
    """近距离反手加成峰值（抛物线：两端 0、中点峰值）。"""

    cross_near: float = 0.1
    """反手加成生效的近端距离（谱面坐标单位）。"""

    cross_far: float = 0.5
    """反手加成生效的远端距离（谱面坐标单位）。"""

    cross_hand: float = 0.2
    """双手交叉（反手）的固定惩罚。"""

    arc_arrival: float = 0.35
    """位移终点是音弧时的惩罚系数（音弧只有最高一级判定）。"""

    switch: float = 0.1
    """换手代价：相邻物件换一次手记一笔。"""

    switch_window: float = 150.0
    """换手代价的衰减窗口（ms），与 ``jack_window`` 对齐。"""

    busy: float = 20.0
    """长条还没松手就要按下一个键的平罚。"""

    balance: float = 4.0
    """左右手使用量不平衡的惩罚。"""


@dataclass
class SpaceConfig:
    """空间归属：中线左边给左手、右边给右手（固化）。"""

    center_x: float = 0.5
    """左右手拆分的**中线**横坐标（Arcaea 场地真中线）。"""

    window: float = 1200.0
    """局部中心的时间窗（ms）。"""

    weight: float = 0.1
    """动态空间强度（按偏离局部中心多少增长）。"""

    home: float = 0.25
    """静态空间强度（左手偏左、右手偏右，线性无阈值）。

    这是"**静态反手**"的罚：一只手停在错的那一侧。原值 0.06 太轻
    （偏侧 0.25 单位只罚 0.038，比一次换手 0.1 还小），会出现整段分法反过来；
    调到 0.25 后三张新参考谱的"完全反过来"明显减少。**同位置快速交互**
    （见 ``Scoring._tight_interaction``）不做这个判断，交给两手交互。
    """

    lane_reverse: float = 2.0
    """**同刻、同 y、相差整数个轨道（1~3 个 = 0.5/1.0/1.5）** 的一对键的静态反手罚（权重）。

    用户口径：两个键同时开始、y 相同（两个地面键 / 一个地面键 + 一个同 y 的天键 /
    两个同 y 的天键）、x 相差整数个轨道时，**空间左右是硬道理**，反手要狠狠罚。
    这种一对键两手必须分开，于是空间上谁左谁右没有任何余地。取 0 关闭。

    间距**不限于相邻的一个轨道**：1、2、3 个轨道都算（判据见 ``_LANE_PAIR_SPACINGS``）。
    """

    reach: float = 0.0
    """**出张扣分**：左手落到右半边 / 右手落到左半边（越过中线）时的额外扣分。

    用户口径：这种"出张"如果**间隔已经相当大（>=100ms，见 ``_REACH_MS``）**，
    给一个中等扣分 —— 时间足够时手完全来得及摆到正确的一侧，出张没有理由，
    这笔账能打破"反手省下抗罚"的平衡。间隔很短时不收（快速交互里两手本来
    就必须来回跨）。取 0 关闭。
    """

    neutral_band: float = 15.0
    """距离中线多近才算"真正居中"（px）。"""

    strong_bias: float = 20.0
    """空间规则"证据充分"的阈值（px）。"""


@dataclass
class SearchConfig:
    """求解器强度（固化；费时与质量的取舍都在这里）。"""

    beam_width: int = 56
    """束宽：每步保留的最优状态数。"""

    refine_passes: int = 6
    """局部优化轮数（翻转 / 交换 / 镜像 / 前缀翻转）。"""

    refine_radius: int = 3
    """局部优化的邻域半径（按物件序号）。"""

    refine_mirror: bool = True
    """是否在局部优化里尝试"整段镜像"。"""

    mirror_max: int = 5
    """整段镜像改动的最大长度（物件数）。"""


@dataclass
class OutputConfig:
    """输出文件与 info 报告的写法（这一组保留可调）。"""

    suffix: str = ".splited"
    """输出后缀：``x.aff`` -> ``x.splited.aff``。"""

    side_case: str = "upper"
    """``side`` 的写法：``upper`` / ``lower``。"""

    report_dense_gap: float = 220.0
    """统计"交互率"时认为"高密度"的间隔上限（ms）。"""


@dataclass
class Config:
    """一份完整的拆谱配置。"""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    space: SpaceConfig = field(default_factory=SpaceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


#: 允许从命令行调整的键 -> (最小值, 最大值)。其余键一律不接受覆盖。
_TUNABLE_RANGES: dict[str, tuple[float, float]] = {
    "cost.jack": (0.0, 500.0),
    "cost.move": (0.0, 100.0),
}


def validate(config: Config) -> list[str]:
    """检查两个可调项是否在合理范围内，返回警告文本。"""
    warnings: list[str] = []
    for key, (low, high) in _TUNABLE_RANGES.items():
        section, name = key.split(".")
        value = getattr(getattr(config, section), name)
        if not isinstance(value, (int, float)):
            warnings.append(f"{key} 不是数字：{value!r}")
        elif not (low <= float(value) <= high):
            warnings.append(f"{key} = {value} 超出建议范围 [{low}, {high}]")
    if config.output.side_case not in ("upper", "lower"):
        warnings.append(f"output.side_case 只能是 upper / lower：{config.output.side_case!r}")
    return warnings


def resolve_width_expr(value: Any, default_width: float) -> float:
    """把 ``'0.2w'`` 这类"按物件宽"的写法换算成像素；数字原样返回。"""
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("w"):
            return float(text[:-1]) * float(default_width)
        return float(text)
    return float(value)
