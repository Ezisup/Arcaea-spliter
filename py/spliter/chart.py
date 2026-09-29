"""谱面数据模型：一个判定物件（:class:`Note`）与一份谱面（:class:`Chart`）。

AFF 的解析与回写在 :mod:`spliter.aff` 里（``parse_aff`` / ``serialize_aff``）；
这个模块只提供两者共用的数据结构，以及 ``Note`` 上的几个便利属性
（``locked`` / ``param`` / ``width`` / ``duration``）。

约定：位置用谱面坐标（Arcaea 是"单位"，不是像素），时间用毫秒；
``Note.line`` 记录它在原文里的行号，回写时按行替换，其余行逐字不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

DEFAULT_META: dict[str, object] = {
    "title": "(untitled)",
    "fallSpeed": 460,
    "judgeLineY": 600,
    "noteWidth": 120,
    "noteHeight": 16,
    "loop": True,
}


def _to_number(text: str) -> float | None:
    """把参数字符串转成数字；不是数字返回 ``None``。"""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #

@dataclass(eq=False)   # eq=False -> 用对象身份做键：物件本身可以当 dict 的 key
class Note:
    """一个物件。

    ``index`` 是**时间升序**后的序号（0 基），``raw_index`` 是文件里的原始顺序。
    两者可能不同 —— 回写谱面请用物件对象本身（见 :func:`serialize_chart`），
    不要用 ``index`` 去对齐行。
    """

    time: int                       # 判定时间（ms，取整）
    x: float                        # 横向位置（画面中心 0，向右为正）
    y: float = 0.0                  # 纵向位置（0 = 地面判定线，向上为正）
                                    # 地面音符恒为 0；
                                    # Arcaea 的 Arctap 可以出现在平面内任意高度
    type: str = "tap"               # tap / hold / ...
    params: dict[str, object] = field(default_factory=dict)
    params_raw: list[str] = field(default_factory=list)
    side: str | None = None         # 原生 'L'/'R' 标记
    side_token: int = -1            # 原始 side token 的下标（-1 = 没有）
    line: int = 0                   # 1 基行号
    raw_line: str = ""              # 原始整行（不含换行符）
    index: int = 0                  # 时间升序后的序号
    raw_index: int = 0              # 文件里的原始顺序
    noinput: int = 0                # 所在 ``timinggroup(noinput)`` 的组号
                                    # （0 = 普通；>0 = 隔离组，见 ``split.Scoring.isolates``）

    # ---- 便捷取值（供代价函数用，避免到处写 params.get） ---- #
    @property
    def locked(self) -> bool:
        """原生带标记（拆谱时必须尊重）。"""
        return self.side is not None

    def param(self, *names: str, default: object = None) -> object:
        """按若干候选键名取参数（``w`` / ``width`` 这类别名）。"""
        for name in names:
            if name in self.params:
                return self.params[name]
        return default

    @property
    def width(self) -> float | None:
        """单行覆盖的物件宽度（没写返回 ``None``）。"""
        value = self.param("w", "width")
        if value is None:
            return None
        number = _to_number(str(value))
        return number if number and number > 0 else None

    @property
    def duration(self) -> float | None:
        """长条尾巴长度（没写返回 ``None``）。"""
        value = self.params.get("duration")
        if value is None:
            return None
        number = _to_number(str(value))
        return number if number is not None and number >= 0 else None


@dataclass
class Chart:
    """一份解析好的谱面。"""

    notes: list[Note] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)
    raw_meta: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)   # 原文按行切分（带行尾符）
    newline: str = "\n"

    @property
    def duration(self) -> int:
        """谱面时长 = 最后一个物件的判定时间。"""
        return self.notes[-1].time if self.notes else 0


def sort_notes(notes: list[Note]) -> list[Note]:
    """按 (时间, x) 升序排序并回填 ``index``。"""
    notes.sort(key=lambda note: (note.time, note.x))
    for index, note in enumerate(notes):
        note.index = index
    return notes


def isolated(source_group: int, target_group: int) -> bool:
    """``source_group`` 的约束是否**不该**作用于 ``target_group``。

    ``timinggroup(noinput)`` 是演出组：里面的物件不受玩家输入，但仍要上手标记。
    所以取折中 —— 组内物件**可以**受组外与**本组**影响，但**不能**影响组外。

    放在这里是因为 ``split`` / ``constraints`` / ``report`` 三处都要用，
    而 ``constraints`` 不能 import ``split``（循环依赖）。**三处共用这一条口径**，
    不要各写一份，否则迟早漂移。
    """
    return source_group != 0 and source_group != target_group


