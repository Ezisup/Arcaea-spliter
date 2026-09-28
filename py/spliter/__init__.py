"""拆谱程序（Arcaea 专用）：读 .aff -> 给每个判定物件分配左右手 -> 写一份副本。

模块：

- :mod:`spliter.aff`         —— AFF 解析 / 无损回写 / 几何标定
- :mod:`spliter.constraints` —— 音弧颜色与长条占手构成的硬约束
- :mod:`spliter.split`       —— 拆谱核心（代价函数 + 束搜索 + 局部优化）
- :mod:`spliter.report`      —— 质量评估与逐键解释
- :mod:`spliter.config`      —— 参数（已固化，只有 ``jack`` / ``move`` 可调）
- :mod:`spliter.aff_cli`     —— 命令行入口

命令行入口::

    python3 spliter/spliter_aff.py 谱面.aff
"""

from __future__ import annotations

from .aff import AffArc, AffChart, apply_arcaea_physics, parse_aff, serialize_aff
from .chart import Note
from .config import Config
from .constraints import ArcConstraints, build_arc_constraints
from .report import analyze, compare, explain
from .split import SplitResult, split

__version__ = "0.2.0"

__all__ = [
    "AffArc",
    "AffChart",
    "ArcConstraints",
    "Config",
    "Note",
    "SplitResult",
    "analyze",
    "apply_arcaea_physics",
    "build_arc_constraints",
    "compare",
    "explain",
    "parse_aff",
    "serialize_aff",
    "split",
    "__version__",
]
