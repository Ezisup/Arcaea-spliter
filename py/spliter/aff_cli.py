"""Arcaea 谱面（.aff）的拆谱命令行 —— 本项目的唯一入口。

    python3 spliter/spliter_aff.py <谱面.aff>
    python3 spliter/spliter_aff.py <谱面.aff> -o out.aff --stats
    python3 spliter/spliter_aff.py <谱面.aff> --jack 5 --move 1.2 --explain 20

流程：解析 .aff -> 音弧颜色转成硬约束 -> 拆谱（代价 + 束搜索 + 局部优化）
      -> 只给 tap / hold / arctap 补手标记 0/1（**arc 一律不改**）-> 回写 + 报告。

拆谱本身只有**两个可调项**：``--jack`` / ``--move``；其余参数已固化在
``config.py`` 的默认值里（Arcaea 实测标定）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .aff import apply_arcaea_physics, parse_aff, serialize_aff
from .config import Config, validate
from .constraints import build_arc_constraints, describe_constraints
from .report import FULL_HEIGHT_IN_LANES, analyze, compare, explain
from .split import split

#: 拆谱器内部用字母表示手，AFF 用 0/1。
HAND_TO_INT = {"L": 0, "R": 1}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spliter-aff",
        description=f"Arcaea 拆谱（spliter v{__version__}）：给判定物件补手标记 0/1。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python3 spliter/spliter_aff.py chart.aff
  python3 spliter/spliter_aff.py chart.aff -o out.aff --stats
  python3 spliter/spliter_aff.py chart.aff --jack 6 --move 1.5 --explain 20

记号约定:
  手标记 0 = 左手，1 = 右手。arc 的第 8 个参数 color 就是它的手标记，
  所以 **arc 不会被修改**；只有 tap / hold / arctap 会补上 0/1。

可调参数（只有这两个）:
  --jack  抗（同手连打）权重，默认 4.0 —— 调大更愿意拆交互、更不愿同手连打
  --move  移动/速度权重，默认 1.0 —— 调大更"偷懒"、更倾向就近按键
""",
    )
    parser.add_argument("input", nargs="?", help="输入 .aff 谱面")
    parser.add_argument("-o", "--out", help="输出文件（默认 <输入名>.splited.aff）")
    parser.add_argument("--jack", type=float, default=None, metavar="W",
                        help="抗（同手连打）权重，默认 4.0")
    parser.add_argument("--move", type=float, default=None, metavar="W",
                        help="移动/速度权重，默认 1.0")
    parser.add_argument("--stdout", action="store_true", help="结果打到标准输出，不写文件")
    parser.add_argument("-f", "--force", action="store_true", help="允许覆盖已存在的输出文件")
    parser.add_argument("--stats", action="store_true", help="打印拆谱前后的对比与质量报告")
    parser.add_argument("--explain", nargs="?", const="40", metavar="N",
                        help="逐键说明'为什么给它这只手'（默认 40，0 = 全部）")
    parser.add_argument("--show-constraints", action="store_true",
                        help="打印音弧颜色产生的硬约束（同刻钉死 / 区间限手）")
    parser.add_argument("--json", action="store_true", help="机器可读结果")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _build_config(args: argparse.Namespace) -> tuple[Config, list[str]]:
    """默认配置 + 两个可调项；像素刻度与起手位按 Arcaea 标定设置。

    分段刻度（六轨模式）要等谱面解析完才知道，由 :func:`_apply_schedule` 补上。
    """
    config = Config()
    apply_arcaea_physics(config)
    if args.jack is not None:
        config.cost.jack = float(args.jack)
    if args.move is not None:
        config.cost.move = float(args.move)
    return config, validate(config)


def _apply_schedule(config: Config, parsed) -> None:
    """把谱面里的 ``enwidencamera`` / ``enwidenlanes`` 折成刻度表装进 config。"""
    schedule = parsed.scale_schedule()
    if schedule:
        config.physics.scale_schedule = schedule


def auto_out_path(source: Path) -> Path:
    """默认输出：``xxx.aff`` -> ``xxx.splited.aff``；无后缀则直接加后缀。"""
    if source.suffix:
        return source.with_name(f"{source.stem}.splited{source.suffix}")
    return source.with_name(f"{source.name}.splited")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config, config_warnings = _build_config(args)
    for warning in config_warnings:
        print(f"警告: {warning}", file=sys.stderr)

    if not args.input:
        parser.print_help()
        return 2
    source = Path(args.input)
    if not source.is_file():
        print(f"错误: 找不到输入文件 {args.input}", file=sys.stderr)
        return 2

    # ---- 输出路径与覆盖检查 ---- #
    # **一拿到输入就查**：不要等谱面算完（可能几十秒）才说"文件已存在"。
    out_path: Path | None = None
    if not args.stdout:
        out_path = Path(args.out) if args.out else auto_out_path(source)
        if out_path.exists() and not args.force:
            print(f"错误: {out_path} 已存在。加 --force 覆盖，或用 -o 指定别的路径。",
                  file=sys.stderr)
            return 2

    # ---- 解析 ---- #
    parsed = parse_aff(source.read_text(encoding="utf-8", errors="replace"))
    notes = parsed.chart.notes
    if not notes:
        print(f"错误: {args.input} 里没有解析出任何物件。", file=sys.stderr)
        return 1

    # ---- 硬约束（音弧颜色） ---- #
    _apply_schedule(config, parsed)          # 六轨段落要换刻度，必须在求解之前
    constraints = build_arc_constraints(parsed, notes,
                                        schedule=config.physics.scale_schedule)

    # ---- 求解 ---- #
    started = time.perf_counter()
    result = split(notes, config, parsed.chart.meta, constraints=constraints)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    # ---- 回写（只补判定物件的手标记） ---- #
    hands_int = {note: HAND_TO_INT.get(hand)
                 for note, hand in result.side_by_note.items() if hand in HAND_TO_INT}
    output_text = serialize_aff(parsed, hands_int)

    if args.stdout:
        sys.stdout.write(output_text)
    else:
        assert out_path is not None
        out_path.write_text(output_text, encoding="utf-8")

    # ---- 报告 ---- #
    # **地面与天空合并统计**：arctap 的手速同样是真实负担，而且往往才是最吃紧的
    # 那部分（实测同一份谱面里天空的最大手速比地面高 50%+）。曾经只统计地面、
    # 再把天空当脚注，等于把瓶颈藏在脚注里。
    all_notes = parsed.chart.notes
    after_stats = analyze(_replace_sides(all_notes, result), config)
    before_stats = analyze(all_notes, config)

    if args.json:
        payload = {
            "input": args.input,
            "output": str(out_path) if out_path else None,
            "elapsedMs": elapsed_ms,
            "counts": {
                "ground": len(parsed.ground_notes()),
                "arctap": len(parsed.arctap_notes()),
                "arcs": len(parsed.arcs),
                "solidArcs": len(parsed.solid_arcs()),
            },
            "constraints": constraints.stats,
            "score": result.score.as_dict(),
            "before": {"L": before_stats.L, "R": before_stats.R, "none": before_stats.none},
            "after": {"L": after_stats.L, "R": after_stats.R, "none": after_stats.none,
                      "alternationRatio": after_stats.alternation_ratio,
                      "maxSpeedSeen": after_stats.max_speed_seen,
                      "maxLaneSpeed": after_stats.max_lane_speed},
            "warnings": [*parsed.warnings, *config_warnings, *result.warnings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.quiet:
        lines = _report_lines(args, parsed, result, out_path, elapsed_ms, before_stats,
                              after_stats, constraints, config_warnings, config)
        stream = sys.stderr if args.stdout else sys.stdout
        print("\n".join(lines), file=stream)
    return 0


def _rank_gaps(records):
    """把已按间隔排好序的手速榜条目分组排名。

    同一间隔算**并列**（同名次），只拿第一对举例，并报出该名次共有几组。
    返回 ``[(rank, record, tied_count), ...]``，每个名次只出一条。
    """
    ranked = []
    rank = 0
    index = 0
    while index < len(records):
        gap = records[index].gap_ms
        same = [r for r in records[index:] if r.gap_ms == gap]
        rank += 1
        ranked.append((rank, records[index], len(same)))
        index += len(same)
    return ranked


def _replace_sides(notes, result):
    from dataclasses import replace
    return [replace(note, side=result.side_by_note.get(note)) for note in notes]


def _report_lines(args, parsed, result, out_path, elapsed_ms, before_stats, after_stats,
                  constraints, config_warnings, config) -> list[str]:
    ground = parsed.ground_notes()
    arctaps = parsed.arctap_notes()
    counts = result.counts()
    lines = [
        f"输入: {args.input}  ({len(parsed.chart.notes)} 个判定物件"
        f" = 地面 {len(ground)} + 天空 {len(arctaps)}；音弧 {len(parsed.arcs)} 条"
        f"，其中实体 {len(parsed.solid_arcs())} 条；时长 {parsed.duration / 1000:.1f}s)",
    ]
    if out_path:
        lines.append(f"输出: {out_path}")
    else:
        lines.append("输出: <stdout>")
    lines.append(f"手标记: L={counts['L']} R={counts['R']} 未标记={counts['none']}"
                 f"（arc 未做任何修改）")
    lines.append(f"求解: {elapsed_ms}ms, 代价 {result.score.total:.2f} "
                 f"(移动 {result.score.move:.2f} / 抗 {result.score.jack:.2f}"
                 f" / 空间 {result.score.bias:.2f} / 换手 {result.score.switch:.2f}"
                 f" / 占手 {result.score.busy:.2f} / 均衡 {result.score.balance:.2f})")
    if args.show_constraints:
        lines.extend(["", "音弧约束:"])
        lines.extend("  " + line for line in describe_constraints(constraints, parsed.chart.notes))
    if args.stats:
        # 位移一律用"跨了几个轨道"讲（1 轨道 = 0.5 坐标 = 相邻轨道间距）：
        # 像素对玩家毫无体感、还随设备分辨率变化，而轨道数是谱面上能直接数的。
        lines.extend([
            "",
            compare(before_stats, after_stats, "拆前", "拆后"),
            "",
            f"最吃位移速度的几处（满高 = {FULL_HEIGHT_IN_LANES:.2f} 轨道；"
            f"这是“手要跑多远”，不是手速）:",
        ])
        lines.extend(f"  {move.describe(config.physics)}" for move in after_stats.hard_moves)
        # 手速榜：同手相邻两键的**最小间隔**，与位移无关。
        # 只看 tap / arctap / hold 头 —— 它们才是"手要再落一次"。
        lines.extend(["", "手速榜（同手相邻两键的最小间隔，与位移无关）:"])
        if after_stats.fastest_gaps:
            # 只显示前几名；**统计用的是全量列表**，并列组数不会因显示截断而少算
            ranked = _rank_gaps(after_stats.fastest_gaps)[:5]
            lines.extend(f"  {record.describe(rank, tied)}"
                         for rank, record, tied in ranked)
        else:
            lines.append("  （没有同手相邻的判定物件）")
        notes = [*result.warnings, *after_stats.warnings]
        if notes:
            lines.extend(["", "需要注意:"])
            lines.extend(f"  - {note}" for note in notes)
    if args.explain is not None:
        limit = None if str(args.explain) == "0" else int(args.explain)
        # 地面 / 天空都给：arctap 的归属同样是判断依据的一部分。
        lines.extend(["", "逐键依据（全部判定物件）:"])
        rows = explain(result, limit=limit)
        for row in rows:
            lines.append(f"  {row.describe()}")
        if limit is not None and len(rows) >= limit:
            lines.append(f"  （已截断到 {limit} 条；用 --explain 0 看全部）")
        lines.append("  （* = 原生标记，不参与决策）")
    for warning in [*parsed.warnings, *config_warnings]:
        lines.append(f"警告: {warning}")
    return lines


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
