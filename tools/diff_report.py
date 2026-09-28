#!/usr/bin/env python3
"""把脚本输出与"人工参考答案"逐键对照，生成一张不一致对照表（Markdown）。

用法::

    python3 spliter/tools/diff_report.py chaos.aff chaos_manual_adjusted.aff
    python3 spliter/tools/diff_report.py chart.aff reference.aff -o diff.md

两边的谱面必须**结构相同**（arc 参数逐字一致、判定物件完全相同）—— 这只比较
手标记。表里每行给：时间 / 类型 / x / y / 脚本给的手 / 参考给的手 / 该时刻的
其它物件（便于判断"这一组整体反了"还是"单个键不同"）。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

from spliter import Config                                    # noqa: E402
from spliter.aff import apply_arcaea_physics, parse_aff       # noqa: E402
from spliter.constraints import build_arc_constraints         # noqa: E402
from spliter.split import Scoring, split                      # noqa: E402

INT_TO_HAND = {0: "L", 1: "R"}


def reference_hands(path: str) -> dict[tuple[int, str, float], str | None]:
    """人工参考答案里的手标记，键是 ``(时间, 类型, x)``。"""
    parsed = parse_aff(Path(path).read_text(encoding="utf-8", errors="replace"))
    out: dict[tuple[int, str, float], str | None] = {}
    for arc in parsed.arctaps:
        out[(arc.time, "arctap", round(arc.arc.x_at(arc.time), 4))] = INT_TO_HAND.get(arc.hand)
    for note in parsed.chart.notes:
        if note.type != "arctap":
            out[(note.time, note.type, round(note.x, 4))] = \
                INT_TO_HAND.get(note.params.get("_hand"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="与人工参考答案对照，输出不一致表")
    ap.add_argument("chart", help="待拆的谱面（.aff）")
    ap.add_argument("reference", help="人工参考答案（.aff）")
    ap.add_argument("-o", "--out", default="", help="输出 Markdown 路径（默认只打印）")
    ap.add_argument("--no-physics", action="store_true", help="不做 Arcaea 标定")
    args = ap.parse_args()

    source = Path(args.chart).read_text(encoding="utf-8", errors="replace")
    parsed = parse_aff(source)
    want = reference_hands(args.reference)
    config = Config()
    if not args.no_physics:
        apply_arcaea_physics(config)
    # 六轨模式（enwiden*）要换刻度；工具必须和 CLI 用同一套口径，否则代价对不上。
    config.physics.scale_schedule = parsed.scale_schedule()
    constraints = build_arc_constraints(parsed, parsed.chart.notes,
                                        schedule=config.physics.scale_schedule)

    started = time.perf_counter()
    result = split(parsed.chart.notes, config, parsed.chart.meta, constraints=constraints)
    elapsed = time.perf_counter() - started

    keys = [(n.time, n.type, round(n.x, 4)) for n in parsed.chart.notes]
    got = dict(zip(keys, result.hands))

    # 按"相邻的不一致键"聚成簇（<=400ms 算同一簇），然后把**整簇**翻成参考那侧，
    # 看总代价差多少：接近 0 说明是平局（两种分法在模型里等价），明显为正才是真分歧。
    # 必须按簇算 —— 这些差异常常是"一整段交替的相位反了"，只翻单个键会
    # 把下一对变成新的冲突，得出误导性的巨大差额。
    scoring = Scoring(parsed.chart.notes, config, parsed.chart.meta, constraints)
    current = scoring.evaluate(result.hands)
    bad_indexes = sorted((i for i, k in enumerate(keys) if want.get(k) != got.get(k)),
                         key=lambda i: parsed.chart.notes[i].time)
    clusters: list[list[int]] = []
    for index in bad_indexes:
        if clusters and parsed.chart.notes[index].time \
                - parsed.chart.notes[clusters[-1][-1]].time <= 400:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    cluster_delta: dict[int, float | None] = {}
    for members in clusters:
        trial = list(result.hands)
        blocked = False
        for index in members:
            want_hand = want[keys[index]]
            if want_hand not in ("L", "R") or not scoring.can_hand(index, want_hand):
                blocked = True
                break
            trial[index] = want_hand
        delta = None if blocked else scoring.evaluate(trial).total - current.total
        for index in members:
            cluster_delta[index] = delta
    missing = [k for k in keys if want.get(k) != got.get(k)]
    unassigned = [k for k in keys if got.get(k) not in ("L", "R")]

    by_time: dict[int, list[int]] = {}
    for index, note in enumerate(parsed.chart.notes):
        by_time.setdefault(note.time, []).append(index)

    lines = [
        f"# {Path(args.chart).name} 与人工参考答案的不一致对照表",
        f"# 共 {len(missing)} 处 / {len(keys)} 个物件；未分配 {len(unassigned)}",
        f"# 代价 {result.score.total:.2f} = " + " + ".join(
            f"{k} {v:.1f}" for k, v in result.score.as_dict().items() if k != "total"),
        f"# 求解 {elapsed:.1f}s",
        "# 「本簇翻成你的代价」= 把相邻(<=400ms)的不一致键**整簇**改成参考那侧后，"
        "总代价的变化；接近 0 表示两种分法在模型里等价（平局）。",
        "",
        f"{'时间':>7} {'类型':<7} {'x':>6} {'y':>5}  {'我的':<4} {'你的':<4}  "
        f"{'本簇翻成你的代价':>16}  同刻其它物件（x/类型/我/你）",
    ]
    for time_ms, note_type, x in sorted(k for k in keys if want.get(k) != got.get(k)):
        note = next(n for n in parsed.chart.notes
                    if (n.time, n.type, round(n.x, 4)) == (time_ms, note_type, x))
        others = " ".join(
            f"{parsed.chart.notes[j].x:+.2f}{parsed.chart.notes[j].type} "
            f"我={result.hands[j]} 你={want.get(keys[j])}"
            for j in sorted(by_time[note.time], key=lambda k: parsed.chart.notes[k].x)
            if parsed.chart.notes[j] is not note)
        delta = cluster_delta.get(
            next(i for i in range(len(keys)) if keys[i] == (time_ms, note_type, x)), None)
        if delta is None:
            shown = "硬约束不允许"
        elif abs(delta) <= 0.1:
            shown = f"{delta:+.3f} 平局"
        else:
            shown = f"{delta:+.3f}"
        lines.append(f"{time_ms:>7} {note_type:<7} {x:>+6.2f} {note.y:>5.2f}  "
                     f"{got.get((time_ms, note_type, x))!s:<4} "
                     f"{want.get((time_ms, note_type, x))!s:<4}  {shown:>14}  {others}")
    text = "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"写出 {args.out}（{len(missing)} 处不一致）", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
