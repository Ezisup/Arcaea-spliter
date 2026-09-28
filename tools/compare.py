"""与人工参考对照：**自动锁定"初次差异 ±10s"的区间**，之后只复查这些区间。

用法::

    # 第一次：全谱求解 -> 找出与参考不同的地方 -> 每处取前后 10s（重叠才相连）。
    # 区间写进 <谱面>.watch；这次报告的是**初次差异全貌**。
    python3 spliter/tools/compare.py halcyon_2 halcyon_manual.aff

    # 之后：只解这些区间（±margin 上下文），只统计区间内的差异。
    # 无论后来改了什么，观察范围都不变；要重定标准就删掉 .watch 或加 --fresh。
    python3 spliter/tools/compare.py halcyon_2 halcyon_manual.aff

    # 临时手查某一段（不看 .watch）
    python3 spliter/tools/compare.py 谱面 参考 --range 127535-147535

约定与细节：

- 参考里**没有标手**的物件直接跳过（不算差异）。
- 区间合并规则：每处差异 ``[t-10s, t+10s]``，**重叠才相连**，不额外延长。
- 区间总覆盖超过全谱一半时，直接整谱求解再筛（更省事也更快）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

from spliter.aff import apply_arcaea_physics, parse_aff          # noqa: E402
from spliter.config import Config                                 # noqa: E402
from spliter.constraints import build_arc_constraints             # noqa: E402
from spliter.split import split                                   # noqa: E402

HAND = {0: "L", 1: "R"}
OTHER = {"L": "R", "R": "L"}
WATCH_WINDOW_MS = 10_000


def reference_hands(path: str) -> dict:
    """参考里每个判定物件的手（没标的就是 ``None``，比对时跳过）。"""
    parsed = parse_aff(Path(path).read_text(encoding="utf-8", errors="replace"))
    table: dict = {}
    for arctap in parsed.arctaps:
        table[(arctap.time, round(arctap.arc.x_at(arctap.time), 4), "arctap")] = HAND.get(arctap.hand)
    for note in parsed.chart.notes:
        if note.type != "arctap":
            table[(note.time, round(note.x, 4), note.type)] = HAND.get(note.params.get("_hand"))
    return table


def parse_regions(text: str) -> list[tuple[int, int]]:
    regions = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lo, _, hi = line.partition("-")
        regions.append((int(lo), int(hi) if hi else int(lo)))
    return regions


def merge_windows(times: list[int], window: int = WATCH_WINDOW_MS) -> list[tuple[int, int]]:
    """每处差异取前后 ``window``，**重叠才相连**（不额外延长）。"""
    spans = sorted((t - window, t + window) for t in times)
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _apply_overrides(specs) -> None:
    """临时改参数：``space.reach=0.5`` / ``cost.jack=5`` / ``_TAPER_HARD_END=180``。"""
    import importlib
    for spec in specs:
        key, _, raw = spec.partition("=")
        low = raw.strip().lower()
        if low in ("true", "false"):
            value: object = low == "true"
        else:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        if key.startswith("_"):
            setattr(importlib.import_module("spliter.split"), key, value)
        else:
            obj: object = _CONFIG
            *path, last = key.split(".")
            for part in path:
                obj = getattr(obj, part)
            setattr(obj, last, value)


_CONFIG = Config()


def solve_core(parsed, ranges, margin, overrides=None):
    notes = parsed.chart.notes
    chosen = list(notes) if ranges is None else [
        n for n in notes if any(lo - margin <= n.time <= hi + margin for lo, hi in ranges)]
    config = Config()
    apply_arcaea_physics(config)
    config.physics.scale_schedule = parsed.scale_schedule()   # 六轨模式换刻度
    global _CONFIG
    _CONFIG = config
    _apply_overrides(overrides or [])
    constraints = build_arc_constraints(parsed, chosen,
                                        schedule=config.physics.scale_schedule)
    started = time.perf_counter()
    result = split(chosen, config, parsed.chart.meta, constraints=constraints)
    return chosen, result, time.perf_counter() - started


def collect(notes, result, table, in_scope):
    """区间内的差异（参考没标手的跳过）。"""
    misses = []
    for index, note in enumerate(notes):
        if not in_scope(note.time):
            continue
        want = table.get((note.time, round(note.x, 4), note.type))
        if want is None:
            continue
        if result.hands[index] != want:
            misses.append((note.time, note.type, round(note.x, 3), round(note.y, 2),
                           result.hands[index], want))
    return misses


def report(misses, elapsed, label):
    inverted = sum(1 for m in misses if m[4] and m[5] and m[4] == OTHER.get(m[5]))
    clusters: list[list] = []
    for miss in sorted(misses):
        if clusters and miss[0] - clusters[-1][-1][0] <= 400:
            clusters[-1].append(miss)
        else:
            clusters.append([miss])
    print(f"{label}：差异 {len(misses)}（'完全反过来' {inverted}）"
          f" 簇 {len(clusters)} 个，用时 {elapsed:.1f}s")
    for cluster in clusters:
        span = f"{cluster[0][0]}~{cluster[-1][0]}" if len(cluster) > 1 else str(cluster[0][0])
        print(f"  --- {span}（{len(cluster)} 处）")
        for time_ms, ntype, x, y, mine, want in cluster[:8]:
            tag = "完全反过来" if mine == OTHER.get(want) else "单键不同"
            print(f"      t={time_ms:>7} {ntype:<6} x={x:+.3f} y={y:+.2f} 我={mine} 参考={want}  [{tag}]")
    return clusters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="与人工参考对照（自动锁定观察区间）")
    parser.add_argument("chart", help="谱面")
    parser.add_argument("reference", help="人工参考")
    parser.add_argument("--watch", default=None, help="观察区间文件（默认 <谱面>.watch）")
    parser.add_argument("--fresh", action="store_true", help="忽略已有区间，按当前差异重算")
    parser.add_argument("--range", action="append", default=[], metavar="LO-HI",
                        help="临时手查这一段（可重复；给出时不用 .watch）")
    parser.add_argument("--margin", type=int, default=3000, help="--fast 时区间两侧上下文（ms）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="临时改参数（space.reach=0.5 / _TAPER_HARD_END=180）")
    parser.add_argument("--fast", action="store_true",
                        help="按区间切片求解（快，但上下文短、结果可能与整谱不同）")
    args = parser.parse_args(argv)

    parsed = parse_aff(Path(args.chart).read_text(encoding="utf-8", errors="replace"))
    table = reference_hands(args.reference)

    if args.range:
        regions = parse_regions("\n".join(args.range))
        chosen, result, elapsed = solve_core(parsed, None if not args.fast else regions,
                                             args.margin, args.set)
        misses = collect(chosen, result, table,
                         lambda t: any(lo <= t <= hi for lo, hi in regions))
        report(misses, elapsed, f"手查区间 {regions}")
        return 0

    # 观察区间文件带上参考名：换一版参考就是换一套标准，不该串味。
    watch_path = Path(args.watch or f"{args.chart}.{Path(args.reference).name}.watch")
    if watch_path.exists() and not args.fresh:
        regions = parse_regions(watch_path.read_text(encoding="utf-8"))
        print(f"观察区间（来自 {watch_path}）：{regions}")
        # 默认**整谱求解**再按区间筛：切片虽然快，但上下文短，结果可能与整谱不同
        # （实测 infinitestrife 逐区间切片 20 处、整谱 12 处）。要快就用 --fast。
        if not args.fast:
            chosen, result, elapsed = solve_core(parsed, None, 0, args.set)
            misses = collect(chosen, result, table,
                             lambda t: any(lo <= t <= hi for lo, hi in regions))
        else:
            misses, elapsed = [], 0.0
            for lo, hi in regions:
                chosen, result, took = solve_core(parsed, [(lo, hi)], args.margin, args.set)
                elapsed += took
                misses += collect(chosen, result, table,
                                  lambda t, lo=lo, hi=hi: lo <= t <= hi)
        report(misses, elapsed, "区间内")
        return 0

    # ---- 第一次：全谱求解，按差异定区间并落盘 ----
    chosen, result, elapsed = solve_core(parsed, None, 0, args.set)
    misses = collect(chosen, result, table, lambda t: True)
    regions = merge_windows([m[0] for m in misses])
    watch_path.write_text(
        f"# {args.chart} vs {args.reference} 的观察区间\n"
        f"# 规则：初次差异每处取前后 {WATCH_WINDOW_MS // 1000}s，重叠才相连、不延长\n"
        + "".join(f"{lo}-{hi}\n" for lo, hi in regions),
        encoding="utf-8")
    report(misses, elapsed, "初次全谱")
    print(f"已写入观察区间 {watch_path}：{regions}")
    print("以后每次只复查这些区间（要看新的差异就删掉该文件或加 --fresh）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
