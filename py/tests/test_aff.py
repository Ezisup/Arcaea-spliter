"""Arcaea（.aff）拆谱的专项自测。

跑法::

    python3 spliter/py/tests/test_aff.py

覆盖：AFF 解析、坐标/缓动换算、回写的"只补地面音符"、音弧颜色硬约束。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SPLITER = Path(__file__).resolve().parents[2]        # spliter/
PY_ROOT = SPLITER / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from spliter import Config                                   # noqa: E402
from spliter.chart import Note                               # noqa: E402
from spliter.aff import (                                    # noqa: E402
    PIXELS_PER_UNIT_X,
    PIXELS_PER_UNIT_Y,
    AffArc,
    apply_arcaea_physics,
    arc_x_at,
    LANE_CENTER,
    lane_to_x,
    parse_aff,
    serialize_aff,
    x_to_lane,
)
from spliter.constraints import build_arc_constraints        # noqa: E402
from spliter.split import Scoring, screen_span, split         # noqa: E402

#: 一个自足的小 AFF：含 tap / hold / 实体弧 / 黑线 + arctap。
TINY_AFF = """AudioOffset:12
TimingPointDensityFactor:1
-
timing(0,172.00,4.00);
(0,1);
(1000,2);
hold(2000,2600,4);
arc(0,2000,-0.5,1.5,s,0,1,0,none,false);
arc(0,4000,0.25,0.25,s,1,1,1,none,true)[arctap(500),arctap(1500)];
"""


class LaneCoordinateTest(unittest.TestCase):
    """地面轨道 <-> Arc 坐标的换算。"""

    def test_lane_to_x(self) -> None:
        # 每条轨道宽 0.5，tap/hold 在轨道中点：x = -0.25 + (lane-1) * 0.5
        # 4 条轨道的 5 条边界 = -0.5 / 0 / 0.5 / 1.0 / 1.5
        self.assertAlmostEqual(lane_to_x(1), -0.25)
        self.assertAlmostEqual(lane_to_x(2), 0.25)
        self.assertAlmostEqual(lane_to_x(3), 0.75)
        self.assertAlmostEqual(lane_to_x(4), 1.25)

    def test_round_trip(self) -> None:
        for lane in (1, 1.5, 2, 2.5, 3.75, 4):
            self.assertAlmostEqual(x_to_lane(lane_to_x(lane)), lane)

    def test_ground_is_symmetric_inside_arc_range(self) -> None:
        """地面 4 条轨道应落在 Arc 标准范围 [-0.5, 1.5] 内，且关于中心对称。"""
        xs = [lane_to_x(lane) for lane in (1, 2, 3, 4)]
        self.assertGreater(xs[0], -0.5)
        self.assertLess(xs[-1], 1.5)
        # 对称中心 = 四条轨道的中间 = **场地真中线 0.5**
        # （2 轨中点 0.25、3 轨中点 0.75，公共边界 0.5 就是中线）
        center = (xs[0] + xs[-1]) / 2
        self.assertAlmostEqual(center, 0.5)
        self.assertAlmostEqual(center, LANE_CENTER)
        self.assertAlmostEqual(lane_to_x(2) - xs[0], xs[-1] - lane_to_x(3))


class ArcEasingTest(unittest.TestCase):
    """arc 横坐标随时间的插值。"""

    def _arc(self, **kwargs) -> AffArc:
        base = dict(t1=0, t2=1000, x1=0.0, x2=1.0, easing="s", y1=0.0, y2=1.0,
                    color=0, arctype="false")
        base.update(kwargs)
        return AffArc(**base)

    def test_endpoints_and_midpoint(self) -> None:
        arc = self._arc()
        self.assertAlmostEqual(arc_x_at(arc, 0), 0.0)
        self.assertAlmostEqual(arc_x_at(arc, 1000), 1.0)
        self.assertAlmostEqual(arc_x_at(arc, 500), 0.5)

    def test_si_curve_is_not_linear(self) -> None:
        arc = self._arc(easing="si")
        # si = 先快后慢：中点应该已经过半
        self.assertGreater(arc_x_at(arc, 500), 0.5)
        # so = 先慢后快：中点应该还不到一半
        self.assertLess(arc_x_at(self._arc(easing="so"), 500), 0.5)

    def test_composite_easing_uses_x_axis(self) -> None:
        """siso：X 用 si、Y 用 so —— 横坐标要按 si 走。"""
        arc = self._arc(easing="siso")
        self.assertAlmostEqual(arc_x_at(arc, 500), 0.75)

    def test_zero_length_arc(self) -> None:
        """水平弧（t1 == t2）停在起始 x。"""
        arc = self._arc(t1=500, t2=500, x1=0.3, x2=0.9)
        self.assertAlmostEqual(arc.x_at(500), 0.3)


class ParseTest(unittest.TestCase):
    """AFF 解析。"""

    def test_parses_all_note_kinds(self) -> None:
        parsed = parse_aff(TINY_AFF)
        kinds = {}
        for note in parsed.chart.notes:
            kinds[note.type] = kinds.get(note.type, 0) + 1
        self.assertEqual(kinds, {"tap": 2, "hold": 1, "arctap": 2})
        self.assertEqual(len(parsed.arcs), 2)
        self.assertEqual(parsed.audio_offset, 12)
        self.assertEqual(parsed.density_factor, 1.0)
        self.assertEqual(parsed.warnings, [])

    def test_header_and_body_split(self) -> None:
        parsed = parse_aff(TINY_AFF)
        self.assertEqual(parsed.header.get("AudioOffset"), "12")
        self.assertEqual([t[0] for t in parsed.timings], [0])

    def test_ground_note_lane_becomes_x(self) -> None:
        parsed = parse_aff(TINY_AFF)
        taps = [n for n in parsed.chart.notes if n.type == "tap"]
        self.assertAlmostEqual(taps[0].x, -0.25, msg="lane 1 -> -0.25")
        self.assertAlmostEqual(taps[1].x, 0.25, msg="lane 2 -> 0.25")

    def test_hold_keeps_duration(self) -> None:
        parsed = parse_aff(TINY_AFF)
        hold = next(n for n in parsed.chart.notes if n.type == "hold")
        self.assertEqual(hold.time, 2000)
        self.assertEqual(hold.params["duration"], 600)

    def test_arctap_position_comes_from_arc(self) -> None:
        """arctap 没有自己的坐标，必须由所属音弧算出。"""
        parsed = parse_aff(TINY_AFF)
        arctaps = parsed.arctap_notes()
        self.assertEqual(len(arctaps), 2)
        # 那条黑线 x1 = x2 = 0.25，所以两个 arctap 都落在 0.25
        for note in arctaps:
            self.assertAlmostEqual(note.x, 0.25)
        self.assertEqual([n.time for n in arctaps], [500, 1500])

    def test_arctap_hand_is_parsed(self) -> None:
        """arctap 可以带手标记：arctap(t,0) / arctap(t,1)。"""
        parsed = parse_aff("-\narc(0,100,0,0,s,1,1,0,none,true)[arctap(50,1),arctap(60)];\n")
        hands = [(a.time, a.hand, a.is_marked) for a in parsed.arctaps]
        self.assertEqual(hands, [(50, 1, True), (60, None, False)])

    def test_arc_classification(self) -> None:
        parsed = parse_aff(TINY_AFF)
        solid = [a for a in parsed.arcs if a.is_solid]
        trace = [a for a in parsed.arcs if a.is_trace]
        self.assertEqual(len(solid), 1)
        self.assertEqual(len(trace), 1)
        self.assertEqual(solid[0].color, 0)
        self.assertTrue(trace[0].arctaps, "黑线上带 arctap")

    def test_flick_is_ignored_with_warning(self) -> None:
        """flick 是废弃格式，不预留、只提示。"""
        parsed = parse_aff("-\nflick(100,0.5,1.0,0,1);\n")
        self.assertEqual(parsed.chart.notes, [])
        self.assertTrue(any("flick" in w for w in parsed.warnings))

    def test_arctap_out_of_range_warns(self) -> None:
        parsed = parse_aff("-\narc(0,100,0,0,s,1,1,0,none,true)[arctap(999)];\n")
        self.assertTrue(any("不在弧区间" in w for w in parsed.warnings))
        self.assertEqual(len(parsed.arctap_notes()), 1, "越界也保留，只是警告")


class SerializeTest(unittest.TestCase):
    """回写：只给地面音符补手标记。"""

    def _hands(self, parsed):
        result = split(parsed.chart.notes, Config(), parsed.chart.meta)
        return {note: (0 if hand == "L" else 1)
                for note, hand in result.side_by_note.items() if hand in ("L", "R")}

    def test_all_judgeables_get_a_hand(self) -> None:
        """tap / hold / arctap 三类都要标记；arc 的参数一律不动。"""
        parsed = parse_aff(TINY_AFF)
        text = serialize_aff(parsed, self._hands(parsed))
        lines = text.splitlines()
        # 地面音符
        self.assertRegex([line for line in lines if line.startswith("(")][0], r",[01]\);$")
        self.assertRegex([line for line in lines if line.startswith("hold(")][0], r",[01]\);$")
        # arctap：arctap(t) -> arctap(t,0/1)
        marked = re.findall(r"arctap\((\d+),([01])\)", text)
        self.assertEqual(len(marked), 2, f"两个 arctap 都要标记：{text}")
        # arc 的参数部分必须逐字不动（只允许 arctap 括号内变化）
        def arc_args(line: str) -> str:
            return line[line.find("("): line.find(")") + 1]
        self.assertEqual([arc_args(l) for l in lines if l.startswith("arc(")],
                         [arc_args(l) for l in TINY_AFF.splitlines() if l.startswith("arc(")],
                         "arc(...) 的参数不能改")

    def test_other_lines_untouched(self) -> None:
        """除"可判定物件所在的行"外，其它行必须逐字不变。

        可判定的行 = 地面音符行，以及挂着 arctap 的 arc 行（arctap 会补手标记，
        但 arc(...) 自己的参数不动，这一点由 test_all_judgeables_get_a_hand 保证）。
        """
        parsed = parse_aff(TINY_AFF)
        text = serialize_aff(parsed, self._hands(parsed))
        before = TINY_AFF.splitlines()
        after = text.splitlines()
        self.assertEqual(len(before), len(after), "行数不能变")

        def is_ground(line: str) -> bool:
            # 地面音符行 + 挂着 arctap 的 arc 行（arctap 括号内会多个手标记）
            return line.startswith("(") or line.startswith("hold(") or "arctap(" in line

        self.assertEqual([l for l in before if not is_ground(l)],
                         [l for l in after if not is_ground(l)],
                         "非地面行必须原样保留")
        # 地面行的个数与顺序也要保持（它们只是各多了一个参数）
        self.assertEqual(len([l for l in before if is_ground(l)]),
                         len([l for l in after if is_ground(l)]))

    def test_no_duplicate_hand_on_rerun(self) -> None:
        """已经带手标记的文件再写一次：原位替换，不能变成 ``(t,lane,0,1)``。"""
        parsed = parse_aff(TINY_AFF)
        hands = self._hands(parsed)
        first = serialize_aff(parsed, hands)
        # 用同一份手标记再写一遍（模拟"对已拆过的文件重跑"）
        again = parse_aff(first)
        same_hands = {note: hands[note] for note in again.chart.notes if note in hands}
        second = serialize_aff(again, same_hands)
        self.assertEqual(first, second, "回写必须幂等")
        self.assertNotIn(",,", second)
        for line in second.splitlines():
            if line.startswith("(") or line.startswith("hold("):
                # tap 原本 2 个参数、hold 原本 3 个，补上手标记后各 +1
                expected = 3 if line.startswith("hold(") else 2
                actual = len([p for p in line[line.find("(")+1:line.rfind(")")].split(",") if p.strip()])
                self.assertEqual(actual, expected + 1, f"参数个数不对: {line}")
        for time, hand in re.findall(r"arctap\((\d+),([01])\)", second):
            self.assertIn(int(hand), (0, 1))

    def test_line_count_preserved(self) -> None:
        parsed = parse_aff(TINY_AFF)
        text = serialize_aff(parsed, self._hands(parsed))
        self.assertEqual(len(text.splitlines()), len(TINY_AFF.splitlines()))


class ConstraintTest(unittest.TestCase):
    """音弧颜色构成硬约束。"""

    def test_same_start_forces_arc_hand(self) -> None:
        """同刻**同点**起手 -> 必须与弧同手（需求里的"例外"之一）。

        弧从 lane1（x = 0.25, y = 0）起步，tap 也落在 lane1，手指点下去正好接住。
        """
        text = "-\narc(1000,2000,-0.25,-0.25,s,0,0,1,none,false);\n(1000,1,0);\n"
        parsed = parse_aff(text)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        tap_index = next(i for i, n in enumerate(parsed.chart.notes) if n.type == "tap")
        self.assertEqual(constraints.forced.get(tap_index), "R",
                         "弧色 1 = 右手，同刻同点的 tap 必须也用右手")
        result = split(parsed.chart.notes, Config(), parsed.chart.meta, constraints=constraints)
        self.assertEqual(result.hands[tap_index], "R")

    def test_two_same_color_arcs_count_as_one_finger(self) -> None:
        """同一时刻两条**同色**实体弧只算一根手指。

        否则"1 个键 + 2 条同色弧"会被数成 3 个触点、误抬成多指段，而多指段会
        整段无视"有弧时不得同色"这条硬约束（用户报告的 bug）。
        字段顺序：arc(t1,t2,x1,x2,easing,y1,y2,color,...)，x 直接用世界坐标。
        """
        text = ("-\n"
                "arc(1000,2000,-0.25,-0.25,s,0,0,0,none,false);\n"
                "arc(1000,2000,0.0,0.0,s,0,0,0,none,false);\n"
                "(1500,2,0);\n")
        parsed = parse_aff(text)
        notes = parsed.chart.notes
        config = Config()
        apply_arcaea_physics(config)
        constraints = build_arc_constraints(parsed, notes)
        scoring = Scoring(notes, config, parsed.chart.meta, constraints)
        tap = next(i for i, n in enumerate(notes) if n.type == "tap")
        self.assertNotIn(tap, scoring.multi_finger,
                         "两条同色弧不该把这一刻抬成多指段")
        self.assertNotIn("L", constraints.allowed_hands(tap),
                         "弧色 0 = 左手在追弧，区间内的键不得再用左手")

    def test_arctaps_on_opposite_color_arcs_keep_their_arc_hand(self) -> None:
        """多指段里，骑在实体弧上的天键必须**听弧色**，不能只按空间分半。

        两条弧的采样点都在左半边、且**右色弧还在左边** —— 只按空间会把两个天键
        都判成 L（用户报告的"本应分给不同手却给了同手"）。
        """
        text = ("-\n"
                "arc(1000,2000,-0.25,-0.25,s,0,0,1,none,false)[arctap(1200)];\n"
                "arc(1000,2000,0.0,0.0,s,0,0,0,none,false)[arctap(1200)];\n"
                "(1200,2,0);\n")
        parsed = parse_aff(text)
        notes = parsed.chart.notes
        config = Config()
        apply_arcaea_physics(config)
        constraints = build_arc_constraints(parsed, notes)
        scoring = Scoring(notes, config, parsed.chart.meta, constraints)
        arctaps = [i for i, n in enumerate(notes) if n.type == "arctap"]
        self.assertEqual(len(arctaps), 2, "两个 arctap 都该被解析成判定物件")
        self.assertTrue(all(i in scoring.multi_finger for i in arctaps),
                        "这一刻应当是多指段（本条用例的前提）")
        result = split(notes, config, parsed.chart.meta, constraints=constraints)
        hands = sorted(result.hands[i] for i in arctaps)
        self.assertEqual(hands, ["L", "R"],
                         "两个天键必须按各自弧色分到不同手（右色弧在左边也一样）")

    def test_same_start_other_place_is_other_hand(self) -> None:
        """同刻但**不在同一个点** -> 不能算"同一次落指"，应判成异手。

        弧从 lane1 起步，tap 在 lane3（两条轨道之外）。那根手指正在起弧，
        不可能同时在 lane3 上点一下 —— 所以 tap 必须换手。
        """
        text = "-\narc(1000,2000,-0.25,-0.25,s,0,0,1,none,false);\n(1000,3,0);\n"
        parsed = parse_aff(text)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        tap_index = next(i for i, n in enumerate(parsed.chart.notes) if n.type == "tap")
        self.assertIsNone(constraints.force_hand(tap_index),
                          "位置对不上就不该被『同刻同手』钉住")
        self.assertIn("R", constraints.forbidden_hands(tap_index),
                      "起弧那只手（右手）正在握弧，应被规则 3 排除")
        result = split(parsed.chart.notes, Config(), parsed.chart.meta, constraints=constraints)
        self.assertEqual(result.hands[tap_index], "L")

    def test_interval_forbids_arc_hand(self) -> None:
        """弧区间内的物件不得用弧的那只手。"""
        text = ("-\narc(0,5000,0.25,0.25,s,0,0,1,none,false);\n"
                "(1000,3,0);\n(2000,3,0);\n")
        parsed = parse_aff(text)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        inside = [i for i, n in enumerate(parsed.chart.notes) if n.type == "tap"]
        for index in inside:
            self.assertEqual(constraints.forbidden_hands(index), {"R"},
                             "弧色 1 = 右手，区间内不得再用右手")
        result = split(parsed.chart.notes, Config(), parsed.chart.meta, constraints=constraints)
        for index in inside:
            self.assertEqual(result.hands[index], "L")

    def test_trace_arc_imposes_nothing(self) -> None:
        """黑线没有颜色语义，不产生约束。"""
        text = "-\narc(0,5000,0.25,0.25,s,0,0,1,none,true);\n(1000,3,0);\n"
        parsed = parse_aff(text)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        self.assertEqual(constraints.forced, {})
        self.assertEqual(constraints.forbidden, {})

    def test_chord_exception_is_recorded(self) -> None:
        """同刻两键分别落在不同颜色的弧区间内 -> 无法同时满足，记为多押例外。"""
        text = ("-\narc(1000,1500,0.5,0.5,s,0,0,0,none,false);\n"
                "arc(1000,5000,0.75,0.75,s,0,0,1,none,false);\n"
                "(1000,2);\n(1000,3);\n")
        parsed = parse_aff(text)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        self.assertGreaterEqual(constraints.stats.get("chord_exceptions", 0), 1,
                                "两条不同颜色的弧同时锁住一个双押 -> 必须记为例外")
        self.assertEqual(constraints.forbidden, {}, "例外情况下不施加区间限手")
        # 例外情况下也要能出结果（不许崩、不许出现未标记）
        result = split(parsed.chart.notes, Config(), parsed.chart.meta, constraints=constraints)
        self.assertEqual(len(result.hands), len(parsed.chart.notes))
        self.assertTrue(all(result.hands))
        self.assertNotEqual(result.hands[0], result.hands[1], "双押仍应分给两只手")

    def test_solver_respects_arc_constraints(self) -> None:
        """求解器必须**真的**遵守音弧硬约束（不是"罚得比较重"）。

        这一段以前跑在真实谱面 ``breachoffaith_2`` 上（630 键人工基线）。
        那张谱已从仓库移除：它覆盖不全面、且每次跑都要十几秒，
        不适合当常规回归。这里用一段**自足的合成谱面**保留同样三条断言 ——
        关键回归是"求解结果满足约束生成器给出的约束"，
        而不依赖任何外部谱面文件。
        """
        text = "\n".join([
            "-",
            "timing(0,172.00,4.00);",
            # 实体弧：0..2000 之间被左手按住（color 0 = L）
            "arc(0,2000,0.25,0.25,s,0,1,0,none,false);",
            # 同刻同点：落在弧头同一点的键必须同手（= L）
            "(0,2);",
            # 区间异手：弧按住期间落下的键不得用 L
            "(500,1);",
            "(900,3);",
            "(1300,2);",
            "(1700,4);",
            # 弧结束之后可以自由
            "(2500,1);",
            "(2500,4);",
            # 再来一条右手的弧，制造"区间异手"的另一侧
            "arc(3000,5000,0.75,0.75,s,0,1,1,none,false);",
            "(3400,2);",
            "(3800,1);",
            "(4200,4);",
            "(5400,2);",
            "(5400,3);",
        ]) + "\n"
        parsed = parse_aff(text)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        config = Config()
        apply_arcaea_physics(config)
        result = split(parsed.chart.notes, config, parsed.chart.meta,
                       constraints=constraints)

        hand_of = {"L": 0, "R": 1}
        forced_checked = forbidden_checked = 0
        for index, note in enumerate(parsed.chart.notes):
            if note.type != "tap":
                continue
            forced = constraints.force_hand(index)
            if forced:
                forced_checked += 1
                self.assertEqual(hand_of[result.hands[index]], hand_of[forced],
                                 f"第 {note.line} 行没有遵守同刻同手")
            blocked = constraints.forbidden_hands(index)
            if blocked:
                forbidden_checked += 1
                self.assertNotIn(result.hands[index], blocked,
                                 f"第 {note.line} 行违反了区间异手")
        # 这段谱面必须真的**碰到**约束，否则测试会"空过"
        self.assertGreater(forced_checked, 0, "样本里没有触发同刻同手约束")
        self.assertGreater(forbidden_checked, 0, "样本里没有触发区间异手约束")

        # 地面多押必须分开两只手
        by_time: dict[int, list[int]] = {}
        for index, note in enumerate(parsed.chart.notes):
            if note.type in ("tap", "hold"):
                by_time.setdefault(note.time, []).append(index)
        chords = 0
        for time, indexes in by_time.items():
            if len(indexes) > 1:
                chords += 1
                hands = {result.hands[i] for i in indexes}
                self.assertEqual(len(hands), len(indexes),
                                 f"t={time} 的地面多押没有分开两只手")
        self.assertGreater(chords, 0, "样本里没有地面双押")

    def test_no_unassigned_hands_on_arc_heavy_chart(self) -> None:
        """再密的弧+键组合也不许出现"没有手"的物件。"""
        text = "-\n" + "\n".join(
            f"arc({i * 200},{i * 200 + 350},0.25,0.75,s,0,1,{i % 2},none,false);"
            f"\n({i * 200 + 100},{1 + i % 4});"
            for i in range(24)) + "\n"
        parsed = parse_aff(text)
        config = Config()
        apply_arcaea_physics(config)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        result = split(parsed.chart.notes, config, parsed.chart.meta,
                       constraints=constraints)
        self.assertEqual(result.counts()["none"], 0, "有空手标记")
        self.assertEqual(len(result.hands), len(parsed.chart.notes))


class SpatialPhaseTest(unittest.TestCase):
    """空间规则必须在"中心附近"也生效，否则左右会整段反过来。

    曾经踩的坑：``neutral_band`` / ``strong_bias`` 是按**屏幕像素**标定的
    （"离中心多近算中性"），而 Arcaea 的坐标刻度是 1072 px/单位。没做换算时
    "15 像素的中性带"变成"15 个世界单位"——比整个判定面还宽两倍，于是
    空间代价处处为 0，严格交替的两段出现"相位任选"，整段左右被反过来。
    """

    def _run(self, text: str):
        parsed = parse_aff(text)
        config = Config()
        apply_arcaea_physics(config)
        result = split(parsed.chart.notes, config, parsed.chart.meta)
        return parsed, result

    def test_two_lane_chord_left_is_left(self) -> None:
        """只有两个 tap 的双押：靠左那个应该是 0（左手），靠右是 1。"""
        text = "-\n(0,2);\n(0,3);\n"
        parsed, result = self._run(text)
        by_lane = {note.params["lane"]: result.side_by_note[note] for note in parsed.chart.notes}
        self.assertEqual(by_lane[2.0], "L", "2 轨（x=0.50，在中心左侧）应为左手")
        self.assertEqual(by_lane[3.0], "R", "3 轨（x=0.75，在中心右侧）应为右手")

    def test_run_phase_follows_positions(self) -> None:
        """一段交互：相位应由左右位置决定，不能整段反过来。

        用"2 轨 + 3 轨"的来回交互（同一 lane 会重复出现，所以按时间取）。
        """
        times = [(0, 2), (250, 3), (375, 2), (500, 3), (625, 2), (750, 3)]
        text = "-\n" + "".join(f"({t},{lane});\n" for t, lane in times)
        parsed, result = self._run(text)
        hands = {note.time: result.side_by_note[note] for note in parsed.chart.notes}
        for time, lane in times:
            want = "L" if lane == 2 else "R"
            self.assertEqual(hands[time], want,
                             f"{lane} 轨（x={lane_to_x(lane):.2f}）在 t={time} 应为 {want}")

    def test_neutral_band_is_scaled_to_coordinates(self) -> None:
        """中性带必须按坐标刻度换算：像素口径 15px != 15 个世界单位。"""
        parsed = parse_aff("-\n(0,2);\n")
        config = Config()
        apply_arcaea_physics(config)
        scoring = Scoring(parsed.chart.notes, config, parsed.chart.meta)
        # 造一个"离局部中心半个轨道"的键：偏置按像素算要明显超过 15px 的中性带
        from spliter.chart import Note
        # 时间近一点，让"局部中心"确实落在两键之间（局部中心窗是 ~300ms 衰减）
        notes = [Note(time=0, x=lane_to_x(2), type="tap"),
                 Note(time=200, x=lane_to_x(3), type="tap")]
        scoring = Scoring(notes, config, {})
        bias_px = abs(scoring.bias[0]) * config.physics.pixel_per_unit_x
        self.assertGreater(bias_px, 15.0,
                           f"该键的偏置按像素算应超过中性带（实际 {bias_px:.1f}px）")
        # 中性带换算后应该远小于偏置（而不是大得把一切归零）
        scaled_band = config.space.neutral_band / config.physics.pixel_per_unit_x
        self.assertLess(scaled_band, abs(scoring.bias[0]))
        self.assertNotAlmostEqual(scoring.bias_cost(0, "L"), scoring.bias_cost(0, "R"),
                                  places=6, msg="中心附近的键也必须能区分左右")


class ArctapAndHoldConstraintTest(unittest.TestCase):
    """arctap 必须参与音弧约束；长条占手期间必须换手；长条尾指处必须同手。

    这三条都是"规则性"的：它们不是权重问题，而是"根本没参与计算"。
    """

    def _run(self, text: str):
        parsed = parse_aff(text)
        config = Config()
        apply_arcaea_physics(config)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        result = split(parsed.chart.notes, config, parsed.chart.meta, constraints=constraints)
        return parsed, constraints, result

    def test_arctap_obeys_arc_interval(self) -> None:
        """音弧区间内的 arctap 不得用该弧的手（早期版本 arctap 被整个漏掉）。"""
        text = ("-\narc(0,5000,0.5,0.5,s,0,0,1,none,false);\n"
                "arc(1000,1000,0.5,0.5,s,1,1,1,none,true)[arctap(1000)];\n")
        parsed, constraints, result = self._run(text)
        arctap_index = next(i for i, n in enumerate(parsed.chart.notes) if n.type == "arctap")
        self.assertEqual(constraints.forbidden_hands(arctap_index), {"R"},
                         "落在 color=1 弧区间内的 arctap 必须被禁用右手")
        self.assertEqual(result.hands[arctap_index], "L")

    def test_arctap_same_start_shares_arc_hand(self) -> None:
        """同刻同点起步的 arctap 应与弧同手（y 也要一样）。"""
        text = ("-\narc(1000,2000,0.5,0.5,s,0,0,0,none,false);\n"
                "arc(1000,1000,0.5,0.5,s,0,0,0,none,true)[arctap(1000)];\n")
        parsed, constraints, result = self._run(text)
        arctap_index = next(i for i, n in enumerate(parsed.chart.notes) if n.type == "arctap")
        self.assertEqual(constraints.force_hand(arctap_index), "L")
        self.assertEqual(result.hands[arctap_index], "L")

    def test_arctap_above_arc_start_is_other_hand(self) -> None:
        """同刻、同 x、但高一个天空高度 -> 不是同一个点，必须换手。

        地面长条/弧的尾指停在 (x, 0)，正上方 (x, 1) 的天空键是另一个点；
        只比 x 会把它误判成"同一根手指能同时按下"（实谱 34883 处的 hold +
        35232 处 arctap 就是这样，曾被算成 16 万的假代价）。
        """
        text = ("-\narc(1000,2000,0.5,0.5,s,0,0,0,none,false);\n"
                "arc(1000,1000,0.5,0.5,s,1,1,0,none,true)[arctap(1000)];\n")
        parsed, constraints, result = self._run(text)
        arctap_index = next(i for i, n in enumerate(parsed.chart.notes) if n.type == "arctap")
        self.assertNotEqual(constraints.force_hand(arctap_index), "L",
                            "天空的 arctap 与地面的弧起点不是同一个位置")
        self.assertEqual(result.hands[arctap_index], "R")

    def test_hold_occupies_hand_during_its_span(self) -> None:
        """长条按住期间，同一个手不能去点别的键。"""
        text = "-\nhold(0,3000,2);\n(1000,3);\n"
        parsed, _, result = self._run(text)
        hold = next(n for n in parsed.chart.notes if n.type == "hold")
        tap = next(n for n in parsed.chart.notes if n.type == "tap")
        self.assertNotEqual(result.side_by_note[hold], result.side_by_note[tap],
                            "长条区间内的键必须换手")

    def test_hold_tail_shares_hand(self) -> None:
        """长条尾指处、同一位置的键应与长条同手。

        前面先放两个键把两只手"摆到位"再测 —— 否则单条长条从起手位出发的
        代价会盖过一切，测到的是起手位参数而不是这条规则。
        """
        # 尾指键放在长条**结束那一刻**（同一位置）。如果放在结束之前 1ms，
        # "长条区间内必须异手"会优先于"尾指同手"，那是另一条规则。
        text = ("-\n(0,2);\n(50,3);\n"
                "hold(1000,2000,2);\n(2000,2);\n")
        parsed, _, result = self._run(text)
        notes = parsed.chart.notes
        hold = next(n for n in notes if n.type == "hold")
        # 尾指处的那个键：时间接近 hold 结束、位置相同
        tail = min((n for n in notes if n.type == "tap" and n.time >= 1900),
                   key=lambda n: abs(n.time - 2000))
        self.assertAlmostEqual(tail.x, hold.x)
        # 规则只要求"同手"，具体给哪只手由空间/节奏决定
        self.assertEqual(result.side_by_note[hold], result.side_by_note[tail],
                         f"尾指位置的键应与长条同手，实际 hold={result.side_by_note[hold]} "
                         f"tail={result.side_by_note[tail]}")

    def test_hold_tail_different_position_may_switch(self) -> None:
        """尾指那一刻但位置不同：不强制同手（手可以从尾巴挪过去）。"""
        text = "-\nhold(0,1000,1);\n(1000,4);\n"
        parsed, _, result = self._run(text)
        self.assertTrue(all(result.hands), "必须给出完整分配")


class PointArcAndMultiFingerTest(unittest.TestCase):
    """双指 vs 多指：弧的两种处理方式（用户口径）。

    两个关键词：**多指**、**不分长短**。

    - 只有双指时（普通配置），所有实体弧无论点弧还是长弧**都占手**，
      区间内其它物件必须换手 —— 这条一直执行得很好，别动；
    - 多指段里，弧拿它**那一刻的采样点**去参与"同 y 从左到右分左右"的排序，
      自己的颜色（手）已经定好，不再去压别的键。**长度完全不参与判断**。
    """

    def _run(self, text: str):
        parsed = parse_aff(text)
        config = Config()
        apply_arcaea_physics(config)
        constraints = build_arc_constraints(parsed, parsed.chart.notes)
        result = split(parsed.chart.notes, config, parsed.chart.meta,
                       constraints=constraints)
        return parsed, constraints, result

    def test_two_finger_point_arc_still_holds_hand(self) -> None:
        """双指配置里，短定点弧照样整段占手（长度不影响判断）。"""
        text = ("-\n"
                "arc(0,60,1,1);\n(60,4);\n"                 # 先把两只手摆到位
                "arc(1000,1080,0.00,0.00,s,1.00,1.00,0,none,false);\n"
                "(1040,3);\n")
        parsed, constraints, result = self._run(text)
        tap = max(i for i, n in enumerate(parsed.chart.notes) if n.type == "tap")
        self.assertEqual(constraints.forbidden_hands(tap), {"L"},
                         "双指时点弧（色 0 = 左手）区间内的键必须换手")
        self.assertEqual(result.hands[tap], "R")

    def test_multi_finger_arc_joins_same_height_ordering(self) -> None:
        """多指段：弧拿采样点参与同 y 排序，地面键不再被它压。

        构造 87964 那一类配置：天上一个定点弧（色 0 = 左）在 x=0.00、
        一条黑线上挂 arctap 在 x=0.75，地面是双押。多指段里
        y=1 的两个触点按 x 分左右 -> 弧在左、arctap 在右；地面双押按 x 分。
        """
        text = ("-\n"
                "arc(1000,1080,0.00,0.00,s,1.00,1.00,0,none,false);\n"
                "(1000,1);\n(1000,2);\n"
                "arc(0,2000,0.75,0.75,s,1,1,1,none,true)[arctap(1000)];\n")
        parsed, constraints, result = self._run(text)
        self.assertTrue(result.scoring.multi_finger_times,
                        "地面双押 + 天键 + 弧采样点 = 4 个触点，应判为多指段")
        self.assertEqual(result.hands, ["L", "R", "R"],
                         "同 y 的按 x 分左右：天上弧在左、arctap 在右；地面双押 L/R")
        self.assertFalse([w for w in result.warnings if "物理上无法" in w],
                         "多指配置不该被报成『三押按不下』")

    def test_multi_finger_arc_sample_point_is_not_a_blocker(self) -> None:
        """多指段里弧只按采样点排序，不再"区间异手"压住别的键。"""
        text = ("-\n"
                "arc(1000,3000,0.00,0.00,s,1.00,1.00,0,none,false);\n"
                "(1000,1);\n(1000,2);\n(1000,3);\n")
        parsed, constraints, result = self._run(text)
        self.assertTrue(result.scoring.multi_finger_times)
        # 三个地面键（x = -0.25 / 0.25 / 0.75）+ 天上的弧采样点：地面那一层
        # 按 x 分左右两半（3 个 -> 左半 2 个 L、右半 1 个 R），弧不参与地面层
        self.assertEqual(result.hands[:3], ["L", "L", "R"])

    def test_one_ms_offset_chord_is_one_moment(self) -> None:
        """1ms 错位和弦（tap + 晚 1ms 的 hold）要按**同一个时刻**分左右。

        谱面里常见这种写法（例如 86004 的 tap 与 86005 的 hold）；按精确时间戳
        拆开会让两个键各自"孤立"，既看不出多指、也分不出左右。
        """
        text = ("-\n"
                "arc(1000,1080,1.00,1.00,s,1.00,1.00,1,none,false);\n"
                "(1000,3);\n"
                "hold(1001,1500,4);\n")
        parsed, _, result = self._run(text)
        self.assertEqual(result.hands[:2], ["L", "R"],
                         f"1ms 错位的一对键应按同一时刻分左右：{result.hands}")

    def test_dense_same_lane_sixteenths_split(self) -> None:
        """同键位 16 分连按（非 bullet）是硬抗里最难受的一种：必须拆成两手交互。

        用户的参考谱在 31432~32185 那段就是 LRL / LRL / LRL；早期版本因为
        `jack_time = 0` 让同键连按免费，整段被判成一只手连点。
        """
        text = "-\n" + "\n".join(f"({1000 + i * 75},2);" for i in range(6))
        parsed, _, result = self._run(text)
        switches = sum(1 for a, b in zip(result.hands, result.hands[1:]) if a != b)
        self.assertGreaterEqual(switches, 3,
                                f"同键位 16 分应拆成两手交互，实际 {result.hands}")
        self.assertTrue(all(h in ("L", "R") for h in result.hands))

    def test_isolated_same_key_bullet_stays_light(self) -> None:
        """孤立的一次同键连按是 bullet：按 jack_time 轻罚，不必拆手。"""
        text = "-\n(0,2);\n(300,2);\n(2000,4);\n(2300,4);\n(5000,1);\n"
        parsed, _, result = self._run(text)
        self.assertEqual(result.hands[0], result.hands[1],
                         f"孤立的同键连按（300ms 间隔）不该被判成交互：{result.hands}")
        self.assertEqual(result.hands[2], result.hands[3],
                         f"孤立的同键连按不该被判成交互：{result.hands}")

    def test_output_never_violates_two_finger_constraints(self) -> None:
        """双指配置里输出必须满足硬约束（不许再出现静默非法手标记）。"""
        text = ("-\n"
                "arc(1000,1100,0.00,0.00,s,1.00,1.00,0,none,false);\n"
                "(1000,1);\n(2000,2);\n"
                "arc(0,3000,0.75,0.75,s,1,1,1,none,false);\n")
        parsed, constraints, result = self._run(text)
        for index, hand in enumerate(result.hands):
            self.assertIn(hand, ("L", "R"), "每个物件都要有手标记")
            self.assertIn(hand, constraints.allowed_hands(index),
                          f"第 {parsed.chart.notes[index].line} 行发出了硬约束不允许的手标记")
        self.assertEqual(result.scoring.unresolved, [])


class ReachUnitTest(unittest.TestCase):
    """max_reach 是按像素标定的，比较"同一时刻单手跨度"时必须换算。"""

    def test_wide_chord_needs_two_hands(self) -> None:
        """相距 2 个世界单位（2144px）的双押：必须分两只手，且各自只按一个键。"""
        from spliter.chart import Note
        notes = [Note(time=0, x=-1.0, type="tap"), Note(time=0, x=1.0, type="tap")]
        config = Config()
        apply_arcaea_physics(config)
        scoring = Scoring(notes, config, {})
        self.assertEqual(scoring.px_to_units(scoring.physics.max_reach),
                         scoring.physics.max_reach / PIXELS_PER_UNIT_X)
        self.assertTrue(scoring.can_fill_group([0, 1]), "双押总该能分给两只手")

    def test_one_hand_cannot_span_too_far(self) -> None:
        """三押里"一只手要按两个相距 804px 的键"是不允许的（max_reach = 620px）。"""
        from spliter.chart import Note
        notes = [Note(time=0, x=-1.5, type="tap"),
                 Note(time=0, x=0.0, type="tap"),
                 Note(time=0, x=1.5, type="tap")]
        config = Config()
        apply_arcaea_physics(config)
        scoring = Scoring(notes, config, {})
        self.assertFalse(scoring.can_fill_group([0, 1, 2]),
                         "-1.5 与 +1.5 相距 1.5 单位 = 804px，一只手跨不过去")
        # 这种三押本身无解：退化分配只能"挑跨度最小的那一对同手"，
        # 同时必须仍是一个合法分配（一键一手/两手都要用上）。
        from spliter.split import _realizations
        for assignment in _realizations(scoring, [0, 1, 2]):
            self.assertEqual(len(assignment), 3)
            self.assertEqual(len(set(assignment)), 2, "退化分配也要两只手都用")
            same = {}
            for slot, hand in enumerate(assignment):
                same.setdefault(hand, []).append(notes[slot].x)
            widest = max((max(v) - min(v)) for v in same.values() if len(v) > 1)
            # 最好的可能：把邻近的一对放同手（跨度 1.5），而不是把两端放一起（跨度 3.0）
            self.assertLessEqual(widest, 1.5 + 1e-9,
                                 f"退化分配应选跨度最小的一对，实际 {widest:.2f} 单位")


class CalibrationTest(unittest.TestCase):
    """屏幕像素刻度：两个轴不一样，位移必须分轴换算再勾股合成。"""

    def test_scale_is_anisotropic(self) -> None:
        """横向刻度明显大于纵向（判定面是斜视的 3D 平面）。"""
        self.assertGreater(PIXELS_PER_UNIT_X, PIXELS_PER_UNIT_Y)
        self.assertNotAlmostEqual(PIXELS_PER_UNIT_X, PIXELS_PER_UNIT_Y, places=3)

    def test_scale_matches_measured_reference(self) -> None:
        """用游戏内截图实测的参照物反推：lane1~lane4 约 804px、y 0~1 约 326px。"""
        lane_span_units = (lane_to_x(4) - lane_to_x(1))     # = 1.5 个单位
        self.assertAlmostEqual(lane_span_units, 1.5)
        self.assertAlmostEqual(lane_span_units * PIXELS_PER_UNIT_X / 804.0, 1.0, places=2)
        # 相邻轨道相差 1 个轨道宽（0.5 单位）= 实测 268px
        self.assertAlmostEqual(0.5 * PIXELS_PER_UNIT_X, 268.0, places=2)
        # 4 条轨道的左右边界就是官方给 arctap 的判定区间
        self.assertAlmostEqual(lane_to_x(1) - 0.25, -0.5)
        self.assertAlmostEqual(lane_to_x(4) + 0.25, 1.5)
        self.assertAlmostEqual(1.0 * PIXELS_PER_UNIT_Y / 326.0, 1.0, places=2)

    def test_apply_sets_scales(self) -> None:
        config = Config()
        apply_arcaea_physics(config)
        self.assertEqual(config.physics.pixel_per_unit_x, PIXELS_PER_UNIT_X)
        self.assertEqual(config.physics.pixel_per_unit_y, PIXELS_PER_UNIT_Y)

    def test_distance_uses_both_axes(self) -> None:
        """同样的 |Δx|，加上纵向位移后距离必须变大（勾股）。"""
        from spliter.aff import parse_aff as _parse
        from spliter.split import Scoring

        chart = _parse("-\narc(0,1000,0.5,0.5,s,0,0,0,none,false);\n"
                       "arc(0,1000,0.5,0.5,s,1,1,0,none,true)[arctap(0)];\n")
        notes = chart.chart.notes
        self.assertEqual(len(notes), 1, "这条黑线上应该只有 1 个 arctap")
        self.assertAlmostEqual(notes[0].y, 1.0, msg="arctap 在 y=1")

        config = Config()
        apply_arcaea_physics(config)
        scoring = Scoring(notes, config, chart.chart.meta)
        # 人工造一个"同一 x、不同 y"的对比
        from spliter.chart import Note
        other = Note(time=100, x=notes[0].x, y=0.0, type="tap")
        notes.append(other)
        scoring = Scoring(notes, config, chart.chart.meta)
        distance = scoring.screen_distance(0, 1)
        self.assertAlmostEqual(distance, PIXELS_PER_UNIT_Y, delta=1.0,
                               msg="只差 y 时，距离应等于纵向刻度 × |Δy|")
        self.assertGreater(distance, 0.0, "纵向位移必须计入距离")


class SixLaneScaleTest(unittest.TestCase):
    """六轨模式（``enwidencamera`` / ``enwidenlanes``）的刻度换算。

    依据（用户口径）：0 轨左边界 = 原 1 轨左边界、5 轨右边界 = 原 4 轨右边界、
    新的 ``y = 1.61`` 对应原来的 ``y = 1.0``。也就是同一块屏幕要装下
    3.0 个弧单位（原来 2.0 个），于是 x 刻度乘 2/3、位移**缩小 1.5 倍**。
    """

    #: 6k 段：[1000, 2000)；前后都是标准四轨。
    SIX_LANE_AFF = """-
timing(0,172.00,4.00);
(0,1);
scenecontrol(1000,enwidencamera,1333.00,1);
scenecontrol(1000,enwidenlanes,1333.00,1);
(1500,1);
scenecontrol(2000,enwidencamera,0.00,0);
scenecontrol(2000,enwidenlanes,0.00,0);
(2500,4);
"""

    def test_lane_coords_are_unchanged_in_six_lane(self) -> None:
        """0/5 轨的坐标是 lane_to_x 的线性外推，**不需要**额外的坐标变换。"""
        self.assertAlmostEqual(lane_to_x(0), -0.75)
        self.assertAlmostEqual(lane_to_x(5), 1.75)
        self.assertAlmostEqual(lane_to_x(2.5), 0.5)      # 中线仍在 0.5

    def test_scenecontrol_parsed(self) -> None:
        chart = parse_aff(self.SIX_LANE_AFF)
        kinds = {(e.kind, e.enable) for e in chart.scenecontrols}
        self.assertEqual(kinds, {("camera", True), ("lanes", True),
                                 ("camera", False), ("lanes", False)})

    def test_schedule_covers_steady_state_only(self) -> None:
        schedule = parse_aff(self.SIX_LANE_AFF).scale_schedule()
        self.assertEqual(len(schedule.segments), 1, "四段事件应折成一个区间")
        lo, hi = schedule.segments[0].start, schedule.segments[0].end
        self.assertEqual((lo, hi), (1000, 2000))
        # 区间内缩小、区间外恢复
        self.assertLess(schedule.at(1500)[0], 1.0)
        self.assertEqual(schedule.at(999), (1.0, 1.0))
        self.assertEqual(schedule.at(2000), (1.0, 1.0))

    def test_scale_factors(self) -> None:
        schedule = parse_aff(self.SIX_LANE_AFF).scale_schedule()
        scale_x, scale_y = schedule.at(1500)
        self.assertAlmostEqual(scale_x, 2.0 / 3.0)
        self.assertAlmostEqual(scale_y, 1.0 / 1.61)

    def test_span_shrinks_by_1_5(self) -> None:
        """同样的坐标跨度，六轨下算出的像素距离应缩小 1.5 倍。"""
        chart = parse_aff(self.SIX_LANE_AFF)
        config = Config()
        apply_arcaea_physics(config)
        config.physics.scale_schedule = chart.scale_schedule()
        # lane0 -> lane5 = 2.5 个单位
        span = lane_to_x(5) - lane_to_x(0)
        self.assertAlmostEqual(span, 2.5)
        outside = screen_span(span, 0.0, config.physics, 500)
        inside = screen_span(span, 0.0, config.physics, 1500)
        self.assertAlmostEqual(outside, 2.5 * PIXELS_PER_UNIT_X)
        self.assertAlmostEqual(inside, 2.5 * PIXELS_PER_UNIT_X * 2.0 / 3.0)
        self.assertAlmostEqual(outside / inside, 1.5, places=6)

    def test_y_1_61_equals_old_y_1_0(self) -> None:
        """``y = 1.61`` 在六轨刻度下应等于原 ``y = 1.0`` 的像素高度。"""
        chart = parse_aff(self.SIX_LANE_AFF)
        config = Config()
        apply_arcaea_physics(config)
        config.physics.scale_schedule = chart.scale_schedule()
        self.assertAlmostEqual(screen_span(0.0, 1.61, config.physics, 1500),
                               PIXELS_PER_UNIT_Y, delta=1.0)

    def test_standard_chart_has_no_schedule(self) -> None:
        """没有 enwiden 事件的谱面不产生任何区间（行为与以前完全一致）。"""
        schedule = parse_aff(TINY_AFF).scale_schedule()
        self.assertFalse(schedule)
        self.assertEqual(schedule.at(0), (1.0, 1.0))
        config = Config()
        apply_arcaea_physics(config)
        self.assertEqual(screen_span(1.0, 0.0, config.physics, 0),
                         PIXELS_PER_UNIT_X)

    def test_camera_and_lanes_are_independent(self) -> None:
        """两个事件语义独立：只开 camera 时 x 刻度不该变。"""
        chart = parse_aff("-\nscenecontrol(0,enwidencamera,0.00,1);\n(10000,1);\n")
        schedule = chart.scale_schedule()
        scale_x, scale_y = schedule.at(5000)
        self.assertEqual(scale_x, 1.0)
        self.assertAlmostEqual(scale_y, 1.0 / 1.61)

    def test_lanes_only_does_not_change_y(self) -> None:
        """反过来：只开 lanes 时 y 刻度不该变。"""
        chart = parse_aff("-\nscenecontrol(0,enwidenlanes,0.00,1);\n(10000,4);\n")
        scale_x, scale_y = chart.scale_schedule().at(5000)
        self.assertAlmostEqual(scale_x, 2.0 / 3.0)
        self.assertEqual(scale_y, 1.0)

    def test_unclosed_six_lane_runs_to_end(self) -> None:
        """开了但没关：延到谱面结束。"""
        chart = parse_aff("-\n(0,1);\nscenecontrol(100,enwidenlanes,0.00,1);\n(5000,4);\n")
        schedule = chart.scale_schedule()
        self.assertEqual(len(schedule.segments), 1)
        self.assertEqual(schedule.segments[0].start, 100)
        self.assertGreaterEqual(schedule.segments[0].end, 5000)


class JackScaleTest(unittest.TestCase):
    """jack / bullet 的死区与宽度判定要跟着六轨刻度走。"""

    def test_dead_zone_uses_time_aware_scale(self) -> None:
        chart = parse_aff(SixLaneScaleTest.SIX_LANE_AFF)
        config = Config()
        apply_arcaea_physics(config)
        config.physics.scale_schedule = chart.scale_schedule()
        from spliter.chart import Note as N
        # 同一对键（相隔 0.5 单位），放在 4k 段 / 6k 段里比
        four = [N(time=500, x=0.25, type="tap"), N(time=560, x=0.75, type="tap")]
        six = [N(time=1500, x=0.25, type="tap"), N(time=1560, x=0.75, type="tap")]
        d4 = Scoring(four, config, {}).screen_distance(0, 1)
        d6 = Scoring(six, config, {}).screen_distance(0, 1)
        self.assertAlmostEqual(d4 / d6, 1.5, places=6)


class ReportUnitsTest(unittest.TestCase):
    """报告口径：位移一律用"轨道"讲，且地面+天空合并统计。

    轨道口径的依据：**1 轨道 = 0.5 个谱面坐标 = 相邻轨道间距**。
    像素对玩家没有体感、还随设备分辨率变化，所以只在代码与标定处保留。
    """

    def _config(self):
        config = Config()
        apply_arcaea_physics(config)
        return config

    def test_lane_unit_matches_aff_constant(self) -> None:
        """report 里重复定义的 LANE_UNIT 必须与 aff.LANE_STEP 一致。"""
        from spliter.aff import LANE_STEP
        from spliter.report import LANE_UNIT
        self.assertEqual(LANE_UNIT, LANE_STEP)

    def test_one_lane_is_268px(self) -> None:
        from spliter.report import lanes
        physics = self._config().physics
        self.assertAlmostEqual(lanes(268.0, physics), 1.0, places=9)
        self.assertAlmostEqual(lanes(536.0, physics), 2.0, places=9)

    def test_adjacent_lane_reads_as_one_lane(self) -> None:
        """相邻轨道（Δx = 0.5）在报告里必须显示成 1.00 轨道。"""
        from spliter.report import lanes
        physics = self._config().physics
        self.assertAlmostEqual(lanes(0.5 * physics.pixel_per_unit_x, physics),
                               1.0, places=9)

    def test_full_height_in_lanes(self) -> None:
        """满高 = 326px，换算成轨道是 1.22（横 4 轨 ≈ 纵 1.22 轨）。"""
        from spliter.report import FULL_HEIGHT_IN_LANES
        self.assertAlmostEqual(FULL_HEIGHT_IN_LANES, 326.0 / 268.0, places=9)

    def test_describe_splits_horizontal_and_vertical(self) -> None:
        """describe 必须把横向/纵向**分开**报，不能只给勾股值。

        勾股值会把两个分量混起来 —— "跨 1.57 轨道"看不出是横向还是纵向。
        """
        from spliter.report import MoveRecord
        physics = self._config().physics
        move = MoveRecord(from_line=1, to_line=2, hand="R", delta_t=87,
                          delta_x=422.0, speed=4851.0,
                          from_x=0.75, to_x=1.25, from_y=1.0, to_y=0.0)
        text = move.describe(physics)
        self.assertIn("横向 1.00 轨道", text)   # Δx=0.5 -> 1 轨道
        self.assertIn("纵向 1.00", text)        # Δy=1.0 -> 满高
        self.assertNotIn("px", text, "给玩家看的行里不该出现 px")
        self.assertIn("轨道/秒", text)

    def test_describe_pure_vertical(self) -> None:
        """纯纵移（同 x）不能报成"横向 0 轨道"，要只报纵向。"""
        from spliter.report import MoveRecord
        physics = self._config().physics
        move = MoveRecord(from_line=1, to_line=2, hand="L", delta_t=200,
                          delta_x=326.0, speed=1630.0,
                          from_x=0.25, to_x=0.25, from_y=0.0, to_y=1.0)
        text = move.describe(physics)
        self.assertIn("纵向 1.00", text)
        self.assertNotIn("横向", text)

    def test_stats_cover_all_notes_not_just_ground(self) -> None:
        """analyze 必须把 arctap 一起算进去 —— 它往往才是最吃手速的部分。

        构造一份"地面很轻松、天空很快"的谱面：如果统计漏掉天空，
        最大手速就会明显偏小。
        """
        from spliter.report import analyze
        config = self._config()
        # 地面：慢速小幅移动（相邻轨、间隔大）
        ground = [
            Note(time=0, x=0.25, y=0.0, type="tap", side="L"),
            Note(time=600, x=0.75, y=0.0, type="tap", side="R"),
        ]
        # 天空：同一只手在 50ms 内横跨 2 轨道（很吃手速）
        sky = [
            Note(time=1000, x=-0.25, y=1.0, type="arctap", side="L"),
            Note(time=1050, x=0.75, y=0.0, type="arctap", side="L"),
        ]
        only_ground = analyze(ground, config)
        everything = analyze([*ground, *sky], config)
        self.assertGreater(everything.max_speed_seen, only_ground.max_speed_seen,
                           "合并统计后的最大手速应当更大（天空被计入）")
        self.assertEqual(everything.total, 4, "物件数必须是地面+天空")
        self.assertEqual(only_ground.total, 2)

    def test_lane_speed_fields_are_populated(self) -> None:
        """轨道口径的字段必须真的被填上（不能恒为 0）。"""
        from spliter.report import analyze
        config = self._config()
        notes = [
            Note(time=0, x=0.25, y=0.0, type="tap", side="L"),
            Note(time=100, x=1.25, y=0.0, type="tap", side="L"),
        ]
        stats = analyze(notes, config)
        self.assertGreater(stats.max_lane_speed, 0.0)
        self.assertGreater(stats.max_lane_speed_limit, 0.0)
        self.assertAlmostEqual(stats.max_move_lanes, 2.0, places=2)  # Δx=1.0 -> 2 轨道

    def test_report_units_follow_six_lane_scale(self) -> None:
        """六轨段落的像素刻度更小，同样坐标跨度换算出的轨道数**不变**。

        这是关键性质：轨道是谱面上的几何单位，不该随像素刻度变化 ——
        否则报告在 6k 段落里会给出误导性的数字。
        """
        from spliter.report import lanes
        chart = parse_aff(SixLaneScaleTest.SIX_LANE_AFF)
        config = self._config()
        config.physics.scale_schedule = chart.scale_schedule()
        from spliter.split import scale_at
        # 4k 段：2.0 坐标 = 4 轨道
        px_4k = scale_at(config.physics, 500)[0]
        self.assertAlmostEqual(lanes(2.0 * px_4k, config.physics, px_4k), 4.0, places=6)
        # 6k 段：像素刻度是 2/3（更小），但**同样 2.0 坐标仍然 = 4 轨道** ——
        # 轨道是谱面几何单位，不该随像素刻度变化。
        px_6k = scale_at(config.physics, 1500)[0]
        self.assertLess(px_6k, px_4k)
        self.assertAlmostEqual(lanes(2.0 * px_6k, config.physics, px_6k), 4.0, places=6)
        # 反面：如果拿 4k 刻度去换算 6k 的像素距离，就会算少（这正是要避免的错法）
        self.assertAlmostEqual(lanes(2.0 * px_6k, config.physics, px_4k),
                               4.0 * 2.0 / 3.0, places=6)


    def test_explain_prints_readable_coordinates(self) -> None:
        """``--explain`` 的 x 不能用 round() —— 坐标只在 0~2 之间，取整全变 0/1。"""
        from spliter.report import explain
        config = self._config()
        parsed = parse_aff(TINY_AFF)
        result = split(parsed.chart.notes, config, parsed.chart.meta)
        rows = explain(result, limit=None)
        self.assertTrue(rows, "应该有可解释的物件")
        text = rows[0].describe()
        self.assertIn("y=", text, "应同时给出 y，才能区分地面/天空")
        # 至少有一条 x 不是 0/1 这种取整值
        self.assertTrue(
            any(abs(r.x) not in (0.0, 1.0) for r in rows),
            "坐标不该全被取整成 0/1")
        self.assertRegex(text, r"x=[+-]\d+\.\d{2}")


class SpeedRankTest(unittest.TestCase):
    """手速榜：同手相邻两键的**最小间隔**，与位移无关。

    口径（用户指定）：

    - 只看非 arc 物件 —— tap / arctap / hold **头**（``SPEED_NOTE_TYPES``）；
    - 间隔按"头 -> 下一个键"算，不改成"尾 -> 下一个键"；
    - 同刻（双押）间隔为 0，不是"两次落下"，不进榜；
    - 单位换算：先 8 分，等效 BPM 超 300 才换 16 分，**再高也不换 32 分**。
    """

    def _config(self):
        config = Config()
        apply_arcaea_physics(config)
        return config

    def _analyze(self, notes):
        from spliter.report import analyze
        return analyze(notes, self._config())

    def test_rhythm_label_uses_eighth_then_sixteenth(self) -> None:
        """8 分优先；等效 BPM > 300 才换 16 分。"""
        from spliter.report import rhythm_label
        # 8 分：间隔 = 30000 / BPM
        self.assertEqual(rhythm_label(250), "相当于 BPM 120 的 8 分音符")
        self.assertEqual(rhythm_label(174), "相当于 BPM 172 的 8 分音符")
        self.assertEqual(rhythm_label(100), "相当于 BPM 300 的 8 分音符")  # 边界含 300
        # 超过 300 -> 16 分：间隔 = 15000 / BPM
        self.assertEqual(rhythm_label(87), "相当于 BPM 172 的 16 分音符")
        self.assertEqual(rhythm_label(50), "相当于 BPM 300 的 16 分音符")

    def test_rhythm_label_never_uses_thirty_second(self) -> None:
        """再快也只降到 16 分，不出现 32 分。"""
        from spliter.report import rhythm_label
        for gap in (10, 25, 40, 60, 75, 99):
            text = rhythm_label(gap)
            self.assertIn("16 分音符", text)
            self.assertNotIn("32", text)

    def test_rhythm_label_matches_canonical_anchor(self) -> None:
        """锚点：172 BPM 的 16 分 = 87ms、8 分 = 174ms。"""
        from spliter.report import rhythm_label
        self.assertIn("172", rhythm_label(87))
        self.assertIn("16 分", rhythm_label(87))
        self.assertIn("172", rhythm_label(174))
        self.assertIn("8 分", rhythm_label(174))

    def test_ranking_only_counts_three_types(self) -> None:
        """排行榜只看 tap / arctap / hold 头。"""
        from spliter.report import SPEED_NOTE_TYPES
        self.assertEqual(set(SPEED_NOTE_TYPES), {"tap", "hold", "arctap"})

    def test_same_hand_adjacent_gap(self) -> None:
        """同一只手的相邻两键构成一条记录，间隔取时间差。"""
        notes = [
            Note(time=0, x=-0.25, y=0.0, type="tap", side="L"),
            Note(time=87, x=0.25, y=0.0, type="arctap", side="L"),
            Note(time=200, x=0.75, y=0.0, type="tap", side="R"),
        ]
        stats = self._analyze(notes)
        self.assertEqual(stats.fastest_gap_ms, 87)
        self.assertEqual(stats.fastest_gap_tied, 1)

    def test_other_hand_between_does_not_break_adjacency(self) -> None:
        """中间插了另一只手的键，这两键对**这只手**仍是相邻的。

        我 200ms 和 400ms 都按了，中间 300ms 是右手 —— 我的左手间隔就是 200ms。
        """
        notes = [
            Note(time=0, x=-0.25, y=0.0, type="tap", side="L"),
            Note(time=100, x=0.75, y=0.0, type="tap", side="R"),
            Note(time=200, x=-0.25, y=0.0, type="tap", side="L"),
        ]
        stats = self._analyze(notes)
        self.assertEqual(stats.fastest_gap_ms, 200,
                         "左手相邻间隔应为 200ms，不被中间的右手键打断")

    def test_simultaneous_chord_not_counted(self) -> None:
        """同刻双押间隔为 0，不是"两次落下"，不进榜。"""
        notes = [
            Note(time=0, x=-0.25, y=0.0, type="tap", side="L"),
            Note(time=0, x=0.25, y=0.0, type="tap", side="L"),
            Note(time=150, x=0.25, y=0.0, type="tap", side="L"),
        ]
        stats = self._analyze(notes)
        self.assertEqual(stats.fastest_gap_ms, 150, "0ms 不该进榜")
        self.assertTrue(all(g.gap_ms > 0 for g in stats.fastest_gaps))

    def test_hold_head_participates_but_tail_is_not_a_key(self) -> None:
        """hold 用**头**参与排名，间隔是"头 -> 下一个键"，不是"尾 -> 下一个键"。"""
        notes = [
            Note(time=0, x=0.25, y=0.0, type="hold", side="L",
                 params={"duration": "400"}),
            Note(time=500, x=0.25, y=0.0, type="tap", side="L"),
        ]
        stats = self._analyze(notes)
        # 头(0) -> 下一个键(500) = 500ms；若错用尾(400) 会得到 100ms
        self.assertEqual(stats.fastest_gap_ms, 500,
                         "hold 的间隔必须从**头**算起")

    def test_arc_not_counted(self) -> None:
        """音弧不是判定物件，不参与手速榜（实谱里 arc 本来就不是 Note）。"""
        from spliter.report import SPEED_NOTE_TYPES
        self.assertNotIn("arc", SPEED_NOTE_TYPES)

    def test_ties_are_counted_from_full_list(self) -> None:
        """并列组数必须来自**全量**列表，不能因显示截断而少算。"""
        notes = []
        for i in range(30):
            notes.append(Note(time=i * 90, x=-0.25, y=0.0, type="tap", side="L"))
            notes.append(Note(time=i * 90 + 5, x=0.25, y=0.0, type="tap", side="R"))
        stats = self._analyze(notes)
        # 左手相邻间隔恒为 90ms，共 29 组
        left_ties = sum(1 for g in stats.fastest_gaps
                        if g.gap_ms == stats.fastest_gap_ms and g.hand == "L")
        self.assertGreater(left_ties, 10, "并列组数不该被 10 条上限截断")

    def test_describe_format_matches_spec(self) -> None:
        """格式：``第一名, 间隔xxms, 相当于BPM…, xx-yy等n组``。"""
        from spliter.report import GapRecord
        record = GapRecord(gap_ms=87, hand="L", from_line=1, to_line=2,
                           from_x=0.0, to_x=0.5, from_y=0.0, to_y=0.0,
                           from_time=99069, to_time=99156)
        text = record.describe(1, tied=27)
        self.assertEqual(
            text,
            "第一名, 间隔 87ms, 相当于 BPM 172 的 16 分音符, 99069-99156等 27 组")

    def test_describe_without_ties_omits_group_count(self) -> None:
        from spliter.report import GapRecord
        record = GapRecord(gap_ms=250, hand="R", from_line=1, to_line=2,
                           from_x=0.0, to_x=0.5, from_y=0.0, to_y=0.0,
                           from_time=100, to_time=350)
        text = record.describe(2, tied=1)
        self.assertEqual(text, "第二名, 间隔 250ms, 相当于 BPM 120 的 8 分音符, 100-350")


if __name__ == "__main__":
    unittest.main(verbosity=2)
