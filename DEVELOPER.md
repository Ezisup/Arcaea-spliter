# Arcaea 拆谱器 —— 开发者文档

面向要继续完善这个项目的人（尤其是 AI）。这里讲**实现怎么组织的、哪些不变式不能破、
哪里是坑**。面向玩家的原理与权重解释在 [`README.md`](README.md)。

当前版本 `0.2.0`，Python ≥ 3.10，**纯标准库**。

---

## 1. 模块结构与职责边界

```
arcaea_spliter/                （仓库根就是这个项目本身）
├── spliter_aff.py          启动器：把 py/ 塞进 sys.path 后转 aff_cli.main
├── pyproject.toml          pip install -e . 后得到 `spliter` 命令
├── initial_prompt.md       最初的原始需求（历史存档）
├── py/spliter/
│   ├── __init__.py         版本号 + 对外 API 汇总
│   ├── chart.py            统一数据模型 Note / Chart（解析器与求解器共用）
│   ├── aff.py              .aff 解析、缓动插值、坐标换算、无损回写
│   ├── constraints.py      音弧颜色 -> 硬约束（ArcConstraints）
│   ├── config.py           全部参数（dataclass）+ 校验
│   ├── split.py            拆谱核心：代价函数 + 束搜索 + 局部优化
│   ├── report.py           质量指标 / 拆前拆后对比 / 逐键解释 / 注意点定位
│   └── aff_cli.py          唯一的命令行入口
├── py/tests/test_aff.py    112 条自测（unittest，全部自足）
├── tools/diff_report.py    与人工参考逐键对照，输出 Markdown 表
├── tools/compare.py        自动锁定观察区间后反复复查
```

### 关键的分层原则

**`split.py` 完全不知道 Arcaea 的存在。** 它只认识 `Note` 与一个可选的
`constraints` 对象，并且只通过鸭子类型问它四件事：

| 方法 | 谁调 | 用途 |
| --- | --- | --- |
| `allowed_hands(index) -> tuple[str, ...]` | `Scoring.can_hand` | 候选生成与可行性检查 |
| `hand_catch(hand, t_from, t_to)` | `move_cost` | 终点是否要接弧头（`arc_arrival`） |
| `hand_release(hand, t_from, t_to)` | `move_cost` | 手被弧钉住的释放点 |
| `arcs_by_hand: dict[str, list]` | `tail_hint` / `_scan_multi_finger` | 弧尾延续提示、多指采样点 |

`_constraint_allows` / `hand_catch` / `hand_release` 都用 `getattr(..., None)`
做容错，**约束缺失或没有某个方法时一律视为放行**。所以 `Scoring(notes, cfg)` 
不带约束也能跑（测试里大量这么用）。

**数据流**（`aff_cli.main`）：

```
parse_aff(text) ─┬─> parsed.chart.notes  ──┐
                 │                          ├─> split(notes, cfg, meta, constraints)
                 └─> build_arc_constraints ─┘         │
                                                       v
                          result.side_by_note  ──> serialize_aff(parsed, hands_int)
```

`Note` 用 `@dataclass(eq=False)`：**按对象身份做字典键**。
`result.side_by_note` 与 `serialize_aff(hands=...)` 都依赖这一点，
所以回写永远用**物件对象**对齐，绝不用 `index`（见 §4.2）。

---

## 2. 求解管线：`split()` 的准确顺序

```python
def split(notes, config, meta=None, constraints=None):
    scoring = Scoring(notes, config, meta, constraints)   # 预计算一切
    groups  = build_groups(notes)                          # 同时刻分组
    _check_lock_conflicts(scoring, groups, warnings)       # 原生 side 冲突
    _check_chord_feasibility(scoring, groups, warnings)    # 和弦可行性
    hands = _beam_search(scoring, groups, config, warnings)
    hands = _refine(scoring, hands, config)
    hands = _enforce_hold_tail(scoring, hands)             # 最后强制收口
    score = scoring.evaluate(hands)
```

四项检查的顺序有意义：**`_enforce_hold_tail` 必须放在 `_refine` 之后**。
"长条尾指同手"是语义要求，不该由代价和平局决定；它在求解过程里只是候选生成
／可行性检查的一个条件，被平局或局部优化绕过的可能性不值得赌，所以最后统一收口。

它的**兄弟规则**"长条占手"（区间内不得同手）不需要这样一个收口 pass —— 那条
已经由 `_feasible_region` 的区间扩展堵住了，见 §2.4。两条规则的分工是：

| 规则 | 保证方式 |
| --- | --- |
| 尾指同手（**必须**同手） | `_enforce_hold_tail` 最后强制 |
| 区间异手（**不得**同手） | `_feasible_region` + `_hold_extended_stop`（正本清源） |

`scoring.unresolved` 非空时追加一条警告（硬约束自相矛盾，已按
"原生标记 → 多指钉手 → 绝对左右"兜底，需人工复核）。

### 2.1 束搜索 `_beam_search`

状态是 `_State(cost, key, hands, last, hold_until, counts)`。逐步推进：

```python
beam = [初始状态]，start_time = 第一个物件时间 − home_ready(350ms)
        起手位 home = {L: 0.35075, R: 0.64925}   # 中线 ± 80px

for group in groups:
    fixed_candidates = None if 该组落在某条长条区间内 else _realizations(scoring, group)
    best = {}
    for state in beam:
        for assignment in candidates:
            expanded = _expand(...)
            if expanded and (key 没见过 or 更便宜): best[expanded.key] = expanded
    states = sorted(best.values(), key=lambda s: (s.cost + _tie_break(s), s.key))
    if not states: <两层兜底，见下>
    beam = states[:beam_width]      # 56

beam.sort(key=lambda s: (s.cost + scoring.balance_cost(s.counts), s.key))
return list(beam[0].hands)
```

**几个要点**：

1. **候选缓存**：只有在"该组落在某条长条区间内"时，候选才依赖当前手标记
   （`hold_conflict` / `hold_tail_hand` 要看已定的手）。其余情况候选是固定的，
   算一次就够。`hold_spans` 用**真实区间**（不是权重侧的 `hold_duration`）
   来判断"在不在区间内"。
2. **`_State.key` 是剪枝的唯一依据**：
   `"last_L|last_R|hold_until_L|hold_until_R|count_L|count_R"`。
   `last[hand]` 决定了下一组的 `move` / `jack` / `hardness` / `cross`，
   `hold_until` 决定长条占手，`counts` 决定均衡项。**同一个 key 只留最优状态**
   是束宽可控的关键。
   `hold_until` **不是** `last` 的冗余：一只手的 `last` 已经移到后面的 tap 上，
   而它按过的长条可能还没结束 —— `hold_until` 必须独立保留。
   （不要因为"看起来能推出来"就删掉它。）
3. **平局打破 `_tie_break`**：`balance_cost(counts) × 1e-3`，量级远小于真实代价差。
   加上它是因为**只按 key 的字典序会让"整段都给左手"这类解系统性胜出** ——
   像素刻度下大量转移代价相同，这个偏置会真的改变结果。
4. **两层兜底**（`states` 为空时）：
   - 第一层：放弃"长条占手"这一条硬约束，用 `_realizations(scoring, group)`
     重新展开（`_expand(..., ignore_hold_conflict=True)`），按最小代价挑，
     并记警告。典型场景是**两条长条同时占着两只手、谱面又在区间里放了天键**
     （实际要第三根手指）。
   - 第二层：仍无解则沿用上一个 beam，记"无法分配左右手"警告。
   **`不能把物件留成"没有手"`** —— 那回写出去就是个空标记。这是硬性要求。
5. **首次出手的代价**（`previous is None`）与后续不同：从起手位走过去，
   超 `max_speed×3` 收 20，超 `comfy_speed` 收 `move×ratio²×0.5`。
   注意这一段**只在 `_expand` 里**，`evaluate()` 的 `_segment_sum` 没有对应项
   —— 见 §3.3 的"已知不一致"。

### 2.2 候选生成 `_realizations`

返回"同一时刻的物件 → 所有合法的'一键一手'分配"。

```python
# 多指配置（组内任一键在 multi_finger 里）：一只手可以有几根手指
options = [ [want] if want 且 can_hand else [所有 can_hand 的手]  for each index ]
if all(options): return list(product(*options))

# 普通配置：DFS，用 frozenset 保证"一键一手"
walk(slot, used):
    for hand in HANDS:
        if hand in used or not can_hand(index, hand): continue
        if 组内已有"与它是 is_tail_pair 的键"且手不同: continue
        if hands is not None:
            if hold_conflict(index, hand, hands): continue
            if hold_tail_hand(index, hands) 要求别的手: continue
        ...
```

**"一键一手"做成硬约束而不是扣分**，理由写在 docstring 里且值得记住：
和弦同色对玩家是"物理上做不到"，**靠加大罚分永远避免不干净** ——
罚得再重，总会在某个局部上下文里显得"划算"。这是项目里最重要的一条设计教训。

**退化分支**（`results` 为空）：枚举 `2^len(group)` 全部掩码，按
`(是否满足可及范围, 最大同手跨度, 两手总跨度)` 字典序最小挑选，即"让最勉强的那只手
尽量轻松"。最后**再兜底**：按 `(sides[index], multi_hand.get(index), 绝对左右, "L", "R")`
顺序取第一个 `can_hand` 的；全都不可行才记进 `scoring.unresolved`。

> **这里绝不能静默用 `slot % 2` 凑数。** 那会把非法手标记直接写进谱面
> （历史上真的发生过：某个谱 22 处非法手标记）。

### 2.3 状态转移 `_expand`

逐 `(slot, index)` 累加：`move`（或首次出手代价）→ `jack` → `hardness` →
`bias` → `hold_conflict_cost` → `hand_cross_cost`。
再加 `hold_conflict` / `hold_tail_hand` 的**硬性拦截**（返回 `None` 剪掉这条路）。

`may_jack(previous, index)` 只在**两个键都不是原生锁定**时才收 jack；
`hardness` 只在**当前键不是锁定**时才收。
理由：锁定的键是既定事实，不该被罚。

### 2.4 局部优化 `_refine` / `_neighbours`

每轮把邻域**全部**评一遍，取最好的接受（steepest descent）。

> 曾经为了提速改成"发现更好就立刻接受"，结果在"每 100ms 挪 100px"这类
> **平坦但需要整体协调**的段落上会卡在次优解（实测比最优贵 15%）。不要改回去。

五种候选：

| # | 名称 | 作用 |
| --- | --- | --- |
| 1 | 单键翻转 | 常规改进；受 `_feasible`（整组）检查 |
| 2a | 窗口内同标记交换 | 把窗口里两个**相同标记**的位置对调，修"等代价平局"的相位 |
| 2b | 邻近交换 | 把"抗"的边界往旁边挪一格；间隔 > 900ms 就 `break` |
| 3 | 前缀翻转 | 翻转某组边界之后的所有标记，修"这一段从哪只手起" |
| 4 | 整段镜像 | 长度 ≤ `mirror_max`(5)、时间跨度 ≤ 2600ms，区间内不能有锁定键 |

**可行性检查必须按"区域 + 整组"**：`_feasible(scoring, hands, index)` 实际委托给
`_feasible_region(scoring, hands, group[0], group[-1]+1)`。把双押左边那个键翻到
和右边同一只手，只检查被翻的那一个键是查不出来的。

**而且区域里的长条要把自己的占手区间并进来**（`_hold_extended_stop`）。
只查"被翻物件**自己那一刻**的组"还漏一种情况：**长条换手会牵动它整段** ——
把长条翻到与区间**后面**某个键同手时，那个键不在区域里，于是查不出来。

> 实测 bug（`cc_test.txt` 109833 / 113166）：波束搜索给的是对的
> （`hold=R, tap=L`），被 `_refine` 翻成了 `hold=L, tap=L` —— 长条按住期间
> 同一只手又去按了旁边一个键。两个条件缺一不可：
> ① 上面这个区域漏洞；② 那种 167ms 的短长条（< `hold_min_duration` = 260）
> 在**权重侧**根本不算占手，所以"翻过去"不花一分代价，局部优化就一定会去翻。
>
> 修完之后 `cc_test.txt` 的代价从 14.08 升到 **16.32**（那个便宜解本来就是
> 靠违规换来的），而 `deinosphainein_2` 的代价**一位没变**（2192.2492）——
> 只在"以前在作弊"的谱面上生效。
>
> 五种邻域全部经过 `_feasible` / `_feasible_region`，所以这一改之后
> `_refine` 不可能再产出"长条占手"违规，不需要再补一个事后收口 pass。

`_feasible_region` 里三种情况：
- **多指组**：允许同手，但手标记必须与 `sides[index] or multi_hand.get(index)` 一致
  （局部优化**不许改钉手**）；
- **`can_fill_group` 为真**：必须一键一手 + 不撞长条 + 尾指同手 + 组内 `is_tail_pair` 同手；
- **三押类无解组**：仍然要求"尽量摊平"，允许的手数上限 `ceil(n/2)`，
  不能退化成"三个键全给一只手"，也不能出现空标记。

---

## 3. 增量打分：`delta` / `_segment_sum` / `PrefixCache`

这是**性能的核心**，也是最容易写出错的地方。

### 3.1 不变式

> **总分 = 所有物件的局部代价之和。**
> 所以改动若干个物件之后，只要重扫"受影响的那一段"，
> 用 `新段之和 − 旧段之和` 修正总分即可。

`_segment_sum(scoring, dense, lo, hi, start_state)` 按扫描顺序重建这一段的
各分项，**与 `Scoring.evaluate()` 走完全同一套逐物件代价**（`evaluate` 就是
把整份谱当"一个 0..n−1 的段"交给它）。这样"两处各算一遍、结果对不上"的问题
从结构上被消掉了。

**已验证**：随机 40 键谱面，逐个物件翻转，`delta()` 与 `evaluate()` 
的分项差异为 0（`tests` 之外可这样手测）。

### 3.2 `delta` 怎么定段的边界

```python
lo, hi = 所有 changed 物件的 order 位置 的最小/最大值
       ∪ 它们各自同组成员的 order 位置
对每个 changed：往前找到"这只手在这之前的那个键"，把 lo 扩到它
                （它本身不变，但"以它为 previous"的键变了）
对每个同组成员：同样往前找同手的那个键
lo = max(0, lo)
hi 再往右扩：直到"两只手都各自又出现过"之后的一小段
```

段边界取"足够大"不影响正确性，只影响一点速度。

**`start_state` 必须把段前状态接上**，否则 `delta` 重扫一段时会漏掉段首那笔
`cross_cost`，与 `evaluate` 对不上。`_segment_sum` 里为此单独维护
`previous_time_index`（"上一个**不同时刻**的有手物件"），
有 `start_state` 时还要从 `lo-1` 往前找一次。

### 3.3 `PrefixCache` 与已知不一致

`PrefixCache.state_at(position)` 惰性给出"扫完前 `position` 个位置"的
`(last_by_hand, previous_hand)`。某一轮局部优化里所有候选都从同一份 `current`
出发，前缀状态只在**被改动的那些位置之前**才会不同。

> 注意：缓存只能停在**组首**（`state_at` 内部会 `_group_bounds` 对齐），
> 因为换手是**按整组**记的。

实测把真实谱面（约 600–800 物件）的局部优化从 18 秒压回个位数。

**已知不一致（`delta` 与 `evaluate` 在特定改动下会差一点）**：
`_expand` 里有"首次出手从起手位走过去"的代价，而 `_segment_sum` 没有对应项。
若一次改动让某只手**第一次出手的位置**发生变化（前面本来没有这只手的键，
改完有了），`delta` 的段内求和就少了那一笔。`delta` 在 `lo` 往前扩到
"这只手之前的那个键"时找不到就会停在 0，此时差异存在但**有界且很小**
（起手项最多 20）。它只影响局部优化里少数候选的排序，不影响最终
`evaluate()` 报告的分数。**如果以后要追求严格一致，就在这里补一个"起手项"**
（`_segment_sum` 在 `lo == 0` 且某手 `last_by_hand` 为空时加同一笔）。

### 3.4 分项与重算

`ScoreParts` 的分项：`move / jack / bias / switch / cross / busy / balance`。
`delta` 里前 7 项中的 6 项走增量，**`balance` 每次整体重算**
（只依赖两手计数，O(n) 但很便宜）。

`jack` 这一项同时装了 `jack_cost` **和** `hardness_cost`（`_segment_sum` 里
两行都 `parts.jack += ...`）。所以报告里"抗"的分项是两者的和，
调试时不要以为只有 `jack_cost`。

---

## 4. 不变式与容易踩的坑

### 4.1 单位系统：px、谱面坐标、轨道

**这是最容易出错的地方。** 多套单位混用会导致阈值静默失效。

- 谱面坐标：Arcaea 是"单位"（横向 2.0 个单位的场地）。
- 屏幕像素：`pixel_per_unit_x = 536` / `pixel_per_unit_y = 326`，**两轴不同**。
- **轨道**：给人看的口径，`1 轨道 = 0.5 坐标 = 相邻轨道间距`。
  只在报告与文档里出现，**绝不参与代价计算**。

规则：
1. **任何位移都用 `screen_span(dx, dy, physics, time_ms)`**
   （= `hypot(dx·536, dy·326)`），**绝不只算一个轴**。
   `time_ms` 要传，否则六轨段落刻度不对（§4.1b）。
2. **凡是"多少像素算多远/多近"的阈值，用之前必须换算**：
   `Scoring.px_to_units(v, time_ms)` 或等价地 `v / pixel_per_unit_x`。
   涉及：`neutral_band`、`strong_bias`、`max_reach`（经 `px_to_units`）、
   `jack_min_distance`、`cross_near` / `cross_far`（在 `cross_cost` 里乘 `scale`）。
3. **`Note.width` 在 AFF 里是 0.5 个单位**（`parse_aff` 把 `meta["noteWidth"]`
   设成 `LANE_STEP`）。绝不能用像素口径的 120，
   放进 AFF 坐标会被当成 120 个单位。
4. **给玩家看的数字用轨道，不用像素**（见 §9.6）。换算走
   `report.lanes(distance_px, physics, unit_px)`，且**必须传当段刻度**。

> 历史事故（值得记住形状）：`jack_cost` 里 `distance` 是**世界单位**（≤1.4），
> 却拿去和 `width × jack_decay`（=168，像素量）比 —— 空间衰减项恒为 ~1，
> **等于整个"抗的空间衰减"失效**。同类问题还有"15 像素的中性带变成 15 个世界单位"
> （比判定面还宽两倍，空间规则被整段关掉，左右只能靠平局乱分）。
>
> 另一类事故在**报告**里：把横向和纵向的位移合成一个勾股值再来报"跨 N 轨道"，
> 读者无法判断那 N 是横向还是纵向 —— 现在拆成两个分量分别报（§9.6）。

**写新代价函数时的自检**：这一项里出现的每个长度，分别是什么单位？
如果同一个表达式里既有 `note.x` 差值又有 `physics.pixel_per_unit_x`，
必须确认是有意为之（`cross_cost` 就是这样：距离用像素，阈值乘 `scale` 转成像素）。

### 4.1b 六轨模式（`enwidenlanes` / `enwidencamera`）的刻度换算

**已实现。** 这两个事件会让同一块屏幕装下更宽的弧坐标范围，所以像素刻度不再是
全局常量，而是**按时间分段**的。相关类型：

| 位置 | 作用 |
| --- | --- |
| `aff.SceneControl` | 一条 `scenecontrol` 事件（时间 / 类型 / 启用 / 过渡时长） |
| `aff.ScaleSegment` / `aff.ScaleSchedule` | 稳态区间 + 倍率，`schedule.at(t) -> (sx, sy)` |
| `aff.build_scale_schedule(events, duration)` | 把开/关事件折成区间表 |
| `PhysicsConfig.scale_schedule` | 挂在 config 上（`None` / 空 = 全程 4k） |
| `split.scale_at(physics, t)` | **唯一的刻度查询入口**（含倍率） |
| `Scoring.scale_x_at(i)` / `scale_y_at(i)` | 按物件下标取刻度 |

#### 事件语义

```aff
scenecontrol(t, enwidencamera, 过渡时长ms, 1);   // 1 = 启用/淡入，0 = 关闭/淡出
scenecontrol(t, enwidenlanes,  过渡时长ms, 1);
```

参考实例：`rivenpilgrim_3` 第 1703–1748 行。该谱 **129333ms 启用、151999ms 关闭**，
中间每 1000ms 重发一次 `0.00,1` 维持状态（这是"保持"而非"重复启用"）。
区间内有 **176 个判定物件**（占全谱 800 的 22%），其中 **28 个是 0/5 轨的地面键**、
14 处 arc 用到 `y=1.61` —— 规模不小，不能当边角情况忽略。

#### 几何（按"变换完成后维持 6k"的状态）

用户给出的口径：

- **0 轨左边界 = 原 1 轨左边界**（`-1.0` ↔ `-0.5`）
- **5 轨右边界 = 原 4 轨右边界**（`+2.0` ↔ `+1.5`）
- **新的 `y = 1.61` = 原来的 `y = 1.0`**

即 **3.0 个弧单位占据原来 2.0 个单位的屏幕宽度**：

| | 4k | 6k | 缩放 |
| --- | --- | --- | --- |
| x 刻度 | `536` px/单位 | `536 × 2/3 = 357.33` | **× 0.6667（缩小 1.5 倍）** |
| y 刻度 | `326` px/单位 | `326 / 1.61 = 202.48` | **× 0.6211（缩小 1.61 倍）** |
| 轨道宽 | 0.5 单位 | **仍 0.5 单位** | 不变 |
| 中线 | `x = 0.5` | **仍 `x = 0.5`** | 不变（0/5 轨左右对称加出） |

**轨道坐标不用改**：`lane_to_x` 已经给出 `0轨 → -0.75`、`5轨 → +1.75`
（因为它是 `-0.25 + (lane-1)*0.5` 的线性外推），与实测谱面解析结果完全一致。
**要改的只有两个像素刻度。**

验算一个具体位移：`0轨 → 5轨` = 2.5 单位 = `2.5 × 536 = 1340px`（4k 刻度）
vs `2.5 × 357.33 = 893px`（6k 刻度），比值正好 **1.5**。

y 方向：`y = 1.61` 在 4k 刻度下是 `525px`，在 6k 刻度下是 `326px` ——
后者正好等于原来 `y = 1.0` 的高度，与"新的 1.61 对应原来的 1.0"自洽。
**这条自洽性是判断刻度算错没有的最快自检**，回归测试里也断言了它。

#### 实现约定（改动前务必先读）

1. **只处理稳态，不算过渡过程**。用户明确说明"从 4 轨扩展到 6 轨过程中的
   坐标变换太复杂，不需要计算"。`build_scale_schedule` 按事件时刻**瞬时切换**，
   `param1`（过渡时长）被解析但**不参与计算**。维持段重发的事件本来就写
   `param1 = 0.00`，瞬时切换是精确的；只有淡入/淡出那一小段
   （实测 1333ms / 5333ms）不精确 —— 这是**已知且有意的近似**。
2. **两个事件语义独立**。`enwidenlanes` 只管 x（加 0/5 轨），
   `enwidencamera` 只管 y（Sky Input 抬到 1.61）。实测成对出现，
   但**不要**假设一定成对 —— 各有单独的回归测试。
3. **所有刻度读取都必须过 `split.scale_at()` / `Scoring.scale_x_at()`。**
   直接读 `physics.pixel_per_unit_x` 会漏掉六轨倍率，
   在 6k 段落里把位移高估 1.5 倍。已经改过的下游：
   `screen_span` / `screen_distance`（加了可选 `time_ms`）、
   `home_distance`、`px_to_units`、`cross_cost`、`is_bullet`、`jack_cost`、
   `bias_cost`、弧尾提示、`_check_lock_conflicts`、以及
   `constraints._same_point`（走 `build_arc_constraints(..., schedule=)`）。
4. **`screen_span` 的 `time_ms` 是可选参数**：不给时退回标准四轨刻度，
   保持旧调用与测试可用。新代码应当**总是**传时刻。
5. **起手位目前仍按 4k 刻度算**（`apply_arcaea_physics` 里
   `home_l/r = 0.5 ∓ 80/unit_px_x`）。6k 下严格说应是
   `0.5 ± 80/357.33` = `0.2761 / 0.7239`（4k 是 `0.3507 / 0.6493`）。
   起手位只在**第一个物件之前**起作用，而六轨段极少出现在开头，
   所以**没有跟着改** —— 这是个有意的取舍，不是遗漏。
6. **`--stdout` 与 `-`（标准输入）不支持**：CLI 把第一个位置参数当文件路径，
   传 `-` 会报"找不到输入文件"。要串联处理请用真实的中间文件。

#### 回归样本

`rivenpilgrim_3` 的 **129333–151999ms**：区间内 176 个判定物件（占全谱 22%）、
28 个 0/5 轨地面键、14 处 arc 用到 `y=1.61`。
解析后 `scale_schedule()` 应折叠成**恰好 1 段** `[129333, 151999)`，
倍率 `(2/3, 1/1.61)`；区间外必须是 `(1, 1)`。

### 4.2 索引与排序

- `Note.index` 是**时间升序**（`sort_notes` 按 `(time, x)` 排）后的序号；
  `Note.raw_index` 是文件里的原始顺序。**两者可能不同。**
  回写用 `Note.line`（1 基行号）或物件对象，**不要用 `index` 对齐行**。
- `Scoring._order` 是扫描顺序（按时间，组内按 x），`_order_of[index]` 是反向表。
  **`dense` 数组（局部优化用）按 `_order` 索引，`hands` 按 `index` 索引**。
  两者混用是隐蔽 bug。`_HandView` 就是为了避免"为取一只手而整表重建"。
- `Note` 是 `eq=False` → **hashable by identity**。可以当 dict 键，但
  **不要**构造两个"看起来一样"的 `Note` 去查表。

### 4.3 顺序假设

- `Scoring._hold_spans` **按 `position`（扫描顺序）追加，等价于按起点有序**。
  `hold_tail_hand` / `hold_conflict` 里因此可以 `if start >= time: break`。
  改动追加逻辑时这条必须保持，否则 `break` 会漏项。
- `constraints.arcs_by_hand[hand]` **按 `max(t1,t2)` 排序**，
  `hand_release` 依赖它做 `if end >= t_to: break`。
- `_segment_sum` 要求 **`lo` 组对齐**（内部会自己 `_group_bounds` 扩一次，
  但调用方最好别依赖它兜底）。

### 4.4 容差与"同一个点"

- `_POS_EPS = 1e-9`：判定"同一个位置"。轨道间隔 0.5，1e-9 足够区分。
- **`_same_pos` 必须同时比 x 和 y。** 地面长条尾指停在 `(x, 0)`，
  正上方 `(x, 1)` 的天空键差 326px —— 一根手指不可能同时按住地面又点天空。
  只比 x 会产生假代价（实谱一处曾被算成 16.5 万）。
- `SAME_HAND_MARGIN_MS = 2`：长条区间边界的前后容差；**也是多指段
  "±2ms 内并成同一个时刻"用的同一个容差**。
- `_spot(x, y)` 把点 `round(_, 3)` 后去重，**多指段数"需要几根手指"用**。
- `_LEVEL_EPS = 0.25`：多指段里判定"同一高度"（y 相差多少算同一层）。

### 4.5 AFF 回写的两个硬性要求

1. **幂等**：`serialize_aff` 对已经标过的行**原位替换**。
   手标记的位次由格式决定、**用固定下标**（`_HAND_INDEX = {"tap": 2, "hold": 3}`），
   **不要用"当前行有几个参数"去推断** —— 对已经标过的行会算错位次、把整行写乱。
2. **`arc(...)` 参数一律不动**（它的第 8 个参数 `color` 就是手标记）。
   只有 `tap` / `hold` 补末尾参数，`arctap` 在括号内补。
   非判定行逐字保留、行数不变。这三条都有测试守着。

### 4.6 `timinggroup(noinput)` 的隔离语义（单向影响）

`noinput` 是**演出组**：里面的物件不受玩家输入，但仍然要**上手标记** ——
否则一堆拆好的键里冒出两个原色会很突兀。所以取折中，规则只有一条：

> **组内物件可以受「组外」和「本组」的影响，但不能影响「组外」。**

实现是「一处判据 + 四处使用」，改任何一处都要回头看其余：

| 位置 | 干什么 |
| --- | --- |
| `aff.parse_aff` | 跟踪 `timinggroup(noinput){ … };`，把组号写进 `Note.noinput` / `AffArc.noinput`（0 = 普通） |
| `chart.isolated(src, dst)` | **唯一的判据**：源物件在隔离组、且和目标是**不同组** → 约束不生效 |
| `split.Scoring.isolates` | 长条侧对 `chart.isolated` 的薄封装（读 `self.noinput`） |
| `constraints` / `report` | 硬约束侧、报告侧直接用 `chart.isolated` |

判据**只放 `chart.py` 一处**：三边都要用，而 `chart` 谁都不依赖
（`constraints` import `split`、`report` import `split`，都不能当共同底座）。

`Scoring.isolates` 的四个调用点 —— **少一个都会漏**：

- `hold_tail_hand`：隔离组的长条不钉组外的键；
- `hold_conflict`：隔离组的长条不挡组外的键；
- `_conflict_holds`（→ `hold_conflict_cost`）：不对组外收占手代价；
- `move_cost`：隔离组的长条**不当占手**，占手效应不外溢。

`constraints.py` 另有三处：规则 2 的「同刻起弧」、规则 3 的「区间异手」、
`_hold_windows` 产出的占手区间。
`report.analyze` 还有一处：**报告不能再报一次假警报** —— 求解器既然允许组外的键
与演出长条同手，`长条占手冲突` 就不该把它数进去（实测会多报 2 处）。

`_scan_multi_finger` 走的是**另一套**实现（按组给落点分桶，见 §5），
它不经过 `isolates`，所以改多指逻辑时别只盯着 `isolates`。

> 这一条修的是实测 bug：`deinosphainein_2` 的 `timinggroup(noinput)` 里有两条
> 演出用长条（`hold(97263,98526,2)` / `(…,3)`），它们在 98526 那一刻多贡献了两个
> 地面落点，把普通配置顶到阈值 3、判成多指段；多指段会**整段无视**
> "有弧时不得同色"，于是那一刻的正常长条被分成了弧的颜色（R）。
> 修完组外落点数回到 2，长条正确判成 L。

**安全性**：没有 noinput 组时 `isolates` 恒为 `False`、落点分桶只剩 0 号桶，
所以普通谱面的行为**逐位不变**（实测同一份合成谱面：代价精确到 9 位小数、
手标记、多指集合、`forbidden` 全部一致）。

**没做的部分**（有意留着，别当成 bug）：弧的**软**代价（`hand_catch` /
`hand_release` / `tail_hint`）以及 move/jack 之类的成对代价仍然看得见 noinput 物件。
这恰恰是让演出物件"融入"的原因 —— 全屏蔽掉它们会随机上色，就是用户不想要的突兀。

---

## 5. 多指段：预扫描 `_scan_multi_finger`

**在拆谱之前一次算清**，结果落在 `Scoring.multi_finger` / `multi_hand` /
`multi_finger_times`。

```python
时刻 -> 该时刻的判定物件      # ±SAME_HAND_MARGIN_MS 内并成同一时刻
+ 仍在按住的长条落点           # start <= time <= end
+ 实体弧在**这一刻的采样点**   # lo <= time <= hi，长度无关
= spots（按 _spot() 去重，**并且按 noinput 组号分桶**）
if len(spots) >= multi_finger_min(3): 该组进入多指模式
```

落点是**按组分桶**的（0 号桶 = 组外）：

- 组外的键只看 0 号桶 —— 隔离组的装饰物件不得把普通配置抬成多指段；
- 隔离组的键看 `0 号桶 | 本组桶` —— 它仍然受组外影响（见 §4.6）。

没有 noinput 组时只剩 0 号桶，行为与"分桶之前"完全一致。

**四个必须记住的细节**：

1. **落在同一个点只算一根手指**。重复计数会把普通配置误判成多指段。
2. **同一只手的多条弧只算一个触点**（`arc_hand_counted`，按组各记一份）。
   否则"1 个键 + 2 条同色弧"会被数成 3 个触点 → 误抬成多指段 →
   而多指段会整段无视"有弧时不得同色"这条硬约束。**这是用户报告过的真实 bug。**
3. **黑线（`arctype=true`）不参与**：`color` 是占位符、没有语义。
   挂在黑线上的天键本来就会被正常流程算一遍，再用黑线颜色"锚"一次是重复计算。
4. **弧的采样点参与同 y 的左右排序**（`_multi_finger_hands`）：
   - 骑在弧采样点上的键**直接听弧色**（弧的颜色改不了）；
   - 其余按 x 排序、均分左右两半；
   - 每只手最多一个弧占位（`padded`），否则 `len(points)` 被抬高、
     左右分半的边界会漂；
   - 兜底：同刻两个键被判成同一只手时按 x 分开，**但若其中一个带弧色锚点就听弧的**。

`can_hand` 在多指段**直接 `return True`**（跳过 `constraints`）：
一根手指在天上按着弧，同一只手的地面手指照样能点键，
所以双指那套"区间异手"在这里不适用。

**钉手是强偏好，不是硬约束**：`can_hand` 刻意**不**判它；
真正落实在 `_realizations`（候选只给钉手）与 `_feasible_region`
（不许改动钉手）。这样"钉手与硬约束打架"时不会两头都判非法 —— 
早期版本在这里两头都判非法，最后退回 `HANDS[slot % 2]` 的静默兜底，
实测某个谱产生 22 处非法手标记。

---

## 6. 硬约束层 `constraints.py`

`ArcConstraints` 只回答两个问题：`force_hand(index)` / `forbidden_hands(index)`，
统一由 `allowed_hands(index)` 暴露。`build_arc_constraints` 的三条规则：

| 规则 | 条件 | 结果 |
| --- | --- | --- |
| 2 同刻同手 | 地面物件与实体弧**同刻起步**（±2ms）**且落点相同** | `forced[index] = 弧的手` |
| 3 区间异手 | 落在实体弧 `[min(t1,t2), max(t1,t2)]` 内 | `forbidden[index] += 弧的手` |
| 4 多押例外 | 同刻多押被迫用不同手，两条规则打架 | **放行**，`chord_exceptions += 1` |

**规则 2 必须按"点"判，不能按"这一时刻所有起弧的颜色"一刀切。**
谱面里常见"同一刻两条不同颜色的弧在不同位置起步 + 一个键落在其中一条的弧头上"
—— 那种键必须与**它所在的那条弧**同手。只有同一个点上真的挤了两条不同颜色的
弧头时才叫"多押例外"。位置容差 `ARC_START_SAME_POS_PX = 134.0`（半条轨道）。

**规则 3 里 hold 按区间重叠判**（`HOLD_ARC_OVERLAP = True`）：只看起点会漏掉
"长条按到一半时另一条弧才起"这种组合。`_is_own_arctap` 排除"挂在弧自己身上
的 arctap"（同一根手指，不构成异手）。

`own_spots` 用 `id(arc)` 做键 —— `AffArc` 是普通 dataclass（`eq=True`），
所以用 `id()` 而不是对象本身，避免结构相同的两条弧互相干扰。

**隔离组的弧/长条不参与这三条规则对"组外"的判定**（`_isolated`，见 §4.6）。
实测 bug 就是组外的长条被装饰弧带着走：`deinosphainein_2` 98526 处，
正常长条本应判 L，却因为多指段放行而跟着弧色判成了 R。

---

## 7. 参数与调参面

`config.py` 里五个 dataclass。**对外只有两个可调项**：

```python
_TUNABLE_RANGES = {"cost.jack": (0.0, 500.0), "cost.move": (0.0, 100.0)}
```

`validate(config)` 只检查这两项的范围 + `output.side_case`。
命令行 `--jack` / `--move` 是仅有的覆盖入口。

### 7.1 参数在哪一层被固化

`apply_arcaea_physics(config, unit_px_x, unit_px_y)` **只写两个轴的刻度**
和由刻度推出的起手位（`home_l/r = 0.5 ∓ 80/unit_px_x`）。
其余全部固化在 dataclass 默认值里。

> 历史：这个函数曾经把 1280px 那套标定值整体换算过来。
> 现在 Arcaea 是唯一分支，换算已经没必要 —— **要改模型就改默认值**，
> 不要再往这里加换算逻辑。

### 7.2 模块级常量（不在 config 里）

时间/几何类的小常量直接写在 `split.py` 顶部，**因为它们是"标定结论"而不是"旋钮"**：

| 常量 | 值 | 作用 |
| --- | --- | --- |
| `_MOVE_COST_MAX` | 40.0 | 单次移动代价封顶 |
| `_TAIL_HINT_MS` / `_TAIL_HINT_PX` / `_TAIL_HINT_BONUS` | 250 / 44 / 0.35 | 弧尾延续的**软偏好** |
| `_STREAM_ALT_MS` / `_STREAM_ALT_BONUS` | 130 / 0.3 | 密集串里的**交替奖励**（负代价） |
| `_REACH_MS` | 100.0 | 出张扣分的间隔门槛 |
| `_TAPER_START_MS` | 100.0 | 所有时间惩罚的收尾起点 |
| `_TAPER_JACK_END` | 120.0 | jack 收尾终点 |
| `_TAPER_HARD_END` | 180.0 | hardness 收尾终点（**故意与 jack 不同**） |
| `_DENSE_TIME_CONVEX` | True | 死区内时间因子取平方 |
| `_STATIC_DURATION_SCALE` | True | 静态空间罚按占手时长放大 |
| `HOLD_ARC_OVERLAP` | True | 长条与弧的冲突按区间重叠判 |
| `_LANE_PAIR_MIN/MAX` / `_LANE_Y_EPS` | 0.45/0.55/1e-3 | `lane_reverse` 的收紧判据 |

> `tools/compare.py --set KEY=VALUE` 能临时改这些（`_` 开头的走
> `setattr(spliter.split, ...)`，否则按 `a.b.c` 路径改 config）。
> 这是做 A/B 实验最快的入口。

---

## 8. 回归防线

### 8.1 自测

```bash
python3 -m unittest discover -s py/tests     # 112 条，0 skip，约 0.16s
```

**全部自足**：不需要任何仓库外的谱面文件，整个包可以独立跑通。

历史上曾有两条跑在真实谱面 `breachoffaith_2` 上的测试
（`test_real_chart_matches_reference_hands` 的 630 键人工基线对照、
`test_real_chart_constraints_are_satisfied`），**已连同
`py/tests/fixtures/breachoffaith_2_hands.json` 一并移除**。原因：

- 那张谱不是理想的测试谱，**覆盖很不全面**（只是当时随手举的例子）；
- 谱面本体不在仓库里，测试在干净检出上永远 skip —— 等于没有防线；
- 每次跑要十几秒，明显拖慢常规回归。

**腾出来的覆盖由合成谱面接上**：
`test_solver_respects_arc_constraints` 用一段自足的
"两条异色实体弧 + 弧头同刻键 + 区间内落键 + 双押"谱面，
保留原来那三条断言（同刻同手被遵守 / 区间异手被遵守 / 地面双押分开两只手），
并且**断言样本真的触发了约束**（否则测试会空过）。
`test_no_unassigned_hands_on_arc_heavy_chart` 覆盖"弧密集时不许出现空手标记"。

> 这两条测试做过**变异验证**：把 `_constraint_allows` 改成恒返回 `True`
> 之后，整套测试有 7 条转红 —— 说明它们真的能抓住"求解器不遵守硬约束"。
> 加约束相关的测试时请保持同样的可证伪性（别写成永远通过）。

### 8.2 每轮都要过的不变式

1. `arc(...)` 的参数**逐字未改**；
2. 所有判定物件都标了手（`result.counts()["none"] == 0`）；
3. **违反硬约束 0 处**（`constraints.allowed_hands` 全覆盖）；
4. **重复处理幂等**（对输出再跑一次，结果逐字节相同）；
5. 自测全绿（112 条 / 0 skip）。

> 上面 1–4 条在**大谱面**上仍然值得手工过一遍：自测是合成小谱，
> 抓不到"真实谱面里长尾组合"的问题。手边的 `halcyon_2` / `gengaozo_2` /
> `rivenpilgrim_3` 等都可以用 `spliter_aff.py` + `diff_report.py` 快速核。

### 8.3 与人工参考对照

```bash
python3 tools/diff_report.py 谱面.aff 参考.aff     # 逐键表 + 代价
python3 tools/compare.py 谱面 参考                 # 自动锁定观察区间
```

`diff_report.py` 的关键列是"**本簇翻成你的代价**"：
把相邻（≤400ms）的不一致键**整簇**改成参考那侧后的总代价变化。
接近 0 = 两种分法在模型里**等价**（平局，不是错）。**必须按簇算** ——
只翻单个键会把下一对变成新冲突，得出误导性的巨大差额。

`compare.py` 的观察区间机制：第一次全谱求解、按差异每处取前后 10s
（重叠才相连、不延长）写进 `<谱面>.<参考名>.watch`，以后只复查这些区间。
**换一版参考就是换一套标准**，所以文件名带参考名，不会串味。
默认**整谱求解再按区间筛**（切片上下文短，结果可能与整谱不同：
实测某个谱逐区间切片 20 处、整谱 12 处）；要快用 `--fast`。

> 参考谱**不是标准答案**（拆谱本来就没有唯一解），所以看的是
> "重不重复出现同一类问题"，而不是追零。

---

## 9. 已知技术债与可清理项

按"改动风险"排序，每条给出位置与影响判断。

### 9.0 六轨模式：已实现，但有两处有意的近似

`enwidenlanes` / `enwidencamera` 的刻度换算**已经实现**（见 §4.1b），
`rivenpilgrim_3` 的 6k 区间会正确按 `(2/3, 1/1.61)` 换算。遗留两点：

1. **过渡过程不建模**：淡入/淡出期间（实测 1333ms / 5333ms）刻度实际在插值，
   本实现按事件时刻瞬时切换。这是用户明确要求的简化
   （"扩展过程中的坐标变换太复杂，不需要计算"）。
2. **起手位仍按 4k 刻度算**：只在"第一个物件之前"起作用，
   而六轨段极少出现在开头，故未跟着改（§4.1b 第 5 条）。

两条都影响极小，但**如果以后要标定 6k 段落的手感，先确认这两点的量级**。

### 9.1 确定是死代码（删除不改变行为）

| 位置 | 说明 |
| --- | --- |
| `chart.py:57` `Note.params_raw` | 声明后**从未被读写** |
| `chart.py:59` `Note.side_token` | 同上 |
| `chart.py:103` `Chart.raw_meta` | 同上 |
| `split.py:1190-1191` `_CROSS_MUTUAL = _CROSS_STILL = 1.0` | 两个倍率相等，`hand_cross_cost` 里的 `factor` 恒为 1.0。公式还在、注释还在，但**分类没有实际效果** |
| `split.py:1130` `_LEVEL_EPS` | 用在 `_multi_finger_hands` 的 `abs(spot[1]-y) > _LEVEL_EPS` 过滤，但 `y` 已是 `round(_, 6)` 的分组键，弧采样点也来自同一批坐标 —— 实践上几乎不触发 |
| `config.py:142` `bullet_stream_tol` | 默认 0 → `is_bullet` 里的规整性分支不进入。注释自己写明"实测**没能区分**" |
| `config.py:73` `physics.hold_tail_relax` | 默认 0 → `duration - 0` 是 no-op。机制保留（注释解释了为什么回到 0） |

### 9.2 默认值使其成为 no-op（有价值但当前不生效）

| 位置 | 默认 | 说明 |
| --- | --- | --- |
| `config.py:236` `space.reach` | 0.0 | `bias_cost` 里 `if self.space.reach > 0` 整段不执行。**"出张扣分"当前是关闭的**（git 历史里 f2ac1fb 把它留成可调项、默认关） |
| `config.py:149` `bullet_hardness` | 0.0 | **会让 bullet 的 hardness 完全归零**。实测同一对孤立的同键键（间隔 120ms）：`0.0 → 0.000`、`0.5 → 0.195`、`1.0 → 0.390`。而密集段里的非 bullet 同键连按（87ms）是 `0.734` |
| `config.py:216` `space.weight` | 0.1 | 生效但很轻；真正的空间主力是 `space.home = 0.25` |
| `config.py:229` `space.lane_reverse` | 2.0 | **生效且很强**（实测反手代价 2.31） |

> `bullet_hardness = 0.0` 这一条值得单独确认：它和 docstring 的口径
> （"不打折就始终被对侧手压住"）方向相反。当前默认下 bullet 的 hardness
> 被完全消掉（0.000），与 `jack` 侧"bullet 按 `jack_time = 0.2` 折价"不对称 ——
> bullet 在 jack 侧只打两折，在 hardness 侧却全免。
> 不确定这是有意还是标定残留 —— 改之前先跑 `diff_report` 看真实谱面。

### 9.3 `space.window` 只喂报告，不参与决策

`SpaceConfig.window = 1200` 只在 `Scoring._build_centers` 里用，
而 `centers` / `bias` **只被 `--explain` 与 `report.explain` 读取**，
`bias_cost` 用的是 `self.bias[index]`（同一份数据）。
所以调 `window` 会改变**解释文本**和 `weight` 那一路的输入 ——
但 `weight=0.1` 很轻，实际影响很小。**不要以为它是主要旋钮。**

### 9.4 `_enforce_hold_tail` 与 `_feasible_region` 的尾指检查重复

`_enforce_hold_tail` 在最后统一强制；`_feasible_region` / `_realizations` /
`_expand` 也各自检查。三处口径必须一致（都用 `±5ms` + `_same_pos`）。
改一处就要改三处 —— 这是有意的冗余（"不值得赌被绕过"），
但**新增检查时不要漏**。

> 注意：这是**尾指同手**那条规则的冗余。它的兄弟规则"**长条占手**"（区间内
> 不得同手）**刻意不做冗余** —— `_refine` 已被 `_hold_extended_stop` 堵死
> （§2.4），再加一个事后收口 pass 就是死代码了。两条规则的处理方式不同，
> 别照搬。

### 9.4b 长条尾指同刻：`>=` 不是 `>`

`move_cost` 里"手被长条占住"的判据是 `tail >= second.time`（**含同刻**）。

尾指同刻只有两种可能，都不是"移动"：

- **原地**（`arc` 尾原地 tap 那种）：手就在那儿，松手即按下 —— 合法且免费，
  `hold_tail_hand` 正要求它同手；
- **别处**：手要同时在两处 —— 不合理，由 `hold_conflict_cost` 收 `busy`。

曾经写成 `tail > second.time`，于是**恰好同刻**时不返回 0，落到下面
"释放点压在键上 → `return 40.0`" 的分支。那个分支原本是给**音弧**准备的
（弧尾必须跟到，同刻确实做不到），对 hold 却白收 40.0，而且和 `busy` **重复收费**
（同手 total 64 而非 20）。实测 `deinosphainin_3` 因此虚高 80.0。

> 症状很难自己发现：那 40.0 是**有界的软惩罚**，不会让解非法、也不会报无解，
> 只是让求解器在这些点上偏向错误答案。`-1/-2ms` 反而正常（它们在更上面的
> `tail > time` 就返回 0 了），**只有恰好同刻**这一个点例外 ——
> 偏移 −2/−1/0/+1/+2 里，只有 0 是坏的。
> `HoldTailInstantTest` 盯着这条（把 `>=` 改回 `>` 会有 4 条测试变红）。

改这里时注意 `_conflict_holds`（line 260 附近）用的是**另一套**判据，
它已经正确地把"结尾同刻、落在**别的位置**"算作冲突、
把"尾指本应对齐的键"排除在外（靠 `_same_pos`）。两边口径要一起看。

### 9.5 报告口径（已修正，勿回退）

**两条口径，改动 `report.py` 前必读：**

**(1) 统计必须含天空物件。** 曾经 `--stats` 只统计地面（`parsed.ground_notes()`），
再把"含天空"另算一张表当脚注 —— 那是错的：

- arctap 的手速同样是真实负担，**而且往往才是最吃紧的部分**
  （实测同一份谱面，天空最大手速比地面高 50%+，见 `report.py` 的
  `test_stats_cover_all_notes_not_just_ground`）；
- 把瓶颈放进脚注，读者会以为"超过舒适移速 0"就等于整份谱很轻松。

现在 `analyze()` 一律吃**全部** `parsed.chart.notes`，`_report_lines` 只有一张表。

**(2) "手速"和"位移速度"是两回事，不要混。** 玩家口语里的手速 =
**同手相邻两键的时间间隔**（与位移无关），所以报告里分成两块：

| 报告项 | 含义 | 数据结构 |
| --- | --- | --- |
| 最吃位移速度的几处 | 位移 ÷ 可用时间（轨道/秒） | `Stats.hard_moves` / `MoveRecord` |
| 手速榜 | 同手相邻两键的最小间隔（ms） | `Stats.fastest_gaps` / `GapRecord` |

手速榜的判定口径（`_fastest_gaps`，**改之前先读这段**）：

- 只收 `SPEED_NOTE_TYPES = ("tap", "hold", "arctap")` ——
  **hold 用它的「头」时间**，间隔就是"头 → 下一个键"，
  **不要改成"尾 → 下一个键"**（用户明确否掉了那个口径，
  `test_hold_head_participates_but_tail_is_not_a_key` 盯着这条）；
- 音弧不是"落一次"，天然不参与（实谱里 arc 本来就不是 `Note`）；
- 同刻双押 `gap == 0`，**不进榜**（那是同时按下去，不是两次落下）；
- "同手相邻"按**手**分组取相邻，中间隔了另一只手的键**不影响**
  （`test_other_hand_between_does_not_break_adjacency`）。

`_fastest_gaps(limit=None)` **默认返回全量**：曾经默认截到 10 条，
结果把 27 组的并列报成 10 组。**并列组数必须来自全量列表**，
只在显示时截前几名（`_rank_gaps(...)[:5]`）。

节奏换算在 `rhythm_label()`：先 8 分（`BPM = 30000/间隔`），
等效 BPM > 300 才改说 16 分（`BPM = 15000/间隔`），**再高也不换 32 分**。
两个公式的分母不同（30000 vs 15000），写错会把 87ms 说成 BPM 690 而不是 172。

**(3) 距离一律用"轨道"，不用像素。** `1 轨道 = 0.5 坐标 = 相邻轨道间距`。
像素随设备分辨率变化、玩家毫无体感；轨道数是谱面上能直接数的。
换算入口是 `report.lanes(distance_px, physics, unit_px)`。

两个**必须注意**的点：

- **纵向不算轨道。** 轨道是横向概念。`MoveRecord.describe` 因此把位移
  **拆成横向/纵向两个分量**分别报，**不报勾股值** ——
  勾股值会让"跨 1.57 轨道"看不出是横向还是纵向（这是踩过的坑）。
  手速仍用勾股（那才是手指真实移动长度），单位是轨道/秒。
- **`unit_px` 必须传当段刻度。** 六轨段落的像素刻度是 2/3，
  若用全局 `physics.pixel_per_unit_x` 换算，**同样的几何位移会被报成不同轨道数**
  （`test_report_units_follow_six_lane_scale` 盯着这条）。
  所以 `MoveRecord` 逐条记录 `unit_px_x`，`analyze()` 汇总时也按每条自己的刻度算。

**(4) 位置一律用「时刻」，不用行号。** `MoveRecord` / `GapRecord` / `Spot`
都带 ms 时刻（`describe()` 输出 `101808-101967` 这种）。行号只在编辑器里有意义，
玩家是按时间找位置的。`MoveRecord` 仍保留 `from_line`/`to_line`
（调试时有用），但**默认不打印**。

**(5) 排行榜与占手判据都要遵守 `SAME_HAND_MARGIN_MS`（= 2ms）。**
谱面里常见 1ms 错位和弦（如 53051 的 arctap 配 53052 的 hold），
求解器已把它当**同一个和弦时刻**处理，报告必须一致，否则会出现假读数：

| 位置 | 曾经的假读数 | 现在的判据 |
| --- | --- | --- |
| 手速榜 | `BPM 15000` | `gap > SAME_HAND_MARGIN_MS` |
| 位移榜 | `1216 轨道/秒` | `delta_t > SAME_HAND_MARGIN_MS` |
| 长条占手 | 1ms 重叠也报冲突 | `duration > delta_t + SAME_HAND_MARGIN_MS` |

这三个都是**先有数据、后发现的既有问题**（不是引入时就有），
`ChordMarginTest` / `test_hold_conflict_ignores_margin` 盯着它们。

> 注意"长条占手"的 `delta_t` 是**长条头 → 下一个键**，不是"尾 → 下一个键"。
> 写文案时别说成"N ms 后就要按键"（那会被读成从尾算），
> 要说清**长条区间**与**下一个键的时刻**，以及**重叠了多少**。

**(6) 注意点要能定位。** `Stats.warnings` 是 `list[Warning]`，
每条可带 `spots: list[Spot]`（最多显示 `WARNING_SPOT_LIMIT = 5` 处，
其余折成"还有 N 处未列出"）。只报计数的警告读者无法回查。
求解器给的 `result.warnings` 仍是纯字符串，`aff_cli` 渲染时用
`hasattr(note, "lines")` 区分两者。

`_report_lines` 里有一处局部 import（`from dataclasses import replace`）
在 `_replace_sides` 内部 —— 无害但可以提到文件顶部。

### 9.6 其他小项

- `split.py:965` 附近有一个**重复的 docstring**（`hardness` 字段在
  `config.py:167-173` 有两段字符串，第二段是孤儿表达式、不构成 docstring）。
- `Scoring.hold_until`（`split.py:169`）是**列表**，与 `_State.hold_until`
  （**字典**）同名不同义。读代码时注意区分。
- `report.py` 里的 `Stats.M` / `hold_blocked` / `same_hand_chord` 等字段
  仍为历史格式留着；当前输出里 `M` 恒为 0（`allow_m` 已删）。

---

## 10. 性能

| 规模 | 耗时 | 备注 |
| --- | --- | --- |
| 92 物件 | ~0.17s | |
| 400 物件 | ~3s | |
| 608 物件（`soundwitch_2`） | ~2.5s | 纯 Python |
| 664 物件（`breachoffaith_3`） | ~7s | |
| 778 物件（`halcyon_2`） | ~6s | |
| 800 物件（`rivenpilgrim_3`，含 6k 段） | ~13s | 有 1 个六轨区间 |
| 1216 物件（`gengaozo_2`） | ~9s | 物件最多但反而不慢 |

**瓶颈在局部优化**（每次改进都重新扫邻域）。控制旋钮：

- `search.refine_passes`（6）/ `refine_radius`（3）—— 调小最快；
- `search.refine_mirror`（True）—— 关掉省一批候选；
- `search.beam_width`（56）—— 影响搜索而非局部优化。

已经做过的优化：`PrefixCache` 惰性前缀、`delta` 只重扫受影响段、
`base_dense` 用 `list()` 的 C 级拷贝。
**多指段定死手标记也会显著加速**（搜索空间变小，实测某个裁切片段 7s → 1.5s）。

---

## 11. 扩展指引

- **加新的代价项**：在 `Scoring` 里写一个纯方法 → 在 `_segment_sum` 里调用
  → 在 `ScoreParts` 加字段 → 在 `delta` 里加一行增量 → 在 `_expand` 里同步
  （**否则束搜索与 `evaluate` 会不一致**）→ 在 `_neighbours` 的可行性检查里
  考虑是否需要新约束。
- **加新的硬约束**：优先做成 `constraints` 对象的一个方法 + `can_hand` 的
  一条判据，这样束搜索、局部优化、候选生成**全都自动尊重它**。
  如果约束依赖"已定的手"，就要像 `hold_conflict` 那样额外在
  `_realizations` / `_expand` / `_feasible_region` 三处检查。
- **换音游**：要改的是整套模型（坐标刻度、速度阈值、代价权重、约束规则），
  不是调几个数。最小改动路径是新增一个像 `constraints.py` 那样的模块 +
  一个新 CLI，让 `split.py` 保持不变 —— **它的分层就是为这个准备的**。
  注意 `_scan_multi_finger` / `tail_hint` 会对 `arcs_by_hand` 做
  `getattr(..., None)`，没有弧的音游自然走不到那些分支。
- **加测试**：`py/tests/test_aff.py` 里的 `TINY_AFF` 是自足的最小谱面，
  新用例优先扩它。凡是"规则性"的东西（不是权重问题）都要有回归测试 ——
  这类 bug 的形态是"根本没参与计算"，靠调参发现不了。

---

## 12. 开发流程建议

1. **改之前**先跑一次 `unittest discover`，确认是 **112 条 / 0 skip** 全绿，
   记下当前状态再动手（自测全部自足，不需要任何外部谱面）。
2. 用小谱面（`TINY_AFF` / 直接构造 `Note`）构造**最小复现**，
   再用真实谱面确认没有回归。
3. 用 `compare.py --set` 做 A/B，而不是改默认值来回试。
4. 改完跑 §8.2 的五条不变式。
5. 提交信息写成"现象 → 根因 → 修法"，git 历史里现有 40 多条都是这个风格，
   对后来人（和 AI）非常有用。
