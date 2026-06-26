# 检测器 × LeRobot 数据集：改造影响评估

本文评估在引入数据集**多模态信息**（见 `docs/dataset_inputs.md`）后，现有已实现的检测器
（见 `docs/detectors.md`）哪些需要**大变更**、哪些**小优化**、哪些**不用变**，并列出新数据
顺带解锁的「新增检测点」。

> 背景：现有检测器基本只吃「ffprobe 元数据 + 视频抽帧」。而每组 LeRobot v3.0 数据实际提供
> 逐帧机器人状态（parquet）、遥操作主控信号、多层时间戳、统计量、相机标定、中文子任务标注等，
> 可做跨模态、跨字段的一致性与质量校验。

---

## 0. 结论速览

| 类别 | 数量 | 成员 |
| --- | --- | --- |
| 前置枢纽改造（摄入 / 编排层） | 1 | 数据摄入层（非检测器，但是一切增益的前提） |
| 检测器需**大变更** | 3 | `static`、`regrasp`、`object_slip` |
| 检测器可**小优化** ✅ 已实现 | 6 | `metadata`、`jump`、`dup_frame`、`endpoint_static`、`freeze`、`brightness` |
| 检测器**不用变** | 5 | `integrity`、`visual`、`noise`、`colormatch`、`gripper_offscreen` |

**一句话**：真正的「大变更」在**数据摄入 / 编排层**，不是某个检测器；它一旦落地，多数检测器
只需小优化甚至零改动即可吃上新信号。

---

## 1. 前置：真正的「大变更」在数据摄入 / 编排层

**现状**为「单视频文件」粒度：

- `agent.py` 用 `discover_videos` glob 出 `.mp4`，逐文件 `inspect_video()`。
- `pipeline._extract_metadata` **只填 ffprobe 字段**（width/height/fps/duration/codec/...）。
- `metadata["robot"]`、`metadata["target_objects"]`、`metadata["task"]` 是已埋好的钩子，但
  **至今为空**（`jump` 因此恒落 `__default__` 阈值，`colormatch` 的 task_hint 恒为空串）。

**目标**改为「LeRobot 组」粒度：

1. 遍历定位到 `.../success/lerobot_RoboMIND/`，**先读 `meta/info.json`** 取本组真实参数。
2. 读 parquet / `meta/*` / `labels/labels.json`，把以下信息灌入 `metadata`（或扩展 `check()`
   入参以便检测器拿到逐帧数组）：
   - `robot`（来自 `info.json.robot_type`）
   - 声明视频规格（codec / 分辨率 / fps / pix_fmt / has_audio）
   - `task` / `target_objects`（来自 `tasks.jsonl`、`labels.json` 子任务名、`language_instruction`）
   - 逐帧关节与夹爪数组（`puppet.*` / `master.*`）、时间戳、`is_intervene`、`frame_index`
   - episode 长度、相机标定（内参 / 外参 / 畸变，**475/500 组有**，须容忍缺失降级）

> 这是一次性枢纽改造，**绝大多数检测器的增益都依赖它先落地**。落地后 `jump.robot` 与
> `colormatch` 的 task/target_objects 钩子无需改检测逻辑即自动生效。

**实现注意**（摘自 `dataset_inputs.md §17`）：

- 不要硬编码 fps / 编码 / episode 数 / 是否有标定 —— 全部因组而异，逐组读取。
- 实际可检测 episode 数 = `data/chunk-000/` 下 parquet 文件数，**不是** `total_episodes`。
- 文件名按真实 `episode_{编号}.*` 遍历（模板写的 `file-{index}` 与实际不符）。
- 优雅降级：标定缺失（25/500）、`key_frame` 为空、某些机型无 `head_position` 等都要兼容。

---

## 2. 需大变更的检测器（3）

共同点：**关键判定信号本就在 parquet 里**，用真实信号替代「从像素推断」可根治已知漏判 / 大幅降本。

### 2.1 `static`（机械臂静止 / 无效操作）✅ 已实现

- **当前**：像素帧差（lite）或 RAFT 光流判整段是否静止。
- **变更**：增加 **parquet 关节运动后端**——直接用 `puppet.arm_*_position_align` 逐帧方差 / 速度
  判静止。
- **为什么是大变更**：`detectors.md` 自陈「三件套均基于全画面 mean|ΔY|，机械臂在大片静止背景中
  只占小块时帧差信号弱，可能漏判——算法层面的固有局限」。关节是运动的**地面真值**，可彻底绕开该局限。
- **实现**（`config.static.backend: joint`）：`static.evaluate_joint_static`（纯函数：
  `max_range = max_j(q99_j-q01_j) < joint_range_thr` 判静止，稳健抗单帧跳变）+ 复用 §1 摄入层
  `vidinspect_agent/lerobot.py`（`metadata["lerobot"]["parquet_path"]` 指针 / `load_episode_frames`
  读 puppet 臂关节；未经摄入时由视频路径自定位）。parquet/pyarrow 缺失时按 `joint_fallback`
  （默认 `lite`）优雅回退。详见 `detectors.md §3.1`。

### 2.2 `regrasp`（二次抓取）✅ 已实现

- **当前**：多模态逐帧判 holding + 代码侧逐臂状态机。
- **变更**：夹爪开合（`puppet.end_effector_*_position_align`）已在 parquet ——抓取 / 释放的
  **时序**用真实信号驱动。逐臂判定是单目标、忽略物体身份，故「夹爪闭合段数」即「抓取次数」，
  **parquet 路径完全不调模型**（无需 API key、零付费调用、用全分辨率时间轴）。
- **收益**：去抖 / 持有段计数落到真实信号上更稳，且可**大幅降低付费多模态调用**。
- **落地**（**已对齐 §1 摄入 / 编排层枢纽**）：`RegraspChecker.check` 改为 **parquet 优先**：
  优先用摄入层（`GroupResolver`）注入到 `metadata["lerobot"]["parquet_path"]` 的 parquet 指针
  （单一数据源，与 `static` / `object_slip` 一致），经 `_lerobot.gripper_opening_from_metadata`
  读各侧开合；未经摄入层注入时才 `find_episode_parquet` 自定位兜底。
  用 episode 自身区间相对归一化阈值化为逐帧「闭合」（复用 §2.3 的 `opening_to_closed`），
  夹爪闭合即持有 → 复用 `detect_regrasp(single_object=True)` 逐臂计数。新增 `regrasp` 配置项
  `use_parquet_gripper` / `gripper_closed_is_low` / `gripper_closed_frac` / `gripper_min_span`。
  **优雅降级**：配置关闭 / 非 LeRobot 布局 / 缺 parquet / 缺 pyarrow / 缺视频 fps / 开合区间
  过小不可判 → 自动回退到原多模态逐帧路径（行为同改造前），`details.source` 标注实际来源。

### 2.3 `object_slip`（物体滑落）✅ 已实现

- **当前**：多模态逐帧判 holding + `gripper_closed`。
- **变更**：关键信号 `gripper_closed` 直接取 parquet 夹爪开合值，不再靠模型推断。
- **收益**：「持有结束时夹爪是否仍闭合」（滑落 vs 主动放下的唯一判据）落到真实信号上，判定更可靠。
- **落地**（见 `detectors.md §3ter.3`，**已对齐 §1 摄入 / 编排层枢纽**）：优先用摄入层
  （`GroupResolver`）注入到 `metadata["lerobot"]["parquet_path"]` 的 parquet 指针（单一数据源），
  经 `_lerobot.gripper_opening_from_metadata` → `lerobot.load_episode_frames` 读逐帧开合；
  未经摄入层注入时（直接调用 pipeline）才 `find_episode_parquet` 自定位兜底。读出的开合用
  episode 自身区间相对归一化阈值化为逐帧「闭合 / 张开」并对齐采样帧；模型仅保留 holding。
  `checkers/_lerobot.py`（夹爪信号 → 闭合序列）、`object_slip` 配置项 `use_parquet_gripper` /
  `gripper_closed_is_low` / `gripper_closed_frac` / `gripper_min_span`，及可选依赖
  `pip install -e ".[lerobot]"`（pyarrow）。**逐臂优雅降级**：parquet 不可用时回退模型推断的
  `gripper_closed`，`details.gripper_source` 标注每臂实际来源。

---

## 3. 可小优化的检测器（6）✅ 已实现

像素侧核心逻辑保留，叠加新数据作为更权威输入或交叉验证。

| 检测器 | 当前实现 | 小优化内容 | 状态 |
| --- | --- | --- | --- |
| `metadata` | ffprobe 读规格 → 阈值比较 | 增加「声明 vs 实测」：`info.json` 的 codec / 分辨率 / fps / pix_fmt / has_audio 与 ffprobe 实测交叉核对 | ✅ |
| `jump` | 像素局部峰值 + 四重守卫，按 `robot` 查阈值表 | 摄入层填 `metadata["robot"]=robot_type`，阈值表才真正生效（现恒落 `__default__=4.0`，等于没分机型）；可选用关节跳变交叉验证 | ✅ |
| `dup_frame` | 像素复制帧比例 + fps 归一化 | fps 归一化改用 `info.json` 声明帧率（更权威；全量 29/30/28 因组而异）；时间戳连续性另起新检查 | ✅（fps 部分；时间戳连续性见 §5） |
| `endpoint_static` | 像素首 / 尾连续静止 + 自适应阈值 | 用关节速度≈0 或 `labels` 的「归位」子任务段交叉验证首尾停留，减少对 64×48 像素自适应阈值的脆弱依赖 | ✅ |
| `freeze` | 像素单段最长冻结时长 | 与关节运动交叉判别：腕部相机（`camera_left/right`）随臂运动，画面冻结但关节在动时把「关节仍在动」作为补充信号写入 details/文案（规范 18 判定归属新检测器 `frame_consistency`，见 §5；`freeze` 仍专司规范 5 长冻结） | ✅ |
| `brightness` | 像素平均亮度中位数 vs 固定阈值 40 | 用 `stats.json` 像素均值作每数据源基线，替代写死阈值（亦可按组标定） | ✅ |

**落地说明**（全部对非 LeRobot 视频 / 缺 parquet / 缺 pyarrow **优雅降级**回原纯像素行为，
并把信号来源写入 `details` 供复核；新增 `checkers/_joints.py` 作为关节运动量共享 helper，
复用 §1/§2 已有的 parquet 读取）：

- **摄入层补充注入**（`lerobot.py`，均为 JSON 可序列化、附加键）：从 `stats.json` 取每相机
  像素均值注入 `metadata["lerobot"]["pixel_luma_baseline"]`（0–255）、把含帧区间的子任务
  注入 `metadata["lerobot"]["subtasks"]`；`pipeline._extract_metadata` 增补 `pix_fmt` / `has_audio` 实测值。
- `metadata`：新增 `spec_match` 结果项，比对声明视频规格与 ffprobe 实测（fps 容差 `fps_tol`，
  默认 1.0；不一致默认 WARN）。纯视频无声明值 → 不产生该项。配置 `metadata.spec_match`。
- `jump`：`robot` 已由 §1 摄入层自动填入，阈值表按机型生效（无需改判定逻辑）。新增**可选**
  关节交叉验证（`jump.joint_cross_validate`，默认关闭）：像素峰值帧处关节也跳变才确认，
  抑制编码 / 光照伪跳变；`details.joint_peak_ratio` 记录峰值帧关节速度 / 中位数之比。
- `dup_frame`：fps 归一化优先用 `info.json` 声明帧率（`dup_frame.prefer_declared_fps`，默认 true），
  `details.fps_source ∈ {declared, measured, probed}`。
- `endpoint_static`：有关节数据时以**关节首尾静止时长**为准（关节是归位静止的地面真值），
  绕开 64×48 像素自适应阈值的脆弱性；末尾「归位 / 复位」子任务段记入 `details.trailing_homing_subtask`。
  配置 `endpoint_static.joint_cross_validate` / `joint_move_speed`，像素值仍保留在 `details` 对照。
- `freeze`：腕部相机（`camera_left/right`）画面冻结但对应臂关节在动 → `details.frame_joint_inconsistent=true`
  并在命中文案中补充「期间对应臂关节仍在运动」；规范 18 的判定归属新检测器 `frame_consistency`（见 §5），
  `freeze` 本身仍报规范 5（单段长冻结）。俯视 / 头部固定机位用整体关节运动。
  配置 `freeze.joint_cross_validate` / `joint_move_speed`。
- `brightness`：有 `stats.json` 基线时阈值 = `baseline × baseline_rel_frac`（默认 0.4）替代写死的 40，
  `details.threshold_source ∈ {stats_baseline, fixed}`。配置 `brightness.use_stats_baseline` / `baseline_rel_frac`。

> 真机验证：在真实组（`tienyi_prod2_dualArm-dexHand`，1280×720/h264/29fps）上跑通——`spec_match`
> 声明=实测、`dup_frame` 用声明 29fps、`endpoint_static` 走关节信号、`freeze` 识别 `camera_left`
> 为腕部相机、`brightness` 用 stats 基线（106.4 → 阈值 42.55）、`jump` 关节峰值比 23.2。

---

## 4. 不用变的检测器（5）

| 检测器 | 当前实现 | 为什么不用变 |
| --- | --- | --- |
| `integrity` | ffmpeg 全解码校验坏文件 | 坏文件守门核心不变；新数据另解锁独立的「三方帧数一致性」新检查，不改本检测器 |
| `visual` | ffmpeg blackdetect 黑屏比 | 纯像素黑屏，metadata 无实质增益 |
| `noise` | Immerkær 噪声方差估计 | 纯空间启发式，无跨模态增益（`camera_model` 可作记录但不影响判定） |
| `colormatch` | 多模态静态属性（物体 / 桌面同色） | **代码无需改**：`task_hint` / `target_objects` 钩子已就绪，摄入层把 labels/tasks 填进 metadata 即自动生效（见 `detectors.md §3ter.4`） |
| `gripper_offscreen` | 多模态逐帧判夹爪是否在画面内 | **代码无需改**（自动用 robot hint）；可选机会：用标定（内参 / 外参）+ 末端位姿做几何投影，省掉付费多模态 |

---

## 5. 顺带解锁的「新增检测点」（属新增，不是改造现有器）

这些是 parquet / meta / labels 带来的全新能力，对应 `数据质检规范汇总.md` 里大量 🟡 待定项。
建议作为**新检测器**实现，而不是塞进现有像素检测器。

| 新增检测点 | 用到的新数据 / 覆盖的待定规范 |
| --- | --- |
| 画面/关节一致性 ✅ 已实现（`frame_consistency`） | parquet 臂关节（"该臂动没动"地面真值）+ 腕部相机帧差（"画面动没动"）→ 规范 18 画面保持一致；无需 AI，详见 `detectors.md §3bis.5` |
| 三方帧数一致性 | `episodes.jsonl.length` = parquet 行数 = mp4 `nb_frames`（高价值低成本） |
| schema / dtype 校验 | parquet 实际 dtype/shape vs `info.json.features`（金标准）；特征齐全 / 多余 |
| 关节合理性 | 超物理范围 / 相邻帧跳变抖动 / 轨迹不平滑 → 规范 7 joint 异常、规范 2 动作不流畅 |
| 主从一致性 | master vs puppet 关节差异 / 滞后、是否真人遥操作、控制延迟估计 |
| 时间戳 / 采样率 | 采样间隔≈1/fps、丢帧 / 时钟回退、各传感源对齐、视频区间↔状态时间轴吻合 |
| `is_intervene` 纯净度 | 逐帧 / 逐特征人工干预比例统计、校验「声称无干预」是否全 0 |
| 标定齐全性 | 内参 / 外参 / 畸变齐全（475/500 有，须容忍缺失降级）、内参合理性 |
| labels 合规 | 帧号越界（`end_frame < length`）、区间单调无重叠、每 episode 有标注、含「归位」收尾 |
| 任务一致性 | `task_index`↔`tasks.jsonl` 映射、与 `info.json.total_tasks` / `language_instruction` 一致 |

---

## 6. 落地顺序建议

1. ✅ **摄入 / 编排层枢纽改造**（解锁一切，见 §1）。
2. ✅ 接上 `jump.robot` 与 `colormatch` 的 task/target_objects 钩子（**零检测逻辑改动即生效**）。
3. 加几条「低成本高价值」新检查：三方帧数一致性、schema 校验、时间戳 / 采样率（见 §5）。
4. ✅ 给 `static` 加关节后端；用 parquet 夹爪信号重做 `regrasp` / `object_slip`（见 §2）。
5. ✅ 6 个检测器小优化（声明 vs 实测 / 声明 fps / 关节首尾静止 / 画面-关节一致性 / stats 亮度基线，见 §3）。
