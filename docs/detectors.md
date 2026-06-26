# VidInspect 检测器原理设计文档

本文面向开发者与维护者，逐个说明 VidInspect Agent 中各检测器的算法原理、关键公式、
阈值与判定逻辑，以及优雅降级策略。使用与配置入门请参见根目录 `README.md`。

---

## 1. 总体架构

### 1.1 数据流

```
视频文件
   │
   ├─ ffprobe 探测 → metadata（width/height/fps/duration/codec/...）
   │
   ▼
_build_checkers(config) 按 config["checks"] 开关装配检测器流水线
   │
   ▼
每个 Checker.check(path, metadata) → list[CheckResult]
   │
   ▼
VideoReport(passed = 没有任何 FAIL, results, metadata)
   │
   ▼
InspectionSummary（批量汇总 total / passed / failed）
```

入口编排见 `src/vidinspect_agent/pipeline.py` 的 `inspect_video()`，批量调度见
`src/vidinspect_agent/agent.py` 的 `VidInspectAgent.inspect_paths()`。

### 1.2 检测器接口

所有检测器继承 `BaseChecker`（`checkers/base.py`），实现单一方法：

```python
def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
    ...
```

`config` 在构造时注入，检测器自取所需配置块。

### 1.3 结果模型与严重级别

`CheckResult`（`models.py`）字段：`name` / `severity` / `message` / `details`。

`Severity` 三档：

| 级别 | 含义 | 是否影响整体 `passed` |
| --- | --- | --- |
| `PASS` | 通过 | 否 |
| `WARN` | 可疑 / 无法评估 / 依赖缺失降级 | **否** |
| `FAIL` | 判定不合格 | **是** |

整体判定：`passed = 不存在任何 severity == FAIL 的结果`（`pipeline._report_failed`）。

> 设计要点：三个时序检测器（static / dup_frame / jump）默认命中报 **WARN**，
> 不会让视频直接 `passed=False`，与黑屏检测一致。需要硬失败时把对应配置块的
> `severity` 改为 `fail`。

### 1.4 装配开关

`config["checks"]` 逐项控制是否启用。装配顺序：
`integrity → metadata → visual → static → dup_frame → jump → endpoint_static → freeze → noise → brightness → gripper_offscreen → regrasp → object_slip → colormatch → occlusion`。
除 `gripper_offscreen`（规范12）/ `regrasp`（规范1）/ `object_slip`（规范21）/ `colormatch`（规范19）/ `occlusion`（规范15）五项付费多模态远程调用**默认关闭**外，其余默认全开。

### 1.5 优雅降级原则

任何检测器遇到「依赖缺失 / 读帧失败 / 超时 / 解码异常」都返回 **WARN** 而非抛异常，
确保单个视频或单个检测器的问题不会拖垮整条流水线。`probe` 阶段失败是唯一例外：
直接返回 `passed=False`（因为后续检测都依赖元数据）。

---

## 2. 基础质检检测器

### 2.1 MetadataChecker（元数据校验）

- **源文件**：`checkers/metadata.py`
- **原理**：用 `ffprobe -show_format -show_streams` 读取元信息，对照 `thresholds` 逐项阈值比较，**无像素级计算**。
- **检查项与判定**：

| 子项 | 判定 | 默认阈值 | 命中级别 |
| --- | --- | --- | --- |
| `resolution` | `width < min_width` 或 `height < min_height` | 640 × 480 | FAIL |
| `fps` | `fps < min_fps` 或 `fps > max_fps` | 15 – 120 | FAIL |
| `duration` | `duration < min_duration_sec` | 0.5 s | FAIL |

- **无法读取时**：分辨率缺失 → FAIL；帧率/时长缺失 → WARN。

### 2.2 IntegrityChecker（完整性 / 可解码性）

- **源文件**：`checkers/integrity.py`
- **原理**：三层递进检查，最后一层做真实解码。
  1. 文件存在性：`path.is_file()` 为假 → FAIL（`file_exists`）。
  2. 文件大小：`st_size == 0` → FAIL（`file_size`）。
  3. 解码验证：执行 `ffmpeg -v error -i <in> -f null -`，把整段视频解码后丢弃，
     只观察 stderr。**只要 ffmpeg 返回非零退出码或 stderr 有任何 error 输出**，
     即判定存在损坏帧 / 解码失败 → FAIL（`decode`，截断保留前 500 字符）。

### 2.3 VisualChecker（黑屏启发式）

- **源文件**：`checkers/visual.py`
- **原理**：用 ffmpeg 滤镜 `blackdetect=d=0.1:pix_th=0.10` 检测黑屏区间，解析 stderr 中
  的 `black_start` / `black_end`，累加黑屏总时长。
- **公式**：

```
black_ratio = Σ(black_end - black_start) / duration_sec
```

- **判定**：`black_ratio > max_black_frame_ratio`（默认 0.3）→ **WARN**。
- **降级**：`duration <= 0`（缺时长）→ WARN，无法评估。

---

## 3. 时序质检检测器

> 三者均移植自 `video_quality_pipeline` 三件套，共享读帧 / 探测工具
> （`checkers/_frames.py`：`probe_fps`、`_sample_indices`、`read_frames_gray`、
> `read_frames_rgb_tensor`）。所有 `score` 语义统一为 **越高越好**。

### 3.1 StaticChecker（机械臂静止 / 无效操作）

- **源文件**：`checkers/static.py`、`checkers/_raft.py`（joint 后端复用摄入层 `vidinspect_agent/lerobot.py`）
- **目标**：检出整段几乎无运动（机械臂静止、无效操作）的视频。
- **后端选择**：`config["static"]["backend"]`，`lite`（默认，纯 CPU）、`raft`（GPU 光流）
  或 `joint`（LeRobot parquet 关节地面真值）。

#### lite 后端（纯 CPU，默认）

1. 均匀采样 `n_frames`（默认 32）帧，下采样到 `max_h`（默认 240）灰度，得 `[T, H, W]`。
2. 相邻帧 L1 帧差序列：`diffs = mean(|gray[i] - gray[i-1]|)`，取峰值 `peak_diff = diffs.max()`。
3. sigmoid 软化为 motion_score：

```
x     = peak_diff / 1.5
score = 1 / (1 + exp(-5 * (x - 1)))
```

4. **判定**：`score < thr`（默认 0.30）→ 静止（命中 `severity`，默认 WARN）。

#### raft 后端（GPU 稠密光流，更准）

1. 采样 `raft_n_frames`（默认 40）帧为 RGB 张量 `[T, C, H, W]`。
2. RAFT 逐相邻帧对计算稠密光流（`compute_raft_flows`，OOM 时自动减半 batch）。
3. 每帧取 top 5% 光流幅值均值，按 `min(H, W)` 归一化得 `rel_i`：

```
rad_i = sqrt(flow_x² + flow_y²)
rel_i = mean(top5%(rad_i)) / min(H, W)
active_ratio = mean( rel_i >= 0.012 )
```

4. **判定**：`active_ratio < raft_thr`（默认 0.10，已标定）→ 静止。
5. **降级**：torch / decord / RAFT 源码 / 权重缺失或无 GPU 时，捕获异常 → **WARN**，
   不影响 lite 后端与整条流水线。

> RAFT 源码与权重路径解析优先级（`_raft.resolve_repo`）：
> `config.raft_repo` > `$RAFT_DIR` > `$WORLDARENA_DIR/.../RAFT` > 内置兜底路径；
> 权重缺省取 `<repo>/models/raft-sintel.pth`。模型按 `(repo, weights, device)` 进程内缓存一次。

#### joint 后端（LeRobot parquet 关节地面真值）

像素三件套都基于「全画面 `mean|ΔY|`」，机械臂在大片静止背景里只占小块时帧差信号弱、
易漏判（算法层面的固有局限）。`joint` 后端直接用 **puppet 关节逐帧位置**这一运动地面真值
绕开该局限（对应 `docs/detector_dataset_impact.md §2.1`）。

1. **定位 parquet**：优先用 §1 摄入层注入的 `metadata["lerobot"]["parquet_path"]` 指针；
   未经摄入时退化为由视频路径经 `lerobot.find_group_root` + `parse_episode_index` +
   `load_group` 定位同 episode 的 `data/<chunk>/episode_*.parquet`（按真实 `episode_{编号}`
   匹配，非 LeRobot 布局 → 回退）。
2. **读 puppet 臂关节**：用 `lerobot.load_episode_frames` 读列，取
   `puppet.arm_left/right_position_align.data`（兼容单臂 `puppet.arm_position_align.data`），
   得每侧 `[T, 7]`（弧度）。
3. **评估**（`static.evaluate_joint_static`，纯函数、可单测）：多臂帧轴对齐到最短长度后拼接，

```
max_range  = max_j ( q99_j - q01_j )         # 每个关节整段稳健峰峰值取最大（抗单帧跳变）
peak_speed = max_t || J[t] - J[t-1] ||₂      # 相邻帧关节向量 L2 速度（rad/帧）
```

4. **判定**：`max_range < joint_range_thr`（默认 0.05 rad ≈ 2.9°）→ 静止（命中 `severity`）。
   `peak_speed` / `mean_speed` / `moving_ratio` 仅作报告。`score = min(1, max_range/thr)`，
   语义同 lite（越高越好）。
5. **优雅降级**：找不到 parquet / pyarrow 缺失 / 列结构异常 / 无任一臂数据时，
   回退到 `joint_fallback`（默认 `lite`，可 `raft` 或 `none`）；回退结果的 `details` 会带
   `joint_fallback_from` 与 `joint_fallback_reason`。`none` 则直接报 WARN。

> 依赖：`joint` 后端需要 `pyarrow`（`pip install -e ".[lerobot]"`）。未安装时按上面策略回退。

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `thr` | 0.30 | lite：motion_score 低于此判静止 |
| `raft_thr` | 0.10 | raft：active_ratio 低于此判静止 |
| `joint_range_thr` | 0.05 | joint：关节稳健峰峰值(rad)低于此判静止 |
| `joint_move_speed` | 0.005 | joint：单帧"在动"的速度阈值(rad/帧，仅报告 moving_ratio) |
| `joint_fallback` | lite | joint：parquet 不可用时回退后端（lite / raft / none） |
| `n_frames` / `raft_n_frames` | 32 / 40 | 采样帧数 |
| `max_h` | 240 | 下采样高度（两后端共用） |

### 3.2 DupFrameChecker（复制帧伪装高帧率导致的卡顿）

- **源文件**：`checkers/dup_frame.py`
- **目标**：检出靠复制帧伪装高帧率、造成卡顿 / 时间变慢的视频。
- **原理**：用 ffmpeg 把视频缩成 `downscale`（默认 64 × 48）灰度 raw 流，逐帧算
  `mean|ΔY|`，在同一序列上算两套量：
  - 宽阈值 `diff_thr`（0.5）→ `dup_mask` → `keep_ratio`（非重复比例）、连续重复段长统计
    （`mean_gap` / `max_gap` / `gap_p95` / 各段长占比）。
  - 严格阈值 `strict_thr`（0.05）→ `dup_ratio_strict`（「真复制」比例）。

```
keep_ratio       = 1 - dup_count / total_pairs
dup_ratio_strict = dup_strict_count / total_pairs
```

#### fps 归一化（关键设计）

阈值在 **20 fps** 上标定（`fps_ref`）。帧率越高，相邻帧天然越相似，若不归一化会把
高帧率静态 / 慢动作正常视频误判为「时间变慢」。归一化系数：

```
norm = fps / fps_ref
mean_gap_thr_eff   = mean_gap_thr * norm
strict_thr_eff     = ratio_strict_thr * norm^strict_fps_pow   # strict_fps_pow=1.5
mean_gap_cap_eff   = mean_gap_cap * norm
static_*_eff       = (10 / 20) * norm                          # 静止段判定也随之放缩
```

#### 判定（两条规则 OR 联合）

- **规则 A（严重卡顿）** = `periodic_stutter` 或 `static_like`
  - `periodic_stutter`：`keep_ratio < keep_ratio_thr` 且 `mean_gap >= mean_gap_thr_eff`
    且 `mid_gap_ratio >= mid_gap_ratio_thr`（周期性复制，中等长度复制段占比高）。
  - `static_like`：`keep_ratio < static_keep_ratio_thr` 且 `mean_gap` / `max_gap`
    均超过对应静止阈值（长段静止复制）。
- **规则 B（时间变慢）**：`dup_ratio_strict > strict_thr_eff` 且 `keep_ratio >= keep_ratio_lo`
  且 `mean_gap < mean_gap_cap_eff`（严格复制比超 fps 归一化阈值，且复制段不过长——连续性约束）。

```
reason  = ("A" if rule_a else "") + ("B" if rule_b else "")   # ∈ {"", "A", "B", "AB"}
problem = rule_a or rule_b
score   = keep_ratio   # 越高越好
```

- **降级**：超时（默认 60 s）、帧数 < 2、ffmpeg 异常 → **WARN**。

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `diff_thr` | 0.5 | 宽阈值（dup_mask） |
| `strict_thr` | 0.05 | 严格阈值（真复制） |
| `keep_ratio_thr` | 0.20 | 规则 A keep_ratio 上限 |
| `mean_gap_thr` | 6.0 | 规则 A 平均复制段长下限（×norm） |
| `ratio_strict_thr` | 0.095 | 规则 B 严格复制比阈值（×norm^1.5） |
| `mid_gap_ratio_thr` | 0.45 | 周期复制：中段（4–14）占比下限 |
| `static_keep_ratio_thr` | 0.09 | 长段静止复制 keep_ratio 上限 |
| `fps_ref` | 20.0 | 阈值标定基准帧率 |
| `downscale` | [64, 48] | ffmpeg 下采样尺寸 |

### 3.3 JumpChecker（跳帧 / 瞬移）

- **源文件**：`checkers/jump.py`
- **目标**：检出单点突变（跳帧、瞬移）而不误报正常的连续大幅运动。
- **原理**：逐帧读取（缩到 `MAX_H=120` 灰度），算相邻 L1 帧差 `diff_i`，核心指标是
  **局部归一化峰值**——某帧帧差除以其 ±W（`WINDOW=30`）邻域均值（排除自身）：

```
local_ratio_max = max_i  diff_i / mean( diff_{i-W..i+W} \ {i} )
score           = 1 / (1 + log(max(local_ratio_max, 1)))   # 越高越好
```

> 帧数不足 `2W+3` 时退化为全局比值 `diff.max() / diff.mean()`。

#### 四重守卫（全部满足才判 jump）

光有大尖峰不够，必须同时满足四个条件以压低误报：

| 守卫 | 条件 | 作用 |
| --- | --- | --- |
| `ratio_hit` | `local_ratio_max >= thr[robot]` | 局部突变足够显著 |
| `magnitude_ok` | `peak_abs >= peak_abs_min`（3.0）或 `global_mean <= frozen_mean`（0.05） | 绝对幅度够大，或整段几乎冻结 |
| `universal_ok` | 非持续运动（`peak_high_run < sustained_run`）且 非边界初始化伪影 | 排除连续大运动 / 开头结尾伪影 |
| `isolation_ok` | 静态相机条件下要求「单帧孤立尖峰」（`peak_isolation >= isolation_min` 且 `peak_high_run <= high_run_max`） | 排除静态相机下的非孤立波动 |

```
problem = ratio_hit and magnitude_ok and universal_ok and isolation_ok
```

#### 孤立性模式（`isolation_mode`）

- `off`：恒为 True（不启用孤立约束）。
- `all`：所有视频都要求孤立尖峰。
- `agibot`：仅 `robot` 以 `agibot` 开头时要求。
- `auto`（默认）：`static_camera` 显式指定则按其值，否则等价 `agibot` 策略。

#### 机器人阈值表

`thr` 按 `metadata["robot"]` 查表（来自 `jump_frame` 的 `filter_meta.json`），无标签用
`__default__`（4.0）。可用 `config["jump"]["thresholds"]` 覆盖。内置示例：

| robot | thr |
| --- | --- |
| `tienkung_station_dualArm-gripper` | 10.0 |
| `tienkung_pro2_dualArm-gripper` | 7.5 |
| `tienkung_pro2_dualArm` | 8.0 |
| `tiangong_dualArm` | 6.0 |
| `tiangong_dexHand` | 10.0 |
| `tienkung_sim_dualArm` / `tienyi_dualArm` | 7.5 / 8.0 |
| `tienyi_mobile_dualArm` | 5.0 |
| `__default__` | 4.0 |

- **降级**：视频打不开 → WARN；有效帧差 < 2 → PASS（帧数过少跳过）。

#### 对规范序号 2.1（机械臂异常-卡顿）的覆盖

规范序号 2.1「机械臂异常-卡顿」（序号 2「动作不流畅」的子类别）不单设检测器，由时序三件套联合覆盖，`jump` 为主判：

| 卡顿表现 | 命中检测器 | 说明 |
| --- | --- | --- |
| 卡顿后机械臂"瞬间追位"的单点突跳 / 瞬移 | `jump` | 本检测器核心场景（`local_ratio_max` 局部尖峰 + 四重守卫） |
| 周期性一顿一顿的微卡（多段短复制帧） | `dup_frame` | 规则 A `periodic_stutter` |
| 单段长时间完全卡住 | `freeze` | 单段最长冻结时长 > 2s |

> **边界**：`jump` 的 `universal_ok` 会**主动抑制持续 / 周期性运动**（`peak_high_run >= sustained_run` 即判定为持续运动而不报），故反复微抖须靠 `dup_frame` 兜底；且三件套均基于**全画面 mean|ΔY|**，当机械臂在大片静止背景中只占小块时帧差信号弱，可能漏判——这是算法层面的固有局限，非装配缺失。

---

## 3bis. 质检规范专项检测器（序号 3 / 5 / 11）

> 三者共用 `_frames.stream_gray_diffs()`：用 ffmpeg 把整段视频**按原始播放顺序**
> 解码为 `downscale`（默认 64×48）灰度 raw 流，逐帧算 `mean|ΔY|`（0–255 量纲），
> 返回相邻帧帧差序列。与时序三件套的「均匀采样」不同，这里需要**连续帧序**才能
> 定位首尾/单段的时长。`score` 同样统一为越高越好。

### 3bis.1 EndpointStaticChecker（开始/结束归位停留时间长，规范序号 3）

- **源文件**：`checkers/endpoint_static.py`
- **目标**：检出视频**开头或结尾**存在超过 2s 的静止（机械臂归位后长时间停留 / 空等），
  这是 `static`（全片均匀采样判整体静止）无法定位的首尾时段问题。
- **原理**：在帧差序列上取**开头连续静止帧数**与**结尾连续静止帧数**，按 fps 换算成秒：

```
leading_static_sec  = leading_run(diff < motion_thr_eff)  / fps
trailing_static_sec = trailing_run(diff < motion_thr_eff) / fps
```

- **自适应阈值（关键设计）**：64×48 下采样会压缩运动幅度动态范围（实测全画面运动
  ~1.3 vs 静止噪声 ~0.5），固定绝对阈值极脆弱。故「静止」阈值取自视频自身运动区间：

```
lo = p10(diffs)   hi = p90(diffs)
motion_thr_eff = max(abs_floor, lo + rel_frac * (hi - lo))   # abs_floor=0.3, rel_frac=0.35
```

  有明显运动的视频阈值自然抬高（不把弱运动误判静止）；真正的归位停留（接近噪声下限）
  仍落在阈值之下。整段几乎无运动（无运动参照）的视频交由 `static` 兜底，本检测器不过度误报。
- **判定**：`leading_static_sec > max_static_sec` 或 `trailing_static_sec > max_static_sec`
  （默认 2.0s）→ 命中（默认 WARN）。
- **降级**：缺 fps（含 ffprobe 兜底失败）/ ffmpeg 解码失败 / 超时 → WARN。

### 3bis.2 FreezeChecker（画面卡死 / 长时间卡帧，规范序号 5）

- **源文件**：`checkers/freeze.py`
- **目标**：检出**单段最长冻结时长**过长（解码器卡住 / 采集阻塞，连续多帧完全相同）。
  与 `dup_frame` 的区别：后者抓「复制帧伪装高帧率的整体卡顿 / 时间变慢」，以比例与
  周期性为判据；本检测器抓**局部单段长冻结**——即使该段只占全片很小比例（此时
  `dup_frame` 的 `static_like` 因 keep_ratio 仍高而不触发）。
- **原理**：用严格绝对阈值 `freeze_thr`（默认 0.1，近似「同一帧」，真实冻结 diff≈0）
  得到冻结掩码，取**最长连续冻结段**，按 fps 换算：

```
max_freeze_sec = longest_run(diff < freeze_thr) / fps
```

- **判定**：`max_freeze_sec > max_freeze_sec`（默认 2.0s）→ 命中（默认 WARN）。
- **降级**：缺 fps / ffmpeg 失败 / 超时 → WARN。

### 3bis.3 NoiseChecker（严重噪点，规范序号 11）

- **源文件**：`checkers/noise.py`
- **目标**：检出画面布满颗粒的严重噪点（高 ISO / 弱光增益 / 传感器噪声）。
- **原理**：Immerkær 快速噪声方差估计（*Fast Noise Variance Estimation*, CVIU 1996）。
  对灰度帧卷积二阶 Laplacian 掩码 M（对常量与一阶/二阶线性亮度变化响应为零，可消去
  大部分边缘与渐变），残差主要由高频噪声贡献：

```
        | 1 -2  1 |
    M = |-2  4 -2 |
        | 1 -2  1 |

    sigma = sqrt(pi/2) * Σ|M * I| / (6 * (W-2) * (H-2))
```

  均匀采样 `n_frames`（默认 16）帧（缩到 `max_h`，默认 720，避免过度下采样削弱噪点），
  逐帧估计 sigma，取**中位数**（对个别强纹理 / 文字帧鲁棒）。
- **判定**：`sigma_median > max_noise_sigma`（默认 8.0，0–255 量纲）→ 命中（默认 WARN）。
- **校验**：实测对标准差为 σ 的高斯噪声估计误差 < 1%（σ=3/10/20 → 3.0/10.0/20.0），
  纯渐变 / 边缘图估计 ≈ 0。
- **降级**：缺 OpenCV / 读帧失败 / 分辨率 < 3px → WARN。
- **限制**：纯空间法属启发式，强纹理 / 高频细节场景可能抬高 sigma，故默认仅报 WARN
  作为人工复核提示项；阈值建议按数据源在 `config['noise']` 标定。

> **规范序号 6（画面掉帧 / 多段短时卡帧 / 慢放）**：已由 `dup_frame` 完整覆盖
> （规则 A `periodic_stutter` 抓多段周期性复制帧，规则 B 抓「时间变慢」），无需新增检测器。

### 3bis.4 BrightnessChecker（画面过暗 / 欠曝，规范序号 20.4）

- **源文件**：`checkers/brightness.py`
- **背景（序号 20 拆分）**：规范序号 20「相机画面问题」原把偏色 / 白平衡异常 / 黑屏 /
  花屏混在一格（定义过宽），按子项拆分后各归其检测器：

| 子项 | 现状 | 归属 |
| --- | --- | --- |
| 20.1 黑屏 | 已覆盖 | `visual`（blackdetect 黑屏时长比，§2.3） |
| 20.2 花屏 | 部分覆盖 | `integrity`（ffmpeg 全解码报错，§5.1）；解码无 error 的传感器/传输层花屏暂未覆盖 |
| 20.3 偏色 / 白平衡异常 | 暂不实现 | 纯色彩判断真实场景误报率高（暖光 / 彩色桌布 / 单色主体），先不做 |
| 20.4 画面过暗 / 欠曝 | **本检测器** | `brightness` |

- **目标**：检出**整体偏暗但非全黑**的欠曝画面（能看到内容但明显过暗）。纯黑（关机 /
  遮挡镜头）归 `visual`，本项与之信号互补，定位在两者之间的「暗而不黑」区段。
- **原理**：均匀采样 `n_frames`（默认 16）帧转灰度（≈ 亮度 Y，缩到 `max_h` 默认 240），
  逐帧取全画面平均亮度，再对各帧取**中位数**（对个别曝光突变帧鲁棒）：

```
luma_i      = mean(gray_i)            # 第 i 帧全画面平均亮度，0–255
luma_median = median(luma_0..luma_{T-1})
score       = clip(luma_median / min_luma, 0, 1)   # 越高越好
```

- **判定**：`luma_median < min_luma`（默认 40.0，0–255 量纲）→ 命中（默认 WARN）。
- **降级**：缺 OpenCV / 读帧失败 → WARN，不抛异常。
- **与噪点（§3bis.3）的边界**：`noise` 只判噪声强度、不判整体亮度；弱光增益噪声由
  `noise` 覆盖，「画面过暗」本身由本检测器覆盖，二者互补。
- **限制**：合法的昏暗场景（刻意压暗的环境）也可能落在阈值下，故默认仅报 **WARN** 作
  人工复核提示；阈值建议按数据源在 `config['brightness']` 标定。纯判定逻辑抽成
  `evaluate_brightness()` 便于单测。

---

## 3ter. 多模态检测器（序号 12）

### 3ter.1 GripperOffscreenChecker（夹爪出境，规范序号 12）

- **源文件**：`checkers/gripper_offscreen.py`、后端 `checkers/_vision.py`
- **目标**：检出夹爪（机械臂末端执行器 / gripper）持续离开相机画面 **1s 以上**。
- **为何走多模态**：「夹爪是否在画面内」是语义识别 + 跟踪问题，纯像素帧差 / 光流无法
  稳健区分「夹爪移出画面」与「夹爪静止 / 被物体短暂遮挡」，故交给多模态模型。
- **两个可插拔维度**（`config['gripper_offscreen']`）：
  - `mode`：`image`（默认）/ `video`
  - `provider`：`gemini` / `openai`（后端实现见 `_vision.py`，统一接口屏蔽 SDK 差异）

#### image 模式（默认，感知交模型 / 判定留代码）

```
1. 本地抽帧：sample_frames_jpeg 按 sample_fps（默认 2）均匀抽帧为 JPEG（缩到 frame_max_h=360）；
   长视频自适应降采样 eff_fps = min(sample_fps, max_frames/duration)，保证 ≤ max_frames（默认120）覆盖全片。
2. 单次请求：把带 "Frame i:" 标号的图片序列 + 提示词发给所选 provider，结构化输出（JSON schema）
   逐帧只回 {index, gripper_visible: bool}。
3. 代码判定：offscreen = not gripper_visible；取最长连续出镜帧段 run，
   max_offscreen_sec = run / eff_fps；> min_offscreen_sec（默认 1.0s）→ 命中（默认 WARN）。
```

- **为何 image 作默认**：判定阈值「连续 ≥1s」对时间分辨率敏感。video 模式下 Gemini 对
  视频默认约 1fps 内部采样、时间戳偏粗，1s 踩在分辨率边界；自抽帧可把采样率定到 ≥2fps，
  让 1s 稳定落到 2~3 帧，时长换算在代码侧确定可复现，且任何视觉模型（gemini/openai）都能跑。
- **缺失帧兜底**：模型未返回的帧 index 一律按 `gripper_visible=True`（不可见才算出镜），避免过度误判。

#### video 模式（整段视频，依赖原生视频理解）

把整段视频交模型，直接返回出镜区间 `[(start_sec, end_sec, confidence)]`，代码按 `min_confidence`
过滤后取最长区间时长与 `min_offscreen_sec` 比较。**目前仅 `gemini` 支持**（Files API 上传→轮询
ACTIVE→结构化输出）；`openai` 后端会抛 `VideoModeUnsupported` → 降级 WARN 并提示改用 image。
适用于帧常被遮挡 / 需时间上下文、或长视频逐帧 token 成本过高的场景。

- **降级**：缺对应 API key / SDK 未安装 / 抽帧失败 / 接口异常 / 后端不支持该 mode → **WARN**，
  不阻塞流水线。
- **关键配置**（`config['gripper_offscreen']`）：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `mode` | `image` | `image` 逐帧 / `video` 区间 |
| `provider` | `gemini` | `gemini` / `openai` |
| `<provider>.model` | `gemini-2.5-flash` / `gpt-4o` | 各后端模型 |
| `<provider>.api_key_env` | `GEMINI_API_KEY` / `OPENAI_API_KEY` | API key 环境变量名 |
| `sample_fps` | 2.0 | image：抽帧采样率（1s 阈值的时间分辨率） |
| `frame_max_h` | 360 | image：抽帧下采样高度（省 token） |
| `max_frames` | 120 | image：单视频抽帧上限（长视频自动降采样） |
| `min_offscreen_sec` | 1.0 | 连续出镜超过此秒数即命中 |
| `min_confidence` | 0.5 | video：出镜区间置信度过滤 |
| `severity` | warn | 命中级别（warn/fail） |

> **启用方式**：`pip install -e ".[gemini]"`（或 `".[openai]"`）→ 设对应 API key →
> 把 `config` 的 `checks.gripper_offscreen` 改为 `true`（默认 false，因属付费远程调用）。

### 3ter.2 RegraspChecker（二次抓取，规范序号 1）

- **源文件**：`checkers/regrasp.py`、后端 `checkers/_vision.py`
- **目标**：检出视频中**多次抓取同一物体**（每次抓取应是单次且有效的）。
- **为何走多模态 + 为何拆解**：整段视频直接问模型「是否发生二次抓取」做不出来——它要
  同时干两件事：**逐帧感知**（谁在抓、抓的是什么）+ **跨时间推理**（判断「再次」）。
  本检测器把它拆成 **逐帧感知交给模型 + 时序判定留给代码**，与 `gripper_offscreen`
  的 image 模式同一套路，判定确定可复现。
- **夹爪信号优先取自 parquet 真实值（§2.2 改造，源文件再加 `checkers/_lerobot.py`）**：
  抓取 / 释放的「时序」本就以真实信号存在于 LeRobot 组 parquet
  （`puppet.end_effector_*_position_align`，见 `dataset_inputs.md §3`）。逐臂判定是单目标、
  忽略物体身份，故「夹爪闭合段数」即「抓取次数」——当视频位于 LeRobot v3.0 组内、装有
  `pyarrow` 且能读到该 episode 夹爪开合时，**优先走 parquet 路径直接逐臂计数**：完全不调
  多模态模型（无需 API key、零付费调用、用全分辨率时间轴），去抖 / 计数落在真实信号上更稳。
  任一前置不满足（配置关闭 `use_parquet_gripper` / 非 LeRobot 组 / 缺 `pyarrow` / 缺视频 fps /
  各侧开合整段不可判）→ **自动回退到多模态逐帧推断路径**，行为同改造前。
  - **来源（已对齐 §1 摄入 / 编排层枢纽）**：优先用摄入层（`GroupResolver`）注入到
    `metadata["lerobot"]["parquet_path"]` 的 parquet 指针（单一数据源，与 `static` /
    `object_slip` 一致），经 `_lerobot.gripper_opening_from_metadata` 读各侧逐帧开合标量；
    未经摄入层注入时（绕过 `GroupResolver` 直接调用 pipeline）才用 `find_episode_parquet`
    按视频路径自定位兜底。
  - **阈值化**：用每 episode 自身开合区间（稳健分位 q05/q95）做**相对归一化**（对单位 /
    标定不敏感）；归一化值 ≤ `gripper_closed_frac` 判闭合（默认约定「值越小越闭合」，
    `gripper_closed_is_low`）；整段几乎不动（区间 < `gripper_min_span`）→ 该侧不可判，
    各侧均不可判则回退模型路径。夹爪闭合即「持有」，复用 `detect_regrasp(single_object=True)`。
  - `details.source ∈ {parquet, ...}` 标注实际来源，走 parquet 时 `details.parquet` 记录路径。

#### 流程（parquet 夹爪优先 / 模型逐帧回退 + 代码侧逐臂状态机）

```
0. parquet 优先（§2.2）：use_parquet_gripper 且 LeRobot 组内、有视频 fps、能读到夹爪开合时，
   读各侧开合 → 相对归一化阈值化为逐帧"闭合" → 闭合即持有 → 直接跳到步骤 4（不调模型 / 不抽帧）。
   任一前置不满足或各侧整段不可判 → 回退下面的多模态路径。
1. 本地抽帧：sample_frames_jpeg 按 sample_fps（默认 2）均匀抽帧为 JPEG（缩到 frame_max_h=360）；
   长视频自适应降采样 eff_fps = min(sample_fps, max_frames/duration)，保证 ≤ max_frames（默认120）覆盖全片。
2. 单次请求：把带 "Frame i:" 标号的图片序列 + 提示词发给所选 provider（gemini/openai），结构化输出
   逐帧、逐夹爪回 {index, grippers:[{side, holding, object_label, confidence}]}。
3. 归约：按夹爪 side（left/right/single）各建一条序列 seq_side[i]=该臂持有标签
   （未持有 / 缺该夹爪条目 / 低于 min_confidence → None；持有但无标签 → 占位 __hold__）。
4. 去抖（关键，压遮挡/误检噪声）：
   - 丢弃长度 < min_hold_frames 的"持有"段（疑似单帧误检）；
   - 桥接长度 < min_release_frames 的"释放"缝隙（遮挡闪断）。
   两阈值由秒换算（× eff_fps 或 parquet 路径的视频 fps），均下限 2 帧以滤掉单帧抖动。
5. 逐臂计数判定：每只机械臂去抖后的独立"持有段"数，某臂 ≥2 段（被真释放隔开）→ 二次抓取（默认 WARN）。
```

#### 按机械臂判定（关键设计）

数据约定**每只机械臂都是单目标**（可能单臂或双臂），故判定单位是**机械臂**而非全局：

- 模型逐帧按画面左右标 `side`（`left` / `right`），单臂用 `single`；代码侧按 side 分别建序列。
- **同一只臂**抓取 → 释放 → 再抓取（该臂持有段 ≥2）→ 命中。
- **双臂各自抓取一次**，甚至 A 臂放下、B 臂接力（A→B 交接）→ 每臂各 1 段 → 正常，不误报。
- 每只臂单目标，故臂内**忽略物体标签**只数持有段（`object_label` 仅留作复核展示）。

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `sample_fps` | 2.0 | 抽帧采样率（抓取/释放时间分辨率；模型回退路径用） |
| `min_hold_sec` | 0.5 | 持有段最短时长（去抖，下限 2 帧） |
| `min_release_sec` | 1.0 | 真释放最短时长（桥接遮挡闪断，下限 2 帧） |
| `min_confidence` | 0.0 | 低于此的"持有"按未持有处理（0=不启用；模型回退路径用） |
| `max_frames` | 120 | 单视频抽帧上限（长视频自动降采样；模型回退路径用） |
| `use_parquet_gripper` | true | 优先用 parquet 夹爪真实信号逐臂计数；false 则始终走模型逐帧推断 |
| `gripper_closed_is_low` | true | 夹爪开合值越小=越闭合（多数 LeRobot/RoboMIND 约定，按组实测可改） |
| `gripper_closed_frac` | 0.5 | 归一化开合值 ≤ 此比例判为闭合（`closed_is_low` 时） |
| `gripper_min_span` | 1e-6 | 整段开合区间小于此（夹爪几乎不动）视为该侧不可判 |

#### 已知局限

- 夹爪仅**重新调整握姿**（未完全松开）属灰区，去抖与阈值都难拿稳。
- parquet 路径下，夹爪开合的「闭合方向」约定按组可能不同，默认「值越小越闭合」；如某组相反
  需用 `gripper_closed_is_low` 校正（开合区间过小的 episode 自动判为不可判，回退模型路径）。
- 模型回退路径下：双臂时若模型把同一只臂的 `side` 在帧间左右互换，会把一段连续持有拆到两条
  序列，可能误报/漏报；提示词已要求 side 全程一致，固定机位下通常稳定，仍属固有风险。物体小 /
  与夹爪同色 / 桌面拥挤时逐帧"是否持有"会不准。故默认仅报 **WARN** 作人工复核提示。
- **降级**：parquet 不可用（非 LeRobot 组 / 缺 `pyarrow` / 缺视频 fps / 各侧不可判）→ 回退
  多模态路径；模型路径缺 API key / SDK / 抽帧失败 / 接口异常 → WARN，均不阻塞流水线。

> **启用方式**：把 `config` 的 `checks.regrasp` 改为 `true`（默认 false）。parquet 路径
> （推荐，零付费）：`pip install -e ".[lerobot]"`（pyarrow）；模型回退路径：
> `pip install -e ".[gemini]"`（或 `".[openai]"`）→ 设对应 API key。

### 3ter.3 ObjectSlipChecker（物体滑落，规范序号 21）

- **源文件**：`checkers/object_slip.py`、后端 `checkers/_vision.py`
- **目标**：检出**夹爪夹住物体后物体滑落**（脱手 / 掉落）。
- **与「二次抓取」的关系（为何不是其子集）**：两者共用同一套架构（本地抽帧 → 多模态逐帧
  感知 → 代码侧逐臂状态机），但**判据不同**：
  - regrasp 看「同一只臂持有段数 ≥2」。它**副作用上**能命中「滑落后又捡起」这一子集，
    但会**错标成二次抓取**，且抓不到「滑落后不补抓」「夹爪夹空继续空走」（这些只有 1 段持有）。
  - 「持有段结束」这个事件本身**区分不了**主动放下与滑落——必须看夹爪状态：
    主动放下时夹爪**张开**，滑落时夹爪**仍闭合**但物体没了。
- **关键新增信号**：在 regrasp 的逐帧 schema 上**额外要求模型回 `gripper_closed`**
  （`_vision._grasp_frame_schema` 的可选字段；regrasp 不要求它，故二者后端调用完全兼容，
  可共享同一次模型请求成本）。
- **夹爪信号优先取自 parquet 真实值（§2.3 改造，源文件再加 `checkers/_lerobot.py`）**：
  `gripper_closed` 是本检测器的唯一判据，而它本就以真实信号存在于 LeRobot 组 parquet
  （`puppet.end_effector_*_position_align`，见 `dataset_inputs.md §3`）。故当视频位于
  LeRobot v3.0 组内、装有 `pyarrow` 且能读到该 episode 夹爪开合时，**优先用 parquet 开合
  真实信号**逐帧判「闭合 / 张开」，模型只继续负责「是否持有」，判定更可靠、且省一份逐帧
  夹爪推断成本。任一条件不满足（非 LeRobot 组 / 缺 `pyarrow` / 缺视频 fps / 列缺失）→
  **逐臂自动回退到模型推断的 `gripper_closed`**，行为同改造前。
  - **来源（已对齐 §1 摄入 / 编排层枢纽）**：优先用摄入层（`GroupResolver`）注入到
    `metadata["lerobot"]["parquet_path"]` 的 parquet 指针（单一数据源），经
    `_lerobot.gripper_opening_from_metadata` → `lerobot.load_episode_frames` 读各侧逐帧开合标量；
    未经摄入层注入时（绕过 `GroupResolver` 直接调用 pipeline）才用 `find_episode_parquet`
    按视频路径自定位兜底。
  - **阈值化**：用每 episode 自身开合区间（稳健分位 q05/q95）做**相对归一化**（对单位 /
    标定不敏感）；归一化值 ≤ `gripper_closed_frac` 判闭合（默认约定「值越小越闭合」，
    `gripper_closed_is_low`）；整段几乎不动（区间 < `gripper_min_span`）→ 该侧不可判（保守）。
  - **对齐**：parquet 一行对应一视频帧，按 `round(t_采样帧 × 视频fps)` 映射到检测器抽到的采样帧。
  - **side 配对**：parquet 侧（left/right）与模型侧精确配对；模型判为单臂(`single`)而 parquet
    恰只有一侧 → 用该侧；无法稳妥配对则回退模型信号（避免左右错配）。
  - `details.gripper_source[side]` 记录每只臂实际用了 `parquet` 还是 `model`，`details.parquet`
    记录 parquet 路径，便于复核。

#### 流程（image 逐帧 + parquet 夹爪 + 代码侧逐臂状态机）

```
1. 本地抽帧：同 regrasp（sample_fps 默认 2，缩到 frame_max_h=360，≤ max_frames 自适应降采样）。
2. 单次请求：模型逐帧、逐夹爪回 {side, holding, gripper_closed, object_label, confidence}。
3. 归约：按 side(left/right/single) 建持有序列 hold_seq[i]=持有标签或 None；
   夹爪闭合序列 closed_seq[i]=是否闭合(True/False/None=未知) —— 优先用 parquet 真实开合
   阈值化得到（映射到采样帧），不可用时回退模型 gripper_closed。
4. 去抖（复用 regrasp 同一套）：丢弃 < min_hold_frames 的持有段；桥接 < min_release_frames
   的释放缝隙（逐臂单目标，忽略标签）。
5. 逐臂滑落判定：对每个"在片尾之前结束"的持有段，检查其后 release_window_frames 帧
   （不越过下一持有段）内的夹爪状态：
     - 窗口内出现张开(gripper_closed=False) → 正常放下，不计；
     - 窗口内已知状态全为闭合(True) → 夹爪仍闭合却已脱手 → 滑落命中；
     - 窗口内全未知 → 无法判别，保守跳过（不误报）。
```

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `sample_fps` | 2.0 | 抽帧采样率（持有/释放/滑落时间分辨率） |
| `min_hold_sec` | 0.5 | 持有段最短时长（去抖，下限 2 帧） |
| `min_release_sec` | 1.0 | 真脱手最短时长（桥接遮挡闪断，下限 2 帧） |
| `release_window_sec` | 0.5 | 持有结束后检查夹爪状态的窗口（下限 1 帧） |
| `min_confidence` | 0.0 | 低于此的"持有"按未持有处理（0=不启用） |
| `max_frames` | 120 | 单视频抽帧上限（长视频自动降采样） |
| `use_parquet_gripper` | true | 优先用 parquet 夹爪真实信号；false 则始终用模型推断 |
| `gripper_closed_is_low` | true | 夹爪开合值越小=越闭合（多数 LeRobot/RoboMIND 约定，按组实测可改） |
| `gripper_closed_frac` | 0.5 | 归一化开合值 ≤ 此比例判为闭合（`closed_is_low` 时） |
| `gripper_min_span` | 1e-6 | 整段开合区间小于此（夹爪几乎不动）视为该侧不可判 |

#### 已知局限

- 物体小 / 与夹爪同色 / 桌面拥挤时逐帧「是否持有」会不准（夹爪状态用 parquet 后不再依赖模型）。
- 夹爪仅**松动微滑**（物体仍在指间）属灰区，难与正常持有区分。
- parquet 夹爪开合的「闭合方向」约定按组可能不同，默认「值越小越闭合」；如某组相反需用
  `gripper_closed_is_low` 校正（开合区间过小的 episode 自动判为不可判，不会误报）。
- 夹爪状态全程未知时保守跳过，可能漏判；故默认仅报 **WARN** 作人工复核提示。
- **降级**：缺 API key / SDK / 抽帧失败 / 接口异常 → WARN，不阻塞流水线；parquet 不可用 →
  逐臂回退模型夹爪信号，不影响 holding 判定。

> **启用方式**：`pip install -e ".[gemini]"`（或 `".[openai]"`）→ 设对应 API key →
> 把 `config` 的 `checks.object_slip` 改为 `true`（默认 false，因属付费远程调用）。

### 3ter.4 ColorMatchChecker（操作物与桌面颜色相同，规范序号 19）

- **源文件**：`checkers/colormatch.py`、后端 `checkers/_vision.py`
- **目标**：检出**被操作物体与桌面（台面背景）颜色 / 纹理过于接近**，导致难以分辨物体
  位置和大小的视频。
- **为何走多模态**：「物体与桌面是否同色难分辨」本质是**语义识别 + 主观可辨识度**判断。
  纯像素也能算「物体区域 vs 桌面背景的颜色对比度」，但**绕不开先定位被操作物体**这一
  语义步骤（哪个才是"被操作物"），不稳健，故整体交给多模态模型一步判定。
- **与 regrasp / object_slip 的本质区别（为何更简单）**：本项是**静态属性**——整段视频
  一个结论（甚至首帧物体在桌上时就能判），**不依赖动作过程，不需要时序状态机**。因此
  只需少量抽帧、逐帧独立判定，再做占比聚合即可，单次请求成本远低于 regrasp / slip。

#### 流程（image 逐帧 + 代码侧占比聚合）

```
1. 本地抽帧：sample_frames_jpeg 按 sample_fps（默认 1，低采样省成本）均匀抽帧为 JPEG
   （缩到 frame_max_h=360）；静态属性故 max_frames 默认仅 16，长视频自适应降采样覆盖全片。
2. 单次请求：模型逐帧回 {index, hard_to_distinguish, object_label, confidence}
   （_vision._colormatch_frame_schema）；看不到可辨识的被操作物体的帧可省略不返回。
   提示词除 robot_hint 外，还会注入 task_hint（见下「物体提示」）帮模型定位被操作物。
3. 代码侧聚合（evaluate_colormatch，纯函数可单测）：
   - 只在「模型返回了判定」的帧里统计（未返回 = 无可辨识物体 → 不计入分母）；
   - min_confidence>0 时，低置信度帧也跳过；
   - 「难分辨」(hard=True) 帧占比 hit_ratio = n_hard / n_judged；
   - 有效判定帧数 < min_judged_frames（默认 2）→ 报 WARN「无法评估」（避免单帧误判）；
   - 否则 hit_ratio >= hit_ratio（默认 0.5）→ 命中（默认 WARN）。
   score = 1 - hit_ratio（越高越好，越容易分辨越好）。
```

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `sample_fps` | 1.0 | 抽帧采样率（静态属性，低采样即可） |
| `max_frames` | 16 | 单视频抽帧上限（少量均匀帧覆盖全片，省成本） |
| `hit_ratio` | 0.5 | 可辨识物体的帧里"难分辨"占比 ≥ 此值即命中 |
| `min_judged_frames` | 2 | 有效判定帧数下限，低于此报 WARN（无法评估） |
| `min_confidence` | 0.0 | 低于此的帧判定跳过（0=不启用） |

#### 物体提示（task_hint，前向兼容钩子）

本项最难的一步是「先定位哪个才是被操作物」（语义问题）。`_task_hint(metadata)` 会从
metadata 抽取并注入两类提示，帮模型锁定目标，**有就用、缺省则返回空串、不影响纯视频检测**：

- `metadata["target_objects"]`（list 或 str）：被操作目标物体名，理想来源是 LeRobot
  `labels/labels.json` 子任务名里的物体（如「抓取**书籍**」）。
- `metadata["task"]`（str）：任务描述，来源 `meta/tasks.jsonl` 或
  `info.json.metadata.language_instruction`。

> 与 `metadata["robot"]` 同属「检测器侧已读取、由数据摄入层负责填充」的钩子：当前
> `pipeline._extract_metadata` 只填 ffprobe 字段，故二者默认为空；待按 LeRobot 组结构
> 摄入（见 `docs/dataset_inputs.md`）后填入即自动生效，无需再改本检测器。

#### 已知局限

- "很难分辨"本身带主观性，模型判定会有边界波动；故默认仅报 **WARN** 作人工复核提示，
  并以「占比 ≥ 阈值」聚合压单帧噪声。
- 物体被夹爪长时间遮挡 / 多目标场景下「哪个是被操作物」可能识别不稳，体现为 `n_judged` 偏低。
- **降级**：缺 API key / SDK / 抽帧失败 / 接口异常 / 无有效判定帧 → WARN，不阻塞流水线。

> **启用方式**：`pip install -e ".[gemini]"`（或 `".[openai]"`）→ 设对应 API key →
> 把 `config` 的 `checks.colormatch` 改为 `true`（默认 false，因属付费远程调用）。

### 3ter.5 OcclusionChecker（首帧夹爪遮挡操作物品，规范序号 15）

- **源文件**：`checkers/occlusion.py`、后端 `checkers/_vision.py`
- **目标**：检出**视频首帧 / 开局**存在夹爪（机械臂末端执行器 / 连杆）遮挡**被操作物体**，
  导致开始时看不清目标物体的位置 / 大小。理想采集应让操作物体在开局清晰可见、不被挡住。
- **为何走多模态**：「夹爪是否挡住被操作物体」本质是**语义识别**——要先定位「哪个才是被操作物」，
  再判断它是否被夹爪 / 机械臂遮挡。纯像素帧差 / 光流绕不开这一步，故整体交给多模态模型一步判定。
- **与 colormatch 的关系（同为静态属性、更轻）**：本项同样是**静态属性**（看视频开局即可），
  不依赖动作过程、不需要时序状态机，复用 colormatch「逐帧独立判定 + 代码侧占比聚合」的同一套路。
  区别仅在于**只抽视频开头一小段**（忠实「首帧」语义）而非全片均匀抽帧。

#### 流程（只抽首段 + 逐帧遮挡 + 代码侧占比聚合）

```
1. 本地抽帧（仅首段）：sample_frames_jpeg(duration_sec=head_sec) 用 ffmpeg -t 只解码视频开头
   head_sec（默认 1.0s）这一段，按 sample_fps（默认 2）抽帧为 JPEG（缩到 frame_max_h=360）；
   首段很短，max_frames 默认仅 8。只抽首段既忠实「首帧」语义，又省 token、且避免抽到中段
   正常抓取时夹爪遮挡物体造成的误报。
2. 单次请求：模型逐帧回 {index, gripper_occludes_object, object_label, confidence}
   （_vision._occlusion_frame_schema）；看不清且非被夹爪挡住的帧可省略不返回，
   因被夹爪完全挡住而看不到则应回 gripper_occludes_object=true。
   提示词除 robot_hint 外，还会注入 task_hint（复用 colormatch 的钩子）帮模型定位被操作物。
3. 代码侧聚合（evaluate_occlusion，纯函数可单测）：
   - 只在「模型返回了判定」的帧里统计（未返回 = 无可辨识物体 → 不计入分母）；
   - min_confidence>0 时，低置信度帧也跳过；
   - 「被遮挡」(occluded=True) 帧占比 hit_ratio = n_occluded / n_judged；
   - 有效判定帧数 < min_judged_frames（默认 1）→ 报 WARN「无法评估」；
   - 否则 hit_ratio >= hit_ratio（默认 0.5）→ 命中（默认 WARN）。
   只抽了首段，故「多数首段帧被遮挡」即对应规范「首帧存在遮挡」。score = 1 - hit_ratio（越高越好）。
```

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `head_sec` | 1.0 | 只抽取视频开头这一段时长（秒）判定（忠实「首帧」语义） |
| `sample_fps` | 2.0 | 首段抽帧采样率（head_sec×sample_fps 约为首段帧数） |
| `max_frames` | 8 | 首段抽帧上限（首段很短，少量帧即可） |
| `hit_ratio` | 0.5 | 可辨识物体的首段帧里「被遮挡」占比 ≥ 此值即命中 |
| `min_judged_frames` | 1 | 有效判定帧数下限，低于此报 WARN（无法评估） |
| `min_confidence` | 0.0 | 低于此的帧判定跳过（0=不启用） |

#### 已知局限

- 「是否被遮挡」在夹爪贴近但未挡住物体时带边界主观性，模型判定会有波动；故默认仅报 **WARN**
  作人工复核提示，并以「占比 ≥ 阈值」聚合压单帧噪声。
- 多目标 / 桌面拥挤 / 物体本身极小时「哪个是被操作物」可能识别不稳，体现为 `n_judged` 偏低 → 报 WARN。
- **降级**：缺 API key / SDK / 抽帧失败 / 接口异常 / 无有效判定帧 → WARN，不阻塞流水线。

> **启用方式**：`pip install -e ".[gemini]"`（或 `".[openai]"`）→ 设对应 API key →
> 把 `config` 的 `checks.occlusion` 改为 `true`（默认 false，因属付费远程调用）。

---

## 4. 检测器速查表

| 检测器 | 默认级别 | 核心指标 | 依赖 | 失败 / 降级行为 |
| --- | --- | --- | --- | --- |
| metadata | FAIL | 分辨率 / 帧率 / 时长阈值 | ffprobe | 缺值 → FAIL/WARN |
| integrity | FAIL | ffmpeg 解码 stderr | ffmpeg | — |
| visual | WARN | 黑屏时长比 | ffmpeg | 缺时长 → WARN |
| static | WARN | motion_score / active_ratio / 关节峰峰值(joint) | OpenCV（lite）/ torch+RAFT（raft）/ pyarrow（joint） | 读帧失败、RAFT 缺失 → WARN；parquet 不可用 → 回退像素后端 |
| dup_frame | WARN | keep_ratio / dup_ratio_strict | ffmpeg + numpy | 超时、帧数不足 → WARN |
| jump | WARN | local_ratio_max + 四重守卫 | OpenCV | 打不开 → WARN，帧少 → PASS |
| endpoint_static | WARN | 首/尾连续静止时长（自适应阈值） | ffmpeg + numpy | 缺 fps / 超时 → WARN |
| freeze | WARN | 单段最长冻结时长 | ffmpeg + numpy | 缺 fps / 超时 → WARN |
| noise | WARN | Immerkær 噪声 sigma 中位数 | OpenCV + numpy | 缺 OpenCV / 读帧失败 → WARN |
| brightness | WARN | 全画面平均亮度中位数（过暗/欠曝） | OpenCV + numpy | 缺 OpenCV / 读帧失败 → WARN |
| gripper_offscreen | WARN（默认关闭） | 最长连续出镜时长（image 逐帧 visible / video 区间） | ffmpeg + google-genai 或 openai + 网络 | 缺 key/SDK、抽帧/接口异常、mode 不支持 → WARN |
| regrasp | WARN（默认关闭） | 去抖后逐臂持有段数（≥2 即二次抓取） | pyarrow（parquet 优先，零付费）/ ffmpeg + google-genai 或 openai + 网络（回退） | parquet 不可用 → 回退模型；缺 key/SDK、抽帧/接口异常 → WARN |
| object_slip | WARN（默认关闭） | 持有结束时夹爪仍闭合却脱手（滑落） | pyarrow（夹爪信号优先）+ ffmpeg + google-genai 或 openai + 网络 | parquet 不可用 → 逐臂回退模型夹爪；缺 key/SDK、抽帧/接口异常 → WARN |
| colormatch | WARN（默认关闭） | 可辨识帧中"操作物与桌面同色难分辨"占比 ≥ 阈值 | ffmpeg + google-genai 或 openai + 网络 | 缺 key/SDK、抽帧/接口异常、无有效判定帧 → WARN |
| occlusion | WARN（默认关闭） | 首段可辨识帧中"夹爪遮挡操作物体"占比 ≥ 阈值 | ffmpeg + google-genai 或 openai + 网络 | 缺 key/SDK、抽帧/接口异常、无有效判定帧 → WARN |

---

## 5. 检测器职责边界与冗余说明

基础质检（metadata / integrity / visual）与时序质检（static / dup_frame / jump）在功能上
存在少量重叠，但各自都有不可替代的判定语义。本节说明为何三者都保留。

### 5.1 integrity —— 唯一严格的坏文件守门员（不可替代）

时序检测器用 OpenCV / decord 读帧，遇到损坏帧通常**静默跳过**（返回 `None` → 降级 WARN），
不会判定文件本身有问题。integrity 用 `ffmpeg -v error` 做全解码校验：
**「OpenCV 能读出部分帧」与「ffmpeg 全程零 error」是两个标准**，前者会吞掉花屏 / 丢包 /
尾部截断。因此 integrity 的全解码虽与后续读帧在**计算上有重叠**，但其判定语义不可替代，
是唯一默认能让 `passed=False` 的坏文件兜底。

### 5.2 metadata —— 提取是刚需，校验是廉价过滤

需区分两层：

- **元数据提取**（`pipeline._extract_metadata`）：**无条件运行**，且其输出是其他检测器的
  输入——`fps` 喂给 dup_frame 做归一化、`duration` 喂给 visual、`robot` 喂给 jump 查阈值表。
  这部分是硬依赖，不可删。
- **MetadataChecker 阈值校验**（分辨率 / 帧率 / 时长越界）：纯数值比较，零像素计算，作为入库前
  的硬性规格过滤很实用。若数据源规格已统一，可经 `config["checks"]["metadata"]` 关闭，但保留
  几乎无成本。

### 5.3 visual —— 与 static 部分重叠，但抓的是不同失效

- **重叠**：全黑视频运动量也极低，`static` 大概率会判为静止；两者都在抓「无有效内容」。
- **差异**：纯黑（关机 / 遮挡镜头）与「画面静止但非黑」是两类失效。带传感器噪点的黑画面，
  static 可能因微弱 motion 漏判，而 `blackdetect` 能直接命中。且 visual 仅报 **WARN**，定位为
  static 的补充提示项。

> 结论：三者均保留。integrity / metadata 提取不可替代；metadata 阈值校验与 visual 成本低且
> 提供独立信号，按数据源质量可经 `config["checks"]` 单独关闭，无需删码。

## 6. 扩展新检测器

1. 在 `checkers/` 新建模块，继承 `BaseChecker`，实现 `check()` 返回 `list[CheckResult]`。
2. 在 `checkers/__init__.py` 导出，并在 `pipeline._build_checkers` 按 `config["checks"]` 开关装配。
3. 在 `config/default.yaml` 增加该检测器的配置块（建议含 `severity` 字段以支持 warn/fail 切换）。
4. 遵循约定：`score` 越高越好；依赖缺失 / 异常返回 WARN 而非抛出；命中默认 WARN。
