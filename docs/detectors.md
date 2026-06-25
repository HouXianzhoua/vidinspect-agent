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

`config["checks"]` 逐项控制是否启用，默认全开。装配顺序：
`integrity → metadata → visual → static → dup_frame → jump → endpoint_static → freeze → noise`。

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

- **源文件**：`checkers/static.py`、`checkers/_raft.py`
- **目标**：检出整段几乎无运动（机械臂静止、无效操作）的视频。
- **后端选择**：`config["static"]["backend"]`，`lite`（默认，纯 CPU）或 `raft`（GPU）。

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

#### 关键阈值

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `thr` | 0.30 | lite：motion_score 低于此判静止 |
| `raft_thr` | 0.10 | raft：active_ratio 低于此判静止 |
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

---

## 4. 检测器速查表

| 检测器 | 默认级别 | 核心指标 | 依赖 | 失败 / 降级行为 |
| --- | --- | --- | --- | --- |
| metadata | FAIL | 分辨率 / 帧率 / 时长阈值 | ffprobe | 缺值 → FAIL/WARN |
| integrity | FAIL | ffmpeg 解码 stderr | ffmpeg | — |
| visual | WARN | 黑屏时长比 | ffmpeg | 缺时长 → WARN |
| static | WARN | motion_score / active_ratio | OpenCV（lite）/ torch+RAFT（raft） | 读帧失败、RAFT 缺失 → WARN |
| dup_frame | WARN | keep_ratio / dup_ratio_strict | ffmpeg + numpy | 超时、帧数不足 → WARN |
| jump | WARN | local_ratio_max + 四重守卫 | OpenCV | 打不开 → WARN，帧少 → PASS |
| endpoint_static | WARN | 首/尾连续静止时长（自适应阈值） | ffmpeg + numpy | 缺 fps / 超时 → WARN |
| freeze | WARN | 单段最长冻结时长 | ffmpeg + numpy | 缺 fps / 超时 → WARN |
| noise | WARN | Immerkær 噪声 sigma 中位数 | OpenCV + numpy | 缺 OpenCV / 读帧失败 → WARN |

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
