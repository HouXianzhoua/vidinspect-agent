# 数据集可用信息清单（检测工具的数据输入参考）

本文枚举**每一组数据**（一个 LeRobot v3.0 数据集目录）能提供给检测工具的全部信息：
**种类 / 作用 / 介绍**。目的是让后续检测器设计不再局限于“探测视频元数据 + 抽帧看画面”，
而是可以把**机器人时序状态、遥操作信号、时间戳、统计量、标定参数、子任务标注**等一并作为输入，
做跨模态、跨字段的一致性与质量校验。

> 本文所有字段均以真实数据核对（样本组：
> `tienkung_station_dualArm-gripper-3cameras_9/..._09_D-O-02_03_20260326/success/lerobot_RoboMIND`），
> 并对数据根目录下全部 500 组做了交叉统计。涉及“因组而异”的项已明确标注。

> **样本性质（重要）**：`tienkung_tabletop_soft_gripper_5episode_v2.1` 下的这 **500 组数据均已通过人工检测（合格样本）**，
> 路径中的 `success/` 也印证了这一点。因此本文档描述的就是**检测工具要处理的目标数据类型**——
> 即这类**数据格式**的视频/机器人数据（本批样本恰好都是人工已判定合格的）。换言之：
> - 这批数据可作为检测器的**正样本/基线**，用来标定阈值、估计正常分布、回归测试（理想情况下应几乎全部判 PASS，误报需重点排查）。
> - 它**不含已知缺陷样本**，无法直接用于评估漏报率（召回）；如需评估对缺陷的检出能力，应另行准备含问题的负样本或做人为注入。

---

## 0. 一组数据的边界与目录结构

“一组”= 一个独立的 LeRobot v3.0 数据集，定位到 `.../success/lerobot_RoboMIND/` 这一层：

```
<robot_type>/<task-session>/success/lerobot_RoboMIND/
├── data/chunk-000/episode_*.parquet          # 时序状态/动作（与视频同步，逐帧）
├── videos/chunk-000/<camera_key>/episode_*.mp4   # 三路彩色视频
├── meta/
│   ├── info.json            # 数据集总览 + feature schema + 相机参数 + 采集信息
│   ├── episodes.jsonl       # 每 episode 索引/长度/任务/平台ID
│   ├── episodes_stats.jsonl # 每 episode 统计量 + 数据/视频定位指针
│   ├── stats.json           # 全数据集逐字段聚合统计
│   └── tasks.jsonl          # 任务索引 ↔ 任务名
└── labels/labels.json       # 中文子任务分段标注
```

**关键约定（务必先读）**

- 数据根目录 `tienkung_tabletop_soft_gripper_5episode_v2.1` 下共 **500 组**。
- `info.json.total_episodes` 描述的是**源数据集规模**（样本组为 134），但**实际交付到该文件夹的只有约 5 个 episode**（样本组 `episode_000129~000133`），对应名字里的 “5episode”。
  → 实际可检测的 episode 数 = `data/chunk-000/` 下 parquet 文件数，**不是** `total_episodes`。
- 几乎所有“具体参数”都因组而异（fps、编码、是否有标定等），检测工具**必须逐组从 `info.json` 读取**，不能写死。
- 命名小坑：`info.json` 的 `data_path/video_path` 模板写的是 `file-{file_index}.*`，但**实际文件名是 `episode_{编号}.*`**（RoboMIND 导出差异）。按真实文件名遍历更稳妥。

---

## 1. 信息种类总览

| # | 信息种类 | 载体文件 | 粒度 | 主要作用 |
|---|---|---|---|---|
| 1 | 彩色视频（三路） | `videos/.../episode_*.mp4` | 逐帧/整段 | 画面质量、视角、时长、编码合规 |
| 2 | 机器人本体状态/动作（puppet） | `data/*.parquet` | 逐帧 | 轨迹合理性、关节范围、抖动/跳变、双臂协同 |
| 3 | 遥操作主控信号（master） | `data/*.parquet` | 逐帧 | 主从一致性、是否真人遥操作、控制延迟 |
| 4 | 逐字段时间戳 | `data/*.parquet` | 逐帧 | 采样间隔、时钟连续性、视频↔状态对齐 |
| 5 | 人工干预标志 `is_intervene` | `data/*.parquet` | 逐帧 | 是否含人工接管、数据纯净度 |
| 6 | 通用帧索引 | `data/*.parquet` | 逐帧 | 帧数一致性、索引连续性、episode 边界 |
| 7 | 数据集 schema / feature 定义 | `meta/info.json` | 数据集 | dtype/shape 校验、特征齐全性 |
| 8 | 视频规格（编码/分辨率/帧率） | `meta/info.json` | 数据集 | 与真实视频比对是否一致 |
| 9 | 相机型号 / 色彩通道 / 分辨率 | `meta/info.json` | 数据集 | 设备一致性、通道顺序（BGR） |
| 10 | 相机标定（内参/畸变/外参） | `meta/info.json` | 数据集 | 标定齐全性、几何一致性（**475/500 组有**） |
| 11 | 采集元信息（时间/采集者/指令） | `meta/info.json` | 数据集 | 溯源、任务-指令匹配、批次核对 |
| 12 | episode 索引表 | `meta/episodes.jsonl` | episode | 长度核对、平台ID溯源、任务归属 |
| 13 | episode 级统计 + 定位指针 | `meta/episodes_stats.jsonl` | episode | 异常 episode 识别、按时间切片对齐 |
| 14 | 数据集级聚合统计 | `meta/stats.json` | 数据集 | 全局分布、归一化参数、离群判断 |
| 15 | 任务表 | `meta/tasks.jsonl` | 数据集 | 任务索引↔任务名映射 |
| 16 | 中文子任务标注 | `labels/labels.json` | episode/帧段 | 分段合理性、帧号越界、标注齐全性 |

---

## 2. 视频数据（`videos/chunk-000/<camera_key>/episode_*.mp4`）

**种类**：三路彩色视频，相机 key 固定为
`camera_observations.color_images.camera_left` / `camera_right` / `camera_top`。

**介绍**：每个 episode 三个机位各一个 mp4；无音频；**无深度图**（虽然 `info.json` 里有
`camera_depth_resolution.*` 字段，但未交付深度视频）。

**真实规格（样本组）**：640×480，h264，yuv420p，30fps，`nb_frames≈2941`。

**因组而异（全量统计）**：
- 编码 `video.codec`：**h264 共 966 / av1 共 534**（按文件计）。
- 帧率 `fps`：**29 有 452 组 / 30 有 45 组 / 28 有 3 组**。

**作为检测输入可做**：清晰度/曝光/色偏/花屏/黑帧/卡顿、视角是否正确、三路画面时长一致性、
真实视频规格与 `info.json` 声明是否一致、帧数与 parquet 行数是否一致。

---

## 3. 机器人本体状态/动作 —— puppet（`data/*.parquet`）

**种类**：执行端（被控机器人）逐帧关节与夹爪数据。

| 字段 | dtype / shape | 作用 / 介绍 |
|---|---|---|
| `puppet.arm_left_position_align.data` | float32 [7] | 左臂 7-DOF 关节位置 |
| `puppet.arm_right_position_align.data` | float32 [7] | 右臂 7-DOF 关节位置 |
| `puppet.end_effector_left_position_align.data` | float32 [1] | 左夹爪开合 |
| `puppet.end_effector_right_position_align.data` | float32 [1] | 右夹爪开合 |

> 部分机器人型号还会有 `puppet.head_position_align.data`（float32 [3]，头部位姿）。**本样本组没有**，以各组 `info.json` 的 feature 列表为准。

**每个上述特征都附带两路伴随字段**：
- `.timestamp`（float64）：该特征的原始采集时间戳。
- `.is_intervene`（int64）：该特征是否处于人工干预。

**作为检测输入可做**：关节是否超出物理范围、相邻帧跳变/抖动、轨迹是否平滑、夹爪开合是否合理、
左右臂是否协同、是否长时间静止（无效采集）。

---

## 4. 遥操作主控信号 —— master（`data/*.parquet`）

**种类**：遥操作侧（主控/示教端）逐帧信号，dtype 为 **float64**（比 puppet 精度更高）。

| 字段 | dtype / shape | 作用 / 介绍 |
|---|---|---|
| `master.arm_left_position_align.data` | float64 [7] | 主控左臂 7-DOF |
| `master.arm_right_position_align.data` | float64 [7] | 主控右臂 7-DOF |
| `master.end_effector_left_position_align.data` | float64 [1] | 主控左夹爪 |
| `master.end_effector_right_position_align.data` | float64 [1] | 主控右夹爪 |

同样每个特征附带 `.timestamp`（float64）与 `.is_intervene`（int64）。

**作为检测输入可做**：主从一致性（master 与 puppet 的关节差异/滞后）、是否真人遥操作（master 是否有合理变化）、
控制延迟估计、master/puppet 时间戳错位。

---

## 5. 时间戳与同步信息

**种类**：多层时间戳，可用于采样率与对齐校验。

| 字段 | 来源 | 作用 |
|---|---|---|
| `timestamp` | parquet（float32，整帧） | 帧级统一时间轴（相对秒） |
| `<feature>.timestamp` | parquet（float64，每特征一路） | 各传感源原始采集时刻 |
| `camera_observations.timestamp` | parquet（float64） | 相机观测时间戳 |
| `videos/<key>/from_timestamp`、`to_timestamp` | episodes_stats.jsonl | 每路视频在该 episode 的起止时间 |

**作为检测输入可做**：采样间隔是否稳定（≈1/fps）、是否丢帧/时钟回退、各传感源时间戳是否对齐、
视频时间区间与状态时间轴是否吻合。

---

## 6. 人工干预标志 `is_intervene`

**种类**：int64 标志，逐帧、逐特征都有；并有 `camera_observations.is_intervene`。

**介绍 / 作用**：标记该帧/该信号是否由人工接管。可用于统计干预比例、筛除非纯净段、
或校验“声称无干预的数据是否确实全 0”。样本组中均为 0。

---

## 7. 通用帧索引（`data/*.parquet`）

| 字段 | dtype | 作用 |
|---|---|---|
| `frame_index` | int64 | episode 内帧序号（从 0 起） |
| `episode_index` | int64 | 所属 episode 编号 |
| `index` | int64 | 跨数据集全局行号 |
| `task_index` | int64 | 任务编号（关联 `tasks.jsonl`） |

**作为检测输入可做**：帧号连续无缺、行数 = `episodes.jsonl.length` = 视频帧数、episode 边界正确、
`task_index` 在 `tasks.jsonl` 中存在。

> 注意：parquet **不含图像列**。`camera_observations.color_images.*` 仅作为 `info.json` 的
> feature（dtype=video）与 stats 出现，真实像素只在 mp4 中。样本组 parquet = **31 列 / 2940 行**。

---

## 8. 数据集 schema 与视频规格（`meta/info.json`）

**种类**：数据集级元信息与 feature 定义。可作为“声明值”，与真实文件比对。

核心字段：
- `codebase_version`（v3.0）、`robot_type`、`fps`、`chunks_size`、`splits`（如 `train: 0:134`）。
- `total_episodes` / `total_frames` / `total_tasks`（源数据集规模，见 §0 说明）。
- `features.*`：每个特征的 `dtype` 与 `shape` —— **校验 parquet 列类型/维度的金标准**。
- 视频特征 `camera_observations.color_images.*.info`：`video.codec`、`video.height/width`、
  `video.fps`、`video.pix_fmt`、`video.channels`、`has_audio`、`video.is_depth_map`。

**作为检测输入可做**：parquet 实际 dtype/shape 是否与 schema 一致、特征是否齐全/多余、
真实 mp4 的编码/分辨率/帧率是否与声明一致。

---

## 9. 相机型号 / 色彩 / 分辨率（`meta/info.json`）

| 字段 | 样本值 | 作用 |
|---|---|---|
| `camera_model.camera_*` | `Orbbec_Gemini336L` | 设备一致性 |
| `camera_color_channel.camera_*` | `bgr` | 通道顺序（注意是 BGR，非 RGB） |
| `camera_color_resolution.camera_*` | `[640, 480]` | 彩色分辨率 |
| `camera_depth_resolution.camera_*` | `[640, 480]` | 深度分辨率（声明，但未交付深度视频） |

> 注：样本组中个别 key 值带前导空格（如 `" bgr"`），解析时建议 `strip()`。

---

## 10. 相机标定：内参 / 畸变 / 外参（`meta/info.json`）

**种类**：相机几何标定参数。**因组而异**：全量 **475/500 组含标定，25 组无**
（样本 books 组恰好属于无标定的 25 组之一）。

含标定时通常包含：
- `camera_intrinsics.<cam>.matrix`：3×3 内参矩阵。
- `camera_intrinsics.<cam>.dist_coeffs`：畸变系数。
- `camera_extrinsics.<cam>`：4×4 外参变换矩阵。

**作为检测输入可做**：标定齐全性检查、内参合理性（焦点/主点在画幅内）、多相机外参一致性、
（若需）投影/对齐类几何校验。检测工具需**容忍标定缺失**并据此降级。

---

## 11. 采集元信息（`meta/info.json` 的 `metadata.*`）

| 字段 | 样本值 | 作用 |
|---|---|---|
| `metadata.collection_time` | `2026-03-26 18:49:36` | 采集时间、批次溯源 |
| `metadata.collector` | 哈希ID | 采集者溯源 |
| `metadata.data_format_version` | `v1.0.0` | 格式版本 |
| `metadata.data_type` | `hdf5`（**全 500 组均是**） | 原始采集格式（再转 LeRobot） |
| `metadata.language_instruction` | 英文任务描述 | 与任务/标注一致性核对 |
| `metadata.trajectory_length` | `2920` | 轨迹长度参考 |

---

## 12. episode 索引表（`meta/episodes.jsonl`）

每行一个 episode：
- `episode_index`：episode 编号。
- `tasks`：任务名数组（样本 `["test01"]`）。
- `length`：帧数 —— **应等于 parquet 行数与视频帧数**。
- `platform_episode_id`：采集平台原始 ID，用于溯源。

**作为检测输入可做**：帧数三方一致性（episodes.length / parquet 行数 / mp4 帧数）、平台ID 唯一性、任务归属正确。

---

## 13. episode 级统计 + 定位指针（`meta/episodes_stats.jsonl`）

**种类**：每 episode 的统计量，**外加该 episode 的物理定位指针**（v3.0 特性，比常规统计更丰富）。

- 定位指针：
  - `data/chunk_index`、`data/file_index`、`dataset_from_index`、`dataset_to_index`。
  - 每路视频 `videos/<key>/chunk_index`、`file_index`、`from_timestamp`、`to_timestamp`。
- 逐字段统计：`stats/<field>/{min,max,mean,std,count,q01,q10,q50,q90,q99}`。

**作为检测输入可做**：按时间区间精确切片对齐视频与状态、定位某 episode 的数据/视频文件、
基于 episode 统计识别离群 episode（如某关节范围异常、count 与 length 不符）。

---

## 14. 数据集级聚合统计（`meta/stats.json`）

**种类**：全数据集逐字段聚合。每个字段含
`min / max / mean / std / count / q01 / q10 / q50 / q90 / q99`，覆盖全部状态/动作/时间戳/索引字段，
以及图像键的像素统计（`camera_observations.color_images.*`）。

**作为检测输入可做**：作为归一化/标准化参数、全局合理范围参考、离群检测基线、
`count` 是否等于 `total_frames`（样本 395660）。

---

## 15. 任务表（`meta/tasks.jsonl`）

每行 `{"task_index": N, "task": "任务名"}`。样本组：`{"task_index": 0, "task": "test01"}`。

**作为检测输入可做**：`task_index` ↔ 任务名映射校验、与 `info.json.total_tasks`、
`metadata.language_instruction` 的一致性核对。

---

## 16. 中文子任务标注（`labels/labels.json`）

**种类**：每个交付 episode 的子任务分段标注（中文）。

结构：
```json
{ "labels": [
  { "episode_index": 129,
    "key_frame": [],
    "subtasks": [ {"start_frame": 0, "end_frame": 284, "label": "移动右臂抓取书籍"}, ... ] }
]}
```

字段：
- `episode_index`：所属 episode。
- `key_frame`：关键帧列表（**样本中为空数组**，需兼容存在但为空）。
- `subtasks[]`：`start_frame` / `end_frame`（帧号区间）+ `label`（中文子任务名）。

样本（整理书籍上架）子任务示例：移动右臂抓取书籍 / 移动右臂将书籍叠放 / 双臂协作对齐书的边缘 /
移动右臂将书籍放置到书架对应孔位 / 移动左臂抓取文件夹 / 机械臂归位 等。

**作为检测输入可做**：帧号是否越界（`end_frame < length`）、区间是否单调/无异常重叠、
是否每个交付 episode 都有标注、子任务序列是否含“归位”收尾、标注语言/内容是否与任务匹配。

---

## 17. 使用建议（写检测工具时）

1. **以组为单位**：遍历到 `success/lerobot_RoboMIND/`，先读 `meta/info.json` 取本组真实参数。
2. **不要硬编码** fps / 编码 / episode 数 / 是否有标定 —— 全部因组而异，逐组读取。
3. **三方帧数一致性**是高价值低成本检查：`episodes.jsonl.length` = parquet 行数 = mp4 帧数。
4. **跨模态对齐**：用 `episodes_stats.jsonl` 的 `from/to_timestamp` 与 parquet 的 `*.timestamp`。
5. **优雅降级**：标定缺失（25/500）、`key_frame` 为空、某些机器人型号无 `head_position` 等都要兼容。
6. **声明 vs 实际**：`info.json` 的 feature schema / 视频规格是“声明值”，应与 parquet/ffprobe 实测比对。
