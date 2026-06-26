# VidInspect Agent

自动化视频数据质检 Agent —— 对视频数据集进行批量元数据校验、完整性检测与质量评估，并输出结构化质检报告。

## 功能

- **元数据检查**：分辨率、帧率、时长、编码格式、文件大小
- **完整性检查**：文件可读性、帧数一致性、损坏帧检测
- **质量启发式**：黑屏/静帧比例、模糊度采样（可扩展）
- **时序质检（移植自 video_quality_pipeline 三件套）**：
  - `static`：机械臂静止 / 无效操作检测。`lite` 后端（默认，纯 CPU，相邻帧 L1 帧差 →
    motion_score）；`raft` 后端（GPU 稠密光流 active_ratio，更准，需 RAFT 源码 + 权重）
  - `dup_frame`：复制帧伪装高帧率导致的卡顿检测（带 fps 归一化阈值，覆盖规范「画面掉帧/慢放」）
  - `jump`：跳帧 / 瞬移检测（局部归一化帧差 + 绝对幅度/孤立尖峰守卫）
- **质检规范专项检测器**：
  - `endpoint_static`：开始/结束归位停留时间过长（首尾静止 > 2s，自适应阈值）
  - `freeze`：画面卡死 / 长时间卡帧（单段最长冻结时长 > 2s）
  - `noise`：严重噪点（Immerkær 噪声方差估计）
  - `brightness`：画面过暗 / 欠曝（规范20子项，全画面平均亮度中位数 < 阈值即命中；黑屏归 `visual`、花屏归 `integrity`，偏色/白平衡误报率高暂不做）
- **多模态检测器（默认关闭，可插拔后端）**：
  - `gripper_offscreen`：夹爪出境（规范12）。`mode=image`（默认，本地抽帧→逐帧判定→代码算
    最长连续出镜时长 >1s 命中）或 `mode=video`（整段视频交模型返回出镜区间，仅 gemini）；
    `provider=gemini|openai` 可切换。需安装对应后端（`".[gemini]"` / `".[openai]"`）并设
    `GEMINI_API_KEY` / `OPENAI_API_KEY`。
  - `regrasp`：二次抓取（规范1）。本地抽帧→模型逐帧、逐夹爪判 `side`/`holding`→代码侧按
    机械臂（left/right/single）分别去抖后统计该臂的独立「持有段」数，某臂 ≥2 段（被真释放隔开）
    即命中。每只机械臂单目标，双臂各抓一次（含 A→B 交接）正常、不误报。`provider=gemini|openai` 可切换。
  - `object_slip`：物体滑落（规范21）。与 `regrasp` 共用架构但判据不同（非其子集）——
    模型逐帧额外回 `gripper_closed`，代码侧逐臂去抖后看「持有结束时夹爪是否仍闭合」：
    仍闭合却脱手即滑落，张开放下为正常。`provider=gemini|openai` 可切换。
  - `colormatch`：操作物与桌面同色（规范19）。静态属性、无需时序——本地少量抽帧→模型逐帧判
    「被操作物体是否与桌面颜色相近、难分辨位置和大小」→代码侧在可识别物体的帧里按「难分辨」
    占比 ≥ 阈值判定。`provider=gemini|openai` 可切换。
- **LeRobot 信号交叉验证（§3 小优化，自动生效、无 LeRobot 数据则降级）**：当视频位于 LeRobot
  组内时，摄入层注入的真实信号会增强上述检测器——
  - `metadata`：新增 `spec_match`，把 `info.json` 声明的 codec/分辨率/fps/pix_fmt/has_audio 与 ffprobe 实测交叉核对；
  - `dup_frame`：fps 归一化优先用 `info.json` 声明帧率（更权威）；
  - `endpoint_static`：用 puppet 关节首尾静止时长（地面真值）替代脆弱的像素自适应阈值，并标注末尾「归位」子任务；
  - `freeze`：腕部相机画面冻结但对应臂关节在动 → 判画面/关节不一致（规范18）；
  - `brightness`：用 `stats.json` 像素均值作每数据源亮度基线，替代写死阈值 40；
  - `jump`：`robot` 阈值表经摄入层自动按机型生效，另有可选的关节跳变交叉验证（默认关闭）。
  逐帧关节读取需 `pyarrow`（`pip install -e ".[lerobot]"`）；缺数据 / 缺依赖一律退回纯像素行为。
- **Agent 编排**：可插拔 Checker 流水线，支持自定义规则与阈值
- **报告输出**：JSON / 终端表格，便于接入 CI 或数据平台

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 检查单个视频
vidinspect inspect /path/to/video.mp4

# 批量检查目录
vidinspect inspect /path/to/dataset --recursive --output report.json
```

## 依赖

- Python 3.10+
- [FFmpeg / FFprobe](https://ffmpeg.org/)（需在 PATH 中可用）
- `numpy`、`opencv-python-headless`（时序质检 static / dup_frame / jump 使用）
- `google-genai`（gemini 后端）或 `openai`（openai 后端）：`gripper_offscreen` 夹爪出境检测使用，
  `pip install -e ".[gemini]"` 或 `".[openai]"`；需设置对应 `GEMINI_API_KEY` / `OPENAI_API_KEY`，
  并在 `config` 把 `checks.gripper_offscreen` 设为 `true`
- `torch`、`decord`、`easydict`（`static` 的 `raft` GPU 后端使用）
  - RAFT 后端还需 RAFT 源码（含 `core/raft.py`）与 `raft-sintel.pth` 权重；
    通过 `config` 的 `static.raft_repo` / `static.raft_model_path` 指定，或设置
    `RAFT_DIR` / `WORLDARENA_DIR` 环境变量。缺失时 `raft` 后端会安全降级为 `warn`，
    默认的 `lite` 后端无需 GPU/RAFT 即可运行。

## 项目结构

```
vidinspect-agent/
├── config/default.yaml      # 默认质检规则与阈值
├── src/vidinspect_agent/
│   ├── agent.py             # Agent 编排入口
│   ├── pipeline.py          # Checker 流水线
│   ├── lerobot.py           # LeRobot 数据集摄入层（组级信息注入 metadata）
│   ├── models.py            # 数据模型
│   └── checkers/            # 各类质检器
└── tests/
```

## 配置

编辑 `config/default.yaml` 调整阈值，例如最小分辨率、允许帧率范围、黑屏比例上限等。

`checks` 段可逐项开关 `static` / `dup_frame` / `jump`；各检测器在 `config/default.yaml`
中有独立配置块（阈值、采样参数、命中严重级别 `severity: warn|fail` 等）。三个检测器
默认在命中时报 `warn`（不影响整体 `passed`，与黑屏检测一致），需要硬失败时将对应
`severity` 改为 `fail`。`jump` 可通过 `thresholds` 按机器人标签自定义阈值。

`static` 默认 `backend: lite`（纯 CPU）；改为 `backend: raft` 可启用 GPU 稠密光流后端
（更准，`raft_thr` 默认 0.10），需提供 RAFT 源码与权重路径，否则自动降级为 `warn`。

## 检测器原理

各检测器的算法原理、关键公式、阈值与判定逻辑详见
[`docs/detectors.md`](docs/detectors.md)。数据集可用信息清单见
[`docs/dataset_inputs.md`](docs/dataset_inputs.md)；引入这些多模态信息后各检测器的
改造影响评估（大变更 / 小优化 / 不用变）见
[`docs/detector_dataset_impact.md`](docs/detector_dataset_impact.md)。

## 开发

```bash
pytest
ruff check src tests
```

## License

MIT
