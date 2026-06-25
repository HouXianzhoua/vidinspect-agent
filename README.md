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
[`docs/detectors.md`](docs/detectors.md)。

## 开发

```bash
pytest
ruff check src tests
```

## License

MIT
