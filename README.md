# VidInspect Agent

自动化视频数据质检 Agent —— 对视频数据集进行批量元数据校验、完整性检测与质量评估，并输出结构化质检报告。

## 功能

- **元数据检查**：分辨率、帧率、时长、编码格式、文件大小
- **完整性检查**：文件可读性、帧数一致性、损坏帧检测
- **质量启发式**：黑屏/静帧比例、模糊度采样（可扩展）
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

## 开发

```bash
pytest
ruff check src tests
```

## License

MIT
