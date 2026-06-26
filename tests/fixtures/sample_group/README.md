# 样例 LeRobot v3.0 组（只读文本元数据）

本目录是一组**真实** LeRobot v3.0 数据的**文本元数据快照**，仅供查阅真实字段结构 / shape / 取值之用，
不参与依赖二进制的检测。源组为：

```
tienkung_tabletop_soft_gripper_5episode_v2.1/
  tienkung_station_dualArm-gripper-3cameras_9/
    tienkung_station_dualArm-gripper-3cameras_09_D-O-02_03_20260326/
      success/lerobot_RoboMIND/
```

## 包含什么

```
meta/
├── info.json            # 数据集总览 + feature schema + 相机参数 + 采集信息
├── episodes.jsonl       # 每 episode 索引/长度/任务/平台ID
├── episodes_stats.jsonl # 每 episode 统计量 + 数据/视频定位指针
├── stats.json           # 全数据集逐字段聚合统计
└── tasks.jsonl          # 任务索引 ↔ 任务名
labels/
└── labels.json          # 中文子任务分段标注
```

各字段含义见 [`docs/dataset_inputs.md`](../../../docs/dataset_inputs.md)。

## 故意不包含什么（以及为什么）

- `videos/chunk-000/<camera_key>/episode_*.mp4`：三路彩色视频，整组约 **374M**，不入库以免拖垮
  仓库体积 / clone / CI。
- `data/chunk-000/episode_*.parquet`：逐帧状态/动作，约 **3.1M**，同样不入库。

要跑**需要真实视频 / parquet 的集成测试或手动验证**，请直接指向本机完整数据（不要复制）：

```
/home/houxianzhou/devbox-media/unify-fileset/lerobot_kai/tienkung_tabletop_soft_gripper_5episode_v2.1/.../success/lerobot_RoboMIND
```

> 这组数据已通过人工检测（路径含 `success/`），可作为检测器的正样本 / 基线，理想情况下应几乎全部判 PASS。
