# Ball ROI Scan Experiment - 2026-06-18

## 背景

项目已经确认：

- 当前 release 输出更接近 `release_pose_frame`
- 严格 `ball_release_frame` 需要判断球最后脱离投篮手指尖后的第一帧
- 之前 ball release debug experiment 证明现有 ball detection + shooting wrist 不足以产生有效 `candidate_ball_release_frame`
- 本次实验用于确认：上一轮 wrist-centered ROI 是否裁偏；如果换成 wrist 上方 / release corridor / 多尺寸 ROI，能否恢复 release 窗口 ball detection

## 实验范围

只测试：

- `BILI_003_A`，系统 release frame 513，人工 `ball_release_frame` 约 517±1
- `BILI_005_A`，系统 release frame 218，人工 `ball_release_frame` 约 223-224

每个 clip 测试 `release_frame - 15` 到 `release_frame + 15`，共 31 帧。

## ROI 扫描配置

记录：

- ROI 类型：`wrist_center`、`wrist_above`、`wrist_above_large`、`forearm_corridor`、`upper_release_zone`
- ROI 尺寸：`220x220`、`280x280`、`340x340`
- 放大倍数：`2.0x`、`3.0x`
- `conf`：`0.05`、`0.1`
- `imgsz`：`640`、`960`、`1280`，以实际实验记录为准
- 每个 clip 合计 120 个配置

## 实验结果

### BILI_003_A

- 最佳配置只是名义最佳，因为所有配置均为 `0/31`
- best tie config：`wrist_center`，`220x220`，`2.0x`，`conf=0.05`，`imgsz=640`
- `frames_with_ball`：0
- `coverage`：0.000
- `longest_consecutive_hit`：0
- `nonzero_configs`：0/120
- 观察：ROI 没有明显裁偏，球在很多帧里落在 ROI 内，但 detector 仍然 0 命中

### BILI_005_A

- 最佳配置只是名义最佳，因为所有配置均为 `0/31`
- best tie config：`wrist_center`，`220x220`，`2.0x`，`conf=0.05`，`imgsz=640`
- `frames_with_ball`：0
- `coverage`：0.000
- `longest_consecutive_hit`：0
- `nonzero_configs`：0/120
- 观察：ROI 没有明显偏到球外，球在 release 邻域很多帧就在框里，但 detector 仍然完全没有召回

## 结论

本次 ROI 参数扫描没有解决 ball detection 缺失问题。

- 当前问题不是 ROI 裁偏几十像素
- 当前 detector 对 Bilibili release 邻域小球召回不足
- 不建议继续 ROI 参数路线
- 不建议把 ROI 扫描脚本或实验代码合入主线
- 不建议现在改 `backend/main.py`
- 不建议继续调 `wrist_y` / `elbow_angle` 权重
- 不建议现在把 ByteTrack / BoT-SORT 接入主线，因为 tracker 需要检测器先产生可用框

## 后续建议

1. 优先换更清晰、更适合 strict `ball_release_frame` 的样例。
2. 如果肉眼能清楚看到球，但当前 detector 仍然 0 召回，则开始做 release 邻域小球标注。
3. 标注目标应聚焦：
   - release 前后 15 帧
   - `ball box` / `ball center`
   - `ball_release_frame`
   - 画面质量备注
4. 之后再评估是否微调一个面向 Bilibili 小球 release 场景的 basketball detector。
5. 暂不进入 3D pose、SAM2、MediaPipe Hands 或动作建议规则引擎。
