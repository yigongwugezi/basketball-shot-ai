# Release Ball Annotation Plan - 2026-06-18

## 背景

项目已经确认：

- 当前系统 release 更接近 `release_pose_frame`。
- 严格 `ball_release_frame` 需要判断球最后脱离投篮手指尖后的第一帧。
- 之前 ball release debug experiment 和 ROI scan experiment 都证明：当前瓶颈是 detector 对 release 邻域小球召回不足。
- 因此下一步不是继续调 release scoring 或 ROI 参数，而是准备一个小而精准的 release 邻域小球标注集。

## 第一批标注候选

| clip_id | 用途 | 标注优先级 | 备注 |
| --- | --- | --- | --- |
| `BILI_001_A` | strict ball_release_frame + detector | 最高 | 严格真值优先样例 |
| `BILI_003_A` | strict ball_release_frame + detector | 最高 | 上半身慢动作，适合观察手指离球 |
| `BILI_005_A` | strict ball_release_frame + detector | 最高 | 罚球慢动作，适合严格 release 复核 |
| `BILI_001_C` | detector supplement | 中 | 小球检测补充 |
| `BILI_002_A` | detector supplement | 中 | 后侧面，适合补 detector 样本 |
| `BILI_006_A` | detector supplement | 中 | 侧面慢动作，但不作为最严格 release 真值 |
| `BILI_008_A` | detector supplement | 中 | 小球检测补充 |
| `BILI_010_A` | detector supplement | 中 | 小球检测补充 |
| `BILI_010_B` | detector supplement | 中 | 小球检测补充 |
| `BILI_010_D` | detector supplement | 中 | 小球检测补充 |

## 第一批不建议标注

- `BILI_001_B`：侧面但远景多人，球太小，release 邻域小球标注价值偏低。
- `BILI_004_A`：后侧面且遮挡明显，不利于可靠球框和真值标注。
- `BILI_006_B`：后方偏侧面，球和手容易被身体遮挡。
- `BILI_006_C`：后方偏侧面续段，可见性一般。
- `BILI_010_C`：低优先级且人多干扰。

## 标注范围

- 每个 clip 优先标 `release_pose_frame` 或当前系统 `release` 附近前后 15 帧。
- 每个 clip 约 31 帧。
- 第一批 10 个 clip 约 310 帧。
- 不需要标整段视频。
- 不需要标所有人物。

## 每帧标注内容

每一帧标：

- `ball box`
- `ball center`，可以由 `ball box` 中心计算
- 是否球清晰可见
- 是否遮挡
- 是否可能误判

## 每个 clip 标注内容

每个 clip 标：

- `release_pose_frame`
- `ball_release_frame`
- `ball_release_frame` 置信度：`high` / `medium` / `low`
- 画面质量：`clear` / `usable` / `poor`
- 是否可用于 detector 训练
- 是否可用于 strict release 真值验证

## 数据用途

1. 判断现有 detector 为什么在 release 邻域 0 召回。
2. 为后续微调 basketball detector 准备小样本。
3. 为 strict `ball_release_frame` 提供真值。
4. 暂不用于动作评分模型。

## 当前决策

- 现在建议开始人工标注。
- 现在不建议训练 detector。
- 等完成第一批标注后，再判断是否需要微调 detector。
- 现在不改 `backend/main.py`。
- 现在不做 3D pose、SAM2、MediaPipe Hands 或动作建议规则引擎。
