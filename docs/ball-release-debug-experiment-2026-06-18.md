# Ball Release Debug Experiment - 2026-06-18

## 背景

项目已区分：

- `release_pose_frame`：当前系统输出更接近该概念
- `ball_release_frame`：严格定义为球最后脱离投篮手指尖后的第一帧

本次实验目标是验证：是否可以只用现有 ball detection + YOLO pose shooting wrist，在 release 附近窗口内通过 ball-wrist distance 连续变化推断严格 `ball_release_frame`。

## 实验方式

以当前预测 release frame 为中心，检查前后约 15 帧窗口。

每帧记录：

- `ball_detected`
- `ball_center`
- `shooting_wrist`
- `ball_wrist_distance`
- `distance_delta`
- `ball_moving_away`
- `pose_detected`
- `visible_keypoints`

本次实验只做诊断，不替换现有 release 逻辑。

## 验证样例

### BILI_001_A

- 系统原 release frame：55
- 人工 ball_release_frame：55
- `candidate_ball_release_frame`：null
- `candidate_confidence`：0.0
- `candidate_reason`：`ball track insufficient: window 内有效 ball-wrist distance 少于 5 帧`
- `coverage`：`frames_with_ball=4`，`frames_with_wrist=31`，`frames_with_distance=4`
- 结论：原 release 已准确；debug helper 没有提供额外价值。

### BILI_003_A

- 系统原 release frame：513
- 人工 ball_release_frame：约 517
- `candidate_ball_release_frame`：null
- `candidate_confidence`：0.0
- `candidate_reason`：`ball track insufficient: window 内有效 ball-wrist distance 少于 5 帧`
- `coverage`：`frames_with_ball=0`，`frames_with_wrist=31`，`frames_with_distance=0`
- 结论：当前 ball detection 在 release 窗口完全缺失，无法推断严格 `ball_release_frame`。

### BILI_005_A

- 系统原 release frame：218
- 人工 ball_release_frame：约 224
- `candidate_ball_release_frame`：null
- `candidate_confidence`：0.0
- `candidate_reason`：`ball track insufficient: window 内有效 ball-wrist distance 少于 5 帧`
- `coverage`：`frames_with_ball=0`，`frames_with_wrist=31`，`frames_with_distance=0`
- 结论：当前 ball detection 在 release 窗口完全缺失，无法推断严格 `ball_release_frame`。

## 总结

本次实验说明：

- `shooting_wrist` 在 3 个样例中较稳定
- ball detection 在 release 附近窗口不连续
- 现有 ball detection + wrist distance 不足以支撑严格 `ball_release_frame` 判断
- 不应把本次 debug helper 合入主线
- 当前不建议继续调整 `wrist_y` / `elbow_angle` 权重
- 后续优化严格 `ball_release_frame`，应优先解决 `ball track continuity`

## 后续建议

1. 暂不改现有 release 算法。
2. 保留当前系统 release 作为 `release_pose_frame`。
3. 如果要继续做严格 `ball_release_frame`，应优先研究：
   - 更稳定的篮球检测
   - YOLO tracking / ByteTrack / BoT-SORT
   - 更高分辨率或局部 ROI ball detection
   - 必要时再评估 hand/fingertip landmark
4. 不建议现在接 SAM2、3D pose 或训练新模型。
5. 不建议把本次失败 helper 代码合入主线。
