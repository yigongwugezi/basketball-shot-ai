# Release Ball Detector v2 数据治理方案

## 1. v1 当前问题

v1 目前存在几类结构性问题。第一，没有独立 `test` 集，当前只有 `train` 和 `val`。第二，离线评估口径不纯，`all evaluation` 包含了 `train` 样本，本质上属于回看训练集。第三，`BILI_005_A` 明确进入训练集，却又被拿来做验收叙事，存在训练污染验收的问题。第四，`NEW` 原始素材到 processed 结果的映射复现性不足，长期依赖临时报告链路，不够稳定。第五，样本质量不稳定，包含清晰、可用、较差、不可见以及不同程度遮挡的混合素材。第六，当前没有正式的 `trusted/quarantine` 治理层，很多样本只是“能不能用”而不是“是否可信”。第七，split 规则是脚本里硬编码的，缺少可审计、可复现、可扩展的数据治理定义。

## 2. v2 目标

v2 的目标不是单纯扩大数据量，而是建立可信、可追溯、可复验的数据版本。原始视频必须可追溯到唯一来源，样本必须经过人工确认可信，`train/val/test` 必须按 `source_video_id` 隔离，`test` 在训练和调参期间完全只读，不参与任何阈值选择和样本选择，`untrusted/quarantine` 必须明确隔离，产品结论只允许基于独立 `test`。

## 3. 建议 manifest 字段

建议 v2 manifest 至少包含以下字段：

- `source_video_id`
- `source_video_path`
- `source_video_sha256`
- `clip_id`
- `frame_index`
- `image_path`
- `label_path`
- `split`
- `trusted_status`
- `action_type`
- `shot_type`
- `camera_view`
- `quality`
- `occlusion`
- `ball_visible`
- `release_pose_frame`
- `strict_ball_release_frame`
- `annotator`
- `reviewer`
- `review_status`
- `notes`

建议补充以下字段，便于后续治理和复现：

- `dataset_version`
- `source_batch`
- `source_type`
- `camera_motion`
- `fps_bucket`
- `resolution_bucket`
- `fallback_flag`
- `low_confidence_flag`
- `person_id_or_group`
- `scene_id`
- `quarantine_reason`

## 4. trusted / quarantine / rejected 规则

`trusted` 的进入条件应尽量严格：标准投篮、球清晰、人工复核通过、`strict_ball_release_frame` 可判断、原视频可追溯的样本才进入 `trusted`。`quarantine` 用来承接上篮、全场远景、白帧、fallback、未人工复核、`low confidence`、严重遮挡、来源不完整等样本，这些样本可以保留研究价值，但不应直接进入主训练集。`rejected` 则用于来源不明、标签明显错误且无法修复、非目标任务且没有研究价值的样本，避免污染后续版本。

## 5. split 规则

split 不能按帧随机切。同一个 `source_video_id` 只能落在一个 split 里，不能跨 `train/val/test` 泄漏。`test` 必须包含新人物、新角度、新场景，用来检验泛化能力。`val` 只用于有限调参，不能替代独立测试。训练期间 `test` 只能只读，不能参与调参、样本筛选或阈值选择。即使样本量小时，也不能牺牲 `test` 的独立性。

## 6. v2 最小可执行路线

1. 先建 manifest schema 文档
2. 从 `batch_001` / `batch_003` 生成 v1 manifest 草稿
3. 人工审核 `source_video` / `clip`
4. 标记 `trusted` / `quarantine` / `rejected`
5. 生成 v2 split plan
6. 写转换脚本 v2
7. 训练 detector v2
8. 独立 `test` 评估

## 7. v1 detector 使用边界

v1 可以继续作为 prototype evidence 和 `release_fusion` diagnostic 链路验证，但不应作为产品级主算法。`BILI_005_A` 也不应继续被当作泛化证明。产品结论必须等待 v2 独立 `test`，不能再依赖训练内样本或回看式评估。

## 8. 下一步建议

下一步最小任务不是直接训练，而是先写 manifest schema，并把治理字段、审核状态、split 规则和可信度分层先固定下来。等 schema 稳定后，再从 `batch_001` / `batch_003` 生成草稿 manifest，随后进入人工审核和 split 规划。
