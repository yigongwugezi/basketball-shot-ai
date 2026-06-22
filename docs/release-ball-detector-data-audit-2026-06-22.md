# Release Ball Detector 数据可信度审计

## 1. 当前结论

v1 detector 可以作为 prototype evidence 使用；暂不能作为产品级主算法；recall=1.000 不能作为产品泛化证明；建议冻结 v1，建立 v2 数据版本。

## 2. 数据集、脚本和配置

正式标注包括 `batch_001` 和 `batch_003` `labels.csv`；转换脚本是 `scripts/prepare_yolo_release_ball.py`；YOLO 配置是 `datasets/processed/yolo_release_ball/data.yaml`；追溯表是 `metadata.csv`；训练参数为 YOLO11n、单类 `ball`、CPU、20 epochs、640px、batch 2；只有 `train/val`，没有正式 `test`。

## 3. 训练集来源

`train` 共 310 帧，263 positive、47 negative，来自 10 个原始视频：`BILI_001_A`、`BILI_005_A`、`NEW_001/IMG_7212.MOV`、`NEW_002/IMG_7215.MOV`、`NEW_003/IMG_7216.MOV`、`NEW_004/IMG_7218.MOV`、`NEW_005/IMG_7219.MP4`、`NEW_006/IMG_7221.MOV`、`NEW_009/IMG_7226.MP4`、`NEW_010/IMG_7227.MP4`。每个 clip 31 个连续 release 邻域帧，没有逐帧随机拆分。

## 4. 验证集和测试集来源

`val` 共 62 帧，42 positive、20 negative，来自 `BILI_003_A_BV1d84y1G7zq.mp4` 和 `NEW_012/IMG_7235.MOV`。没有独立 `test`。离线评估实际评估全部 372 张 processed 图片，包含 `train` 与 `val`，不是独立测试集。`metadata` 可追溯 `source_batch`、`clip_id`、`frame_index`；`NEW` 原始文件映射只在 `tmp` 报告中，长期复现性不足。

## 5. basketball51 / 2p0 / BILI_005_A

`basketball51` 和 `2p0` 未进入训练、`val` 或带 GT 的离线评估；`2p0` 只用于后端/API 验证，无人工 `ball bbox`，不能算正式 recall。`BILI_005_A` 明确进入 `train`，frame 203-233，也进入离线全量评估。frame 218 是 `release_pose_frame=yes`，不是 strict ball release；strict 人工真值是 frame 224。`BILI_005_A` frame 218 的 API 成功只能证明接入有效，不能证明泛化。

## 6. 泄漏和评估污染风险

未发现同一 clip 跨 `train/val`；未发现相邻帧跨 `train/val`；372 张图片 SHA-256 无完全重复。但 `train` 与 `all evaluation` 明确重叠，属于训练集回测。`BILI_005_A` 同时用于训练和 API 验收，存在明显评估污染。BILI `train/val` 均为库里素材，人物和拍摄风格相近。未做感知哈希，无法完全排除近重复。

## 7. 标签质量风险

标签格式为 YOLO：`0 x_center y_center width height`。305 个 positive 标签、67 个 empty negative 标签。未发现错误类别、零宽高或明显越界。发现 6 个标签重建后顶部轻微越界，约半像素：`BILI_005_A 224/225`、`BILI_003_A 522/523`、`NEW_005 333/335`。原因是转换脚本允许中心点约 1 像素误差，并使用取整人工 `center`，而非由 `bbox` 重新计算中心。`batch_003` 有最终人工 QC；`batch_001` 没有同等级正式独立双人复核 manifest。

## 8. 视频内容质量风险

数据并非全部是清晰标准投篮。positive 中 `clear 98`、`usable 179`、`poor 28`；遮挡 `none 80`、`partial 197`、`heavy 28`。`NEW_004`、`NEW_006` 是 `game/follow-ball` 素材，初筛曾标为 `not_recommended`；`NEW_004` 有两张疑似白帧；`NEW_006` 是 `fallback window`，仅 6 positive、25 negative；`NEW_012` 是低置信度困难样本，release 帧 `ball_visible=no`。很多 `NEW` 素材是移动机位、竖屏、约 30 FPS，需要人工抽样看图。

## 9. recall=1.000 的可信度判断

这是内部小样本结果，不代表产品泛化。release exact：11 positive，其中 `train 10`、`val 1`；release ±1：32 positive，其中 `train 29`、`val 3`；release ±3：73 positive，其中 `train 65`、`val 8`。`val release exact` 的 1.000 实际只有 `1/1`；`val release ±3` 的 1.000 实际只有 `8/8`。更完整 `val` 效果是 `recall@0.5=39/42=0.929`。

## 10. 当前 detector 使用边界

可以作为 prototype evidence 使用，可以继续服务于后端 evidence、前端展示、`release_fusion diagnostic` 链路验证；暂不可以作为产品级主算法；暂不可以用 `BILI_005_A` 或训练窗口作为泛化证明；需要重建 v2 数据版本并补独立 `test`。

## 11. 下一步建议

冻结当前模型和 v1 split。建立正式 manifest：原视频文件名/哈希、clip、frame、split、标注者、复核者、质量、置信度。按原始视频、人物、拍摄场景划分 `train/val/test`，`test` 在训练和调参期间完全隔离。将未人工看过、`fallback`、白帧、上篮、非标准投篮标记为 `untrusted/quarantine`。v2 核心训练先使用人工确认的标准投篮，困难样本单独分层保留。转换时由 `bbox` 重新计算中心，修复半像素边界问题。产品结论以独立 `test` 为准。
