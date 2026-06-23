# Release Ball Detector v2 数据工具调研

## 1. 结论摘要

当前阶段可以使用候选数据量产、自动预标注和人工审核流水线提速工具，帮助我们更快搭建 v2 数据治理与审核流程。但不能直接把外部数据或自动标签当作 trusted 训练集。所有外部数据和自动标签都必须进入 manifest，并标记 `label_source`、`review_status`、`trusted_status`。项目仍按专业产品级标准推进，只是当前学生研究、比赛阶段允许 research-only 实验。

## 2. 工具分层

- Supervision：主线 QA / 可视化工具
- LocateAnything：research-only 自动预标注工具
- Roboflow Universe：candidate external dataset 来源
- RF-DETR core：v2 detector benchmark 候选模型
- MotionBricks：长期动作生成 / 虚拟教练研究池

## 3. LocateAnything 使用边界

LocateAnything 当前阶段可用于 research-only 辅助预标注 basketball bbox，也可用于临时实验、流程验证和人工审核提速。其输出标签必须标记：

- `label_source=locateanything_auto`
- `trusted_status=quarantine` 或 `research_only`
- `commercial_use=no`

LocateAnything 产出不进入 v2 trusted train，不用于产品级评估结论。未来商业化时，要剔除、重标或使用合规链路替换。建议目录只作为实验池，例如 `datasets/experimental/locateanything_prelabel/`，但本次不创建目录。

## 4. Supervision 使用计划

Supervision 应优先用于 label 可视化、bbox 合法性检查、contact sheet、视频帧抽样 QA、detector 输出可视化。后续可以做一个 QA 工具，读取图片和 YOLO label，输出可视化审核图。它不替代人工审核，只提高审核效率，最适合马上接入 v2 manifest QA。

## 5. Roboflow Universe 使用计划

Roboflow Universe 用来找候选篮球数据集。每个数据集必须记录 `dataset_name`、`url`、`license`、`image_count`、`classes`、可能用途和风险。它们只能进入 `candidate_external_dataset` 池，必须经过 `license_check`、`sample_preview`、`human_review`、manifest 分层后，才能决定是否使用。不能默认所有 Roboflow 数据都可商用或适合 release ball detector。

## 6. RF-DETR 使用计划

RF-DETR core 只在 v2 trusted split 和独立 test 建好之后再训练，用于和 YOLO11n baseline 做对照实验。当前不替换 YOLO v1 detector。暂不使用 Plus、XL、2XL，除非单独确认许可、成本和推理速度。评估必须使用独立 test，不使用 train 回测。

## 7. MotionBricks 使用计划

MotionBricks 当前不用于 basketball bbox 标注、release ball detection 或 release frame 判定，先放入长期研究池。未来可能用于动作模板、3D 参考、虚拟教练、synthetic motion research，但当前阶段不增加主线复杂度。

## 8. 建议数据量产流水线

建议流程如下：

`candidate_external_dataset`
→ `license_check`
→ `sample_preview`
→ `auto_label_or_import`
→ `supervision_contact_sheet`
→ `human_review`
→ `manifest trusted/quarantine/rejected`
→ `v2 split`
→ `training`
→ `independent_test_evaluation`

## 9. P0 试验建议

P0 阶段建议先做 LocateAnything research-only 小实验。只选 1-2 个用户自己确认的视频，只抽 20-50 张出手附近帧，prompt 优先使用 `basketball`，人工审核框是否准确。只记录实验结论，不进入 trusted。下一步再用 Supervision 生成 contact sheet 辅助审核。

## 10. 下一步

下一步应先做 LocateAnything research-only 小实验方案，再做 Supervision QA / contact sheet 脚本。不要直接训练新 detector，也不要把实验数据混入正式 annotations。
