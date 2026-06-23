# AI 投篮动作分析项目技术路线与工具决策记录

## 1. 项目标准

项目当前主要用于学生研究、老师项目、比赛展示、原型验证。但技术标准仍按专业产品级推进，不按普通学生作业标准降低要求。所有临时实验、数据来源、许可风险都要留痕、隔离、可替换。当前阶段允许 research-only 实验，但不能污染未来 trusted 数据链路。

## 2. 当前 release ball detector 状态

v1 detector 继续作为 prototype evidence。`release_fusion` diagnostic 已完成后端字段与前端展示。v1 不作为产品级主算法，`BILI_005_A` 不再作为泛化证明。产品级结论等待 v2 trusted split 和独立 test。当前目标是先让检测、解释、数据治理链路完整，而不是夸大 detector 泛化能力。

## 3. 数据治理原则

所有样本进入 manifest。所有样本必须标记 `source`、`label_source`、`review_status`、`trusted_status`。数据按 `trusted / quarantine / rejected` 分层。`train / val / test` 必须按 `source_video_id` 隔离，`test` 不参与训练和调参。外部数据不能直接进 trusted。训练样本不能拿来做产品级泛化证明，后续结论必须基于独立 test。

## 4. LocateAnything 决策

当前学生研究、比赛阶段可将 LocateAnything 用作 research-only 辅助预标注工具，也可用于临时实验、流程验证和人工审核提速。其输出必须标记 `label_source=locateanything_auto`。`trusted_status` 必须是 `quarantine` 或 `research_only`，并明确 `commercial_use=no`。它不进入 v2 trusted train，不作为未来商业产品训练数据，也不作为产品级评估结论依据。未来商业化时，应剔除、重标或用合规链路替换。这里不是因为学生项目降低标准，而是有意识地把研究阶段工具和正式产品数据链路隔离。

## 5. Supervision 决策

Supervision 作为主线工具优先使用。它适合用于 label 可视化、bbox QA、contact sheet、视频帧抽样、detector 输出可视化和人工审核辅助，也可服务 v2 manifest QA 和人工审核流程。它适合尽早接入，帮助提高数据审核效率和可解释性，但不替代人工审核，只辅助审核和可视化。

## 6. Roboflow Universe 决策

Roboflow Universe 用于寻找候选篮球数据集。每个 dataset 必须单独记录 URL、license、classes、image_count、用途和风险，只进入 candidate external dataset 池，不直接进入 trusted train。任何候选数据都必须经过 `license_check`、`sample_preview`、`human_review`、manifest 分层，不能默认 Roboflow Universe 上所有数据都可商用或适合 release ball detector。它更适合用于扩充候选数据来源，而不是直接量产 trusted 数据。

## 7. RF-DETR 决策

RF-DETR core 后续作为 YOLO11n 的 v2 detector benchmark。必须等 v2 trusted split 和独立 test 建好后再训练，不现在直接替换 YOLO。暂不碰 Plus、XL、2XL，除非单独确认许可、成本和推理速度。比较指标必须基于独立 test，而不是训练回测。目标是建立 YOLO11n baseline vs RF-DETR core 的可信对照实验。

## 8. MotionBricks 决策

MotionBricks 暂不进入 release ball detector 主线，先放入长期研究池。它未来可能用于动作模板、3D 动作参考、虚拟教练或 synthetic motion research，但当前不用于篮球 bbox 标注或 release frame 检测，也不应在当前阶段增加主线复杂度。

## 9. 后续路线

1. 完成 manifest schema 文档并 push。
2. 写数据工具调研文档。
3. 做 LocateAnything research-only 小实验。
4. 做 Supervision QA/contact sheet 工具。
5. 生成 v1 manifest 草稿。
6. 人工审核 source_video/clip。
7. 生成 trusted/quarantine/rejected。
8. 生成 v2 split plan。
9. 训练 YOLO11n v2 baseline。
10. 训练 RF-DETR core 对照实验。
11. 独立 test 评估。
12. 再决定 detector 是否参与 release_fusion 主决策。

## 10. 信息沉淀原则

后续 ChatGPT、Codex 讨论决定的重要技术路线、数据原则、工具边界、许可判断，都应沉淀到 docs，不能只留在聊天记录里。这样更方便换对话、团队协作、答辩和长期维护。每个阶段结束后，应优先补文档再继续堆功能。重要实验必须记录输入、输出、数据来源、是否 trusted、是否可用于产品结论。
