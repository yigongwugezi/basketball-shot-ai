# LocateAnything Research-only 预标注小实验方案

## 1. 实验目的

当前项目仍按专业产品级标准推进。本实验只用于学生研究、比赛阶段的 research-only 辅助预标注验证。目标是验证 LocateAnything 是否能在出手附近帧稳定定位 basketball bbox，从而减少前期人工标注压力。本实验不产生 v2 trusted 数据，不作为产品级模型训练依据。

## 2. 实验边界

LocateAnything 输出只能进入 research-only / quarantine，不进入 v2 trusted train，不用于独立 test，不用于产品级泛化结论。未来商业化时必须剔除、重标或用合规链路替换。所有结果必须记录 `label_source=locateanything_auto`。

## 3. 输入样本选择

先选 1-2 个用户自己确认可信的视频。每个视频抽取 20-50 张出手附近帧，优先覆盖：

1. release 前，球还在手上
2. strict release，球刚完全离手
3. release 后，球在空中
4. 球小、模糊、遮挡的困难帧

不使用 `BILI_005_A` 作为泛化证明。不使用未人工确认的比赛远景或上篮视频作为正向结论样本。

## 4. 推荐 prompt

优先尝试：

- `basketball`
- `ball`
- `orange basketball`
- `basketball in player's hand`

可选尝试：

- `basketball hoop`
- `rim`
- `player`

但本实验核心只评估 basketball bbox。

## 5. 人工审核记录表

建议用下面的 Markdown 表格记录实验结果：

| sample_id | source_video_id | frame_index | prompt | detected | bbox_quality | false_positive | missed | ball_visibility | occlusion | reviewer | review_status | notes |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|
| 示例 | 示例 | 0 | basketball | yes | accurate | no | no | yes | none | reviewer_01 | accepted_for_research | 示例记录 |

bbox_quality 建议取值：

- `accurate`
- `slightly_off`
- `wrong_object`
- `missed`
- `unusable`

review_status 建议取值：

- `accepted_for_research`
- `needs_manual_fix`
- `rejected`

## 6. manifest 标记建议

如果后续把结果落到 manifest，必须标记：

- `label_source=locateanything_auto`
- `trusted_status=quarantine` 或 `research_only`
- `commercial_use=no`
- `review_status=reviewed` / `needs_fix` / `rejected`
- `notes=research_only_do_not_use_for_product_training`

## 7. 成功/失败判断标准

如果 20-50 张中大多数出手附近帧能准确框到 basketball，则值得进入下一步 Supervision contact sheet + 临时训练流程。如果漏检、误检、框偏移严重，则只保留为研究记录，不继续投入。即使效果很好，也不自动升级为 trusted 数据。

## 8. 下一步

方案文档提交后，下一步才抽取 `tmp/locateanything_sample_frames` 中的临时截图。截图和 LocateAnything 输出不提交到 GitHub。人工审核结果可作为 docs 或 manifest 草稿记录。后续用 Supervision 生成 contact sheet 辅助审核，暂不训练新 detector。
