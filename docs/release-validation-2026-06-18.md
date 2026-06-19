# Release 出手帧识别验证记录 - 2026-06-18

## 背景

本次验证基于 commit：
a43225955d9d76b920c84f61532e69908ecbcead

该 commit 引入 release 多信号打分，并在 /api/analyze 的 frames 中新增 selection_method、confidence、evidence。

## 验证范围

使用本地已有 datasets/samples/basketball51 下 8 个视频，全部完整请求 /api/analyze。

## 视频列表

- 2p0_2p0_v165_010726_x264.mp4
- 2p1_2p1_v120_005022_x264.mp4
- 3p0_3p0_v121_004247_x264.mp4
- 3p1_3p1_v127_004402_x264.mp4
- ft0_ft0_v150_001722_x264.mp4
- ft1_ft1_v108_012728_x264.mp4
- mp0_mp0_v167_003100_x264.mp4
- mp1_mp1_v133_001252_x264.mp4

## 逐视频结果

| 视频 | API | 五阶段 | release | time | method | conf | evidence 摘要 | 人工判断 | 质量备注 |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| 2p0_2p0_v165_010726_x264.mp4 | 成功 | 有 | 119 | 3.971s | pose_trajectory_score | 0.72 | wrist_y 90.5, elbow 149.3, kp 17, no ball | 准确 | 远景，球太小 |
| 2p1_2p1_v120_005022_x264.mp4 | 成功 | 有 | 79 | 3.295s | pose_trajectory_score | 0.78 | wrist_y 138.8, elbow 157.3, kp 17, no ball | 偏晚，约 6 帧 / 0.25s | 远景，球小 |
| 3p0_3p0_v121_004247_x264.mp4 | 成功 | 有 | 119 | 4.760s | pose_trajectory_score | 0.89 | wrist_y 143.7, elbow 172.2, kp 12, ball 49.6, moving away | 偏晚，约 6 帧 / 0.24s | 远景，球小 |
| 3p1_3p1_v127_004402_x264.mp4 | 成功 | 有 | 90 | 3.003s | fallback_ratio | 0.30 | 有效骨架帧不足，固定比例兜底 | 无法判断 | 统计板画面，无有效投篮 |
| ft0_ft0_v150_001722_x264.mp4 | 成功 | 有 | 103 | 4.296s | pose_trajectory_score | 0.80 | wrist_y 109.0, elbow 176.0, kp 17, no ball | 无法判断 | 罚球前/站位远景，出手不清楚 |
| ft1_ft1_v108_012728_x264.mp4 | 成功 | 有 | 91 | 3.795s | pose_trajectory_score | 0.71 | wrist_y 135.2, elbow 178.6, kp 13, no ball | 无法判断 | 远景，多人遮挡，球难辨 |
| mp0_mp0_v167_003100_x264.mp4 | 成功 | 有 | 139 | 5.797s | pose_trajectory_score | 0.61 | wrist_y 154.4, elbow 161.1, kp 17, no ball | 无法判断 | 采访画面，不是投篮 |
| mp1_mp1_v133_001252_x264.mp4 | 成功 | 有 | 103 | 4.296s | pose_trajectory_score | 0.80 | wrist_y 116.7, elbow 127.7, kp 13, no ball | 无法判断 | 转播切镜头，出手瞬间不连续 |

## 汇总统计

- 总验证视频数：8
- 成功分析数量：8
- release 基本准确数量：1
- release 偏早数量：0
- release 偏晚数量：2
- 无法判断数量：5

## 结论

当前 release 算法值得继续保留。它没有破坏 /api/analyze，8 个视频都稳定返回 setup、dip、release、follow_through、landing，也能返回 selection_method、confidence、evidence。

当前没有看到稳定偏早。在可判断样本里，更像是存在轻微偏晚倾向。

但这批样例里很多是远景、转播切镜、统计板或采访，不适合作为严格精度评估集，所以现在不建议马上改算法。

## 后续建议

1. 不要马上修改 release scoring。
2. 先收集更适合动作分析的真实训练视频：
   - 全身入镜
   - 固定机位
   - 球清晰可见
   - 出手动作连续
   - 最好侧面或正侧面
3. 如果后续清晰样本仍稳定偏晚，再只做一个最小算法调整：
   - 增加轻微“越晚越扣分”的时间惩罚
   - 或优先选择 dip 后第一个高腕点 / 大肘角峰值附近候选
4. 不要在当前证据下接入 3D pose 或动作建议规则引擎。
