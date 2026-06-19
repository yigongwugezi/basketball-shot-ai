# Bilibili 素材 release 验证报告

## 背景

之前 `basketball51` 验证只能作为系统稳定性测试。当前 Bilibili 裁剪素材更接近 AI 投篮动作分析 MVP 的真实输入，因此本报告基于 `datasets/bilibili_clip_plan.csv` 中已下载并已裁剪完成的 Bilibili 片段，验证当前 release 出手帧识别效果。

本次验证不修改算法、不调整权重、不引入新模型，只使用当前 `/api/analyze` 接口。

## 使用的视频列表

从 `datasets/bilibili_clip_plan.csv` 中优先选择：

- `priority = high`
- `download_status = downloaded`
- `clip_status = cut_done`
- `output_file_path` 真实存在
- 角度优先 `side` / `back-side` / `side-upper-body`
- notes 看起来是真实投篮、慢动作、侧面/后侧面素材

最终验证 8 个 Bilibili 裁剪片段：

- `BILI_001_A`
- `BILI_002_A`
- `BILI_003_A`
- `BILI_004_A`
- `BILI_005_A`
- `BILI_006_A`
- `BILI_006_B`
- `BILI_006_C`

## 逐片段结果

| clip_id | title 简述 | source_type | view_angle | API | 五阶段 | release | time | method | conf | evidence 摘要 | 人工判断 | 偏差估计 | 质量备注 |
|---|---|---|---|---|---|---:|---:|---|---:|---|---|---|---|
| `BILI_001_A` | 库里全身侧面固定镜头 | `pro_fixed_full_body` | `side` | 成功 | 有 | 55 | 1.833s | `pose_trajectory_score` | 0.82 | wrist_y 66.3, elbow 153.1, kp 16, no ball | 准确 | 约 0 帧 / 0s | 侧面、全身较清楚、固定镜头，球偏小但可判断 |
| `BILI_002_A` | 库里后侧面投篮 | `coach_analysis_amateur` | `back-side` | 成功 | 有 | 45 | 1.500s | `pose_trajectory_score` | 0.95 | wrist_y 127.0, elbow 174.0, kp 15, ball 44.6, moving away | 偏晚 | 约 6-12 帧 / 0.2-0.4s | 后侧面远景，球较小，但比 basketball51 更可判断 |
| `BILI_003_A` | 库里罚球上半身侧面慢动作 | `pro_slow_motion_detail` | `side-upper-body` | 成功 | 有 | 513 | 17.100s | `pose_trajectory_score` | 0.78 | wrist_y 85.9, elbow 147.0, kp 12, no ball | 准确 | 约 0-6 帧 / 0-0.2s | 上半身慢动作，手和球清楚，是很好的 release 观察样本 |
| `BILI_004_A` | 库里运球衔接投篮后侧面 | `pro_slow_motion_off_dribble` | `back-side` | 成功 | 有 | 147 | 5.069s | `pose_trajectory_score` | 0.75 | wrist_y 125.8, elbow 98.5, kp 12, ball 59.8 | 无法判断 | 无法估计 | 背侧遮挡明显，身体挡住球和手腕 |
| `BILI_005_A` | 库里完整罚球慢动作侧面 | `pro_free_throw_slow_motion` | `side` | 成功 | 有 | 218 | 8.720s | `pose_trajectory_score` | 0.72 | wrist_y 108.3, elbow 139.1, kp 15, no ball | 准确 | 约 0-6 帧 / 0-0.24s | 侧面、全身入镜、慢动作，适合 MVP 标准测试 |
| `BILI_006_A` | 利拉德三分慢动作侧面 | `pro_slow_motion_dual_view` | `side` | 成功 | 有 | 142 | 4.733s | `pose_trajectory_score` | 0.81 | wrist_y 46.1, elbow 156.3, kp 16, no ball | 偏晚 | 约 6 帧 / 0.20s | 侧面、固定机位、全身清楚，球不够稳定可见 |
| `BILI_006_B` | 利拉德后侧面慢动作 | `pro_slow_motion_dual_view` | `back-side` | 成功 | 有 | 109 | 3.633s | `pose_trajectory_score` | 0.81 | wrist_y 17.5, elbow 145.6, kp 14, no ball | 无法判断 | 无法估计 | 后侧面，手和球被身体遮挡，release 难确认 |
| `BILI_006_C` | 利拉德后侧面慢动作续段 | `pro_slow_motion_dual_view` | `back-side` | 成功 | 有 | 155 | 5.167s | `pose_trajectory_score` | 0.74 | wrist_y 52.9, elbow 118.6, kp 16, no ball | 无法判断 | 无法估计 | 后侧面，球与手腕可见性一般，release 不好严格判断 |

## 汇总统计

- 总验证视频数：8
- 成功分析数量：8
- release 基本准确数量：3
- release 偏早数量：0
- release 偏晚数量：2
- 无法判断数量：3

## 结论

Bilibili 素材比 `basketball51` 更适合 MVP 投篮动作分析验证，尤其是慢动作、固定机位、侧面或正侧面的裁剪片段。它们更接近真实产品输入，也更容易人工判断 release 是否接近真正出手瞬间。

当前 release 算法在 Bilibili 素材上能稳定跑通，没有破坏五阶段返回，也能给出 `selection_method`、`confidence`、`evidence`。在可判断样本里，当前没有看到稳定偏早，更像是存在轻微偏晚倾向。

当前 release 算法不建议马上改。现阶段更应该围绕更清晰、固定机位、球可见的标准样例继续验证。只有在后续更清晰固定机位样本中仍然稳定偏晚时，才考虑最小算法修正。

## 最适合作为 MVP 标准测试样例

优先推荐：

- `BILI_001_A`：侧面、全身、固定镜头，适合基础 release 验证。
- `BILI_003_A`：上半身侧面慢动作，适合观察手腕/球离手，但不适合全身指标。
- `BILI_005_A`：侧面完整罚球慢动作，适合作为 MVP 标准样例。
- `BILI_006_A`：利拉德侧面慢动作，适合作为跨球员泛化样例，但当前算法略偏晚。

谨慎使用：

- `BILI_002_A`：后侧面远景，能验证稳定性，但 release 容易偏晚。
- `BILI_004_A`、`BILI_006_B`、`BILI_006_C`：背侧遮挡较多，不适合严格 release 标注，只适合稳定性测试。

## 后续建议

1. 暂不修改 release scoring。
2. 先围绕 `BILI_001_A`、`BILI_003_A`、`BILI_005_A`、`BILI_006_A` 建立 MVP 标准验证集。
3. 继续收集更适合动作分析的真实训练视频：
   - 全身入镜
   - 固定机位
   - 球清晰可见
   - 出手动作连续
   - 最好侧面或正侧面
4. 如果后续清晰固定机位样本仍稳定偏晚，再只做一个最小算法调整：
   - 增加轻微“越晚越扣分”的时间惩罚
   - 或优先选择 dip 后第一个高腕点 / 大肘角峰值附近候选
5. 不要在当前证据下接入 3D pose 或动作建议规则引擎。

## 是否建议写入 docs

建议写入 `docs`。这批 Bilibili 素材比 `basketball51` 更接近项目真实 MVP 输入，值得沉淀成正式验证记录，作为后续 release 算法是否调整的依据。

## 是否建议 commit / push

建议提交并推送本 Markdown 报告与 `AI_PROGRESS.md` 的摘要更新。不要提交任何原始视频、裁剪视频、大文件或 B 站视频内容本体。
