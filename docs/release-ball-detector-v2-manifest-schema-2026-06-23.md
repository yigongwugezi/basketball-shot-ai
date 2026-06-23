# Release Ball Detector v2 Manifest Schema

## 1. 设计目标

v2 manifest 的目标不是单纯记录训练样本，而是把原始视频追溯、人工复核、`trusted/quarantine/rejected` 分层、视频级 `split`、独立 `test` 和产品级评估可信度串成一条可审计的数据链路。相比 v1 只靠脚本把 `labels.csv` 转成 YOLO 训练集，v2 需要让每一行样本都能回到原始视频、原始片段、原始帧和人工审核结论，这样后续训练、调参和验收才有明确边界。

## 2. manifest 粒度

建议 manifest 采用 `frame-level` 粒度，即每一行对应一个具体 frame 样本，而不是只对应一个视频。这样才能精确表达某一帧是否可用于 release 识别、是否通过人工确认、是否属于 `trusted`，以及它最终被分到 `train`、`val`、`test` 还是保留在 `none`。每一行都应该能追溯到 `source_video`、`clip`、`frame`、`image`、`label` 和审核状态，避免只知道“这段视频用了”，却不知道“哪一帧为何可用”。

## 3. 字段定义表

下表给出建议 schema。`是否必填` 里的“是”表示进入正式 `trusted` 数据前应具备；“建议”表示治理上强烈建议补齐，但当前 v1 可能暂时缺失。

| 字段名 | 是否必填 | 建议取值 | 说明 |
|---|---|---|---|
| `dataset_version` | 是 | `v1` / `v2` / `v2.1` | 数据版本号，用于区分不同治理阶段。 |
| `source_video_id` | 是 | 字符串，例如 `BILI_001_A` / `NEW_012` | 原始视频唯一标识，必须可追溯。 |
| `source_video_path` | 是 | 相对路径或绝对路径 | 原始视频文件路径。 |
| `source_video_sha256` | 是 | SHA-256 十六进制字符串 | 原始视频哈希，用于确认来源未变化。 |
| `source_type` | 建议 | `practice` / `game` / `fallback` / `unknown` | 原始素材类型。 |
| `source_batch` | 是 | `release_ball_batch_001` / `release_ball_batch_003` | 标注批次来源。 |
| `clip_id` | 是 | 字符串，例如 `BILI_005_A` | 片段 ID。 |
| `frame_index` | 是 | 整数 | 原始视频中的帧号。 |
| `image_path` | 是 | 路径字符串 | 当前样本图像路径。 |
| `label_path` | 是 | 路径字符串 | 对应标注文件路径。 |
| `split` | 是 | `train` / `val` / `test` / `none` | 数据切分。 |
| `trusted_status` | 是 | `trusted` / `quarantine` / `rejected` | 数据可信层级。 |
| `quarantine_reason` | 建议 | 字符串 | 进入 `quarantine` 的原因说明。 |
| `action_type` | 建议 | `shot` / `layup` / `pass` / `dribble` / `unknown` | 动作大类。 |
| `shot_type` | 建议 | `set_shot` / `jump_shot` / `free_throw` / `three_point` / `mid_range` / `unknown` | 投篮类型。 |
| `camera_view` | 建议 | `side` / `front` / `back` / `diagonal` / `broadcast` / `unknown` | 拍摄视角。 |
| `camera_motion` | 建议 | `static` / `handheld` / `moving` / `follow_ball` / `unknown` | 摄像机运动类型。 |
| `fps_bucket` | 建议 | `lt30` / `30_60` / `gt60` / `unknown` | 帧率区间。 |
| `resolution_bucket` | 建议 | `sd` / `hd` / `fhd` / `uhd` / `unknown` | 分辨率区间。 |
| `quality` | 是 | `clear` / `usable` / `poor` / `unusable` | 画面质量分层。 |
| `occlusion` | 是 | `none` / `partial` / `heavy` | 遮挡程度。 |
| `ball_visible` | 是 | `yes` / `no` / `uncertain` | 球是否可见。 |
| `release_pose_frame` | 建议 | `true` / `false` 或 `yes` / `no` | 是否为 release pose 候选帧。 |
| `strict_ball_release_frame` | 建议 | `true` / `false` 或 `yes` / `no` | 是否为严格球离手帧。 |
| `fallback_flag` | 建议 | `true` / `false` | 是否来自 fallback 窗口或降级链路。 |
| `low_confidence_flag` | 建议 | `true` / `false` | 是否属于低置信样本。 |
| `person_id_or_group` | 建议 | 字符串或分组 ID | 人物或人群分组，用于控制泛化切分。 |
| `scene_id` | 建议 | 字符串 | 场景 ID，用于区分场地、机位或环境。 |
| `annotator` | 建议 | 人名、工号或匿名 ID | 标注人。 |
| `reviewer` | 是 | 人名、工号或匿名 ID | 审核人。 |
| `review_status` | 是 | `pending` / `reviewed` / `needs_fix` / `rejected` | 审核状态。 |
| `notes` | 建议 | 自由文本 | 备注、纠错说明或特殊情况。 |

## 4. 枚举值建议

建议关键枚举统一如下，避免后续脚本和审核表不一致：

- `split`: `train` / `val` / `test` / `none`
- `trusted_status`: `trusted` / `quarantine` / `rejected`
- `action_type`: `shot` / `layup` / `pass` / `dribble` / `unknown`
- `shot_type`: `set_shot` / `jump_shot` / `free_throw` / `three_point` / `mid_range` / `unknown`
- `camera_view`: `side` / `front` / `back` / `diagonal` / `broadcast` / `unknown`
- `camera_motion`: `static` / `handheld` / `moving` / `follow_ball` / `unknown`
- `quality`: `clear` / `usable` / `poor` / `unusable`
- `occlusion`: `none` / `partial` / `heavy`
- `ball_visible`: `yes` / `no` / `uncertain`
- `review_status`: `pending` / `reviewed` / `needs_fix` / `rejected`

## 5. 必填规则

进入 `trusted` 的样本，至少必须具备以下字段：

- `source_video_id`
- `source_video_sha256`
- `clip_id`
- `frame_index`
- `image_path`
- `label_path`
- `trusted_status`
- `split`
- `quality`
- `ball_visible`
- `review_status`
- `reviewer`

如果这些字段不完整，样本不应直接进入正式 `trusted` 训练集，而应先放入 `quarantine` 或 `none`，等待补齐后再人工确认。

## 6. trusted / quarantine / rejected 判定规则

`trusted`：标准投篮、球清晰、人工复核通过、`strict_ball_release_frame` 可判断、原视频可追溯的样本进入 `trusted`。这类样本可以进入主训练链路。

`quarantine`：上篮、全场远景、白帧、fallback、未人工复核、low confidence、严重遮挡、来源不完整等样本进入 `quarantine`。这类样本保留研究价值，但不能直接混入主训练集。

`rejected`：来源不明、标签明显错误且无法修复、非目标任务且无研究价值的样本进入 `rejected`。这类样本不参与训练，也不参与正式评估。

## 7. split 合法性规则

- 同一个 `source_video_id` 不能跨 `split`
- `test` 样本不能参与训练和调参
- `split=none` 用于 `quarantine`、`rejected` 或尚未分配的样本
- 不能按连续帧随机拆分

这些规则的核心是视频级隔离，而不是帧级随机打散。否则同一段视频的相邻帧会同时出现在训练和测试中，评估结果会虚高。

## 8. 与现有 v1 数据的映射

现有 v1 的 `batch_001` 和 `batch_003` 主要字段可以映射如下：

- `source_batch`：直接映射到新 schema 的 `source_batch`
- `clip_id`：直接映射到新 schema 的 `clip_id`
- `frame_index`：直接映射到新 schema 的 `frame_index`
- `image_file`：可映射到 `image_path` 的原始文件名部分
- 转换脚本生成的 `yolo_image_file`：可映射到 `image_path` 的 processed 位置
- `split`：可从当前脚本中的 `SPLIT_BY_CLIP` 规则迁移到新 schema
- `ball_visible`：直接映射到 `ball_visible`
- `is_release_pose_frame`：可映射到 `release_pose_frame`
- `is_ball_release_frame`：可映射到 `strict_ball_release_frame`
- `label_confidence`：可辅助映射到 `low_confidence_flag`
- `ball_visibility_quality`：可映射到 `quality`
- `occlusion`：直接映射到 `occlusion`
- `notes`：直接映射到 `notes`，但需要人工清洗成正式治理备注

以下字段当前无法从正式文件中可靠得到，必须明确标为待补充，不能编造：

- `source_video_sha256`
- `reviewer`
- `person_id_or_group`

另外，`source_video_id` 在当前 v1 文件里只能通过 `clip_id` 和上游素材命名规则间接理解，正式 v2 需要在 manifest 中明确存成独立字段，而不是继续依赖临时映射。

## 9. 示例行

以下是 CSV 风格的**示例**记录，不代表真实数据，也不应伪装成正式样本：

```csv
dataset_version,source_video_id,source_video_path,source_video_sha256,source_type,source_batch,clip_id,frame_index,image_path,label_path,split,trusted_status,quarantine_reason,action_type,shot_type,camera_view,camera_motion,fps_bucket,resolution_bucket,quality,occlusion,ball_visible,release_pose_frame,strict_ball_release_frame,fallback_flag,low_confidence_flag,person_id_or_group,scene_id,annotator,reviewer,review_status,notes
v2,BILI_005_A,/data/raw/BILI_005_A.mp4,sha256-example-1,practice,release_ball_batch_001,BILI_005_A,224,/data/processed/train/images/example_0001.jpg,/data/processed/train/labels/example_0001.txt,train,trusted,,shot,jump_shot,diagonal,static,30_60,hd,usable,heavy,yes,true,true,false,false,group_a,scene_01,ann_01,rev_01,reviewed,示例 trusted shot 样本
v2,NEW_012,/data/raw/NEW_012.MOV,sha256-example-2,practice,release_ball_batch_003,NEW_012,167,/data/processed/val/images/example_0002.jpg,/data/processed/val/labels/example_0002.txt,none,quarantine,fallback_and_low_confidence,shot,unknown,broadcast,moving,lt30,fhd,poor,heavy,yes,false,false,true,true,group_b,scene_02,ann_02,rev_02,reviewed,示例 quarantine fallback 样本
v2,UNKNOWN_001,/data/raw/UNKNOWN_001.mp4,,unknown,release_ball_batch_003,UNKNOWN_001,1,/data/processed/quarantine/images/example_0003.jpg,/data/processed/quarantine/labels/example_0003.txt,none,rejected,source_missing_and_label_invalid,unknown,unknown,unknown,unknown,unknown,unknown,unusable,heavy,uncertain,false,false,false,false,unknown,unknown,ann_03,rev_03,rejected,示例 rejected / unknown 样本
```

## 10. 下一步

下一步不应该直接训练，而应该先：

- 生成 v1 manifest 草稿
- 人工审核 `source_video` / `clip`
- 补 `source_video_sha256`
- 再生成 v2 split plan

等这些基础治理字段稳定后，再进入转换脚本更新和 detector v2 训练。
