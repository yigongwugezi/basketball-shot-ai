# AI_PROGRESS

## 当前项目状态

AI 投篮动作分析 MVP 已经跑通基础链路：

- 前端上传投篮视频并请求后端分析。
- 后端抽取关键帧，运行 YOLO pose 骨架检测。
- 后端运行篮球、篮筐、球员检测管线。
- 前端展示 setup、dip、release、follow_through、landing 关键帧，并支持点击放大。
- 已有基础 2D 角度和拍摄机位可信度提示。

## 本次修改

本次只优化 release 出手帧识别，不新增模型、不新增数据库、不重构项目架构。

release 识别从原来的“手腕 y 最小 + 肘角较大”升级为多信号打分：

- 投篮侧手腕越高，越接近 release。
- 投篮侧肘角越大，越接近 release。
- 可见关键点越多，候选帧越可信。
- 如果检测到 ball，则球心与投篮侧手腕越近越加分。
- 如果下一采样帧里球与手腕距离变大，作为球开始远离手腕的辅助证据。
- release 必须晚于 dip。
- release 后必须还能找到 follow_through 候选帧。
- 有效骨架帧不足时回退到 FRAME_PLAN 固定比例。

## 修改文件

- `backend/main.py`
- `shot-analyzer-prototype/app.js`
- `AI_PROGRESS.md`

## 后端接口变化

`/api/analyze` 的现有结构保持不变，`frames` 仍包含：

- `setup`
- `dip`
- `release`
- `follow_through`
- `landing`

每个 frame 新增可选字段：

- `selection_method`
- `confidence`
- `evidence`

release 帧在有效骨架帧足够时返回：

- `selection_method: "pose_trajectory_score"`
- `confidence: 0 到 1`
- `evidence`: 包含 wrist_y、elbow_angle、visible_keypoints、ball_detected、ball_wrist_distance、ball_moving_away、score

fallback 时 release 帧返回：

- `selection_method: "fallback_ratio"`
- `confidence: 0.3`
- `evidence: "有效骨架帧不足，使用固定比例兜底"`

## 前端展示变化

关键帧主图信息区新增一行选择说明：

- selection method
- confidence
- 简短 evidence

如果后端没有返回这些字段，前端会保持原展示，不会报错。

## 当前风险 / 已知问题

- 仍然是 2D 画面投影估计，不是真实 3D 出手瞬间。
- ball 检测来自现有检测管线，可能漏检或误检，只作为辅助信号。
- 采样帧数量有限，release 仍可能落在真实出手前后若干帧。
- 为了让 release 识别利用 ball 信号，采样阶段会额外运行目标检测，分析耗时可能略有增加。
- 当前文本与部分历史文件可能存在编码显示问题，不影响本次算法字段结构。

## 手动测试步骤

1. 启动后端：

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8020 --reload
```

2. 打开前端：

```text
http://127.0.0.1:8020/
```

3. 上传一段投篮视频并点击分析。

4. 检查 `/api/analyze` 返回的 `frames` 是否包含 5 个关键帧。

5. 检查 release frame 是否包含：

```json
{
  "selection_method": "pose_trajectory_score",
  "confidence": 0.0,
  "evidence": {}
}
```

或在骨架不足时包含：

```json
{
  "selection_method": "fallback_ratio",
  "confidence": 0.3,
  "evidence": "有效骨架帧不足，使用固定比例兜底"
}
```

6. 检查前端关键帧主图信息区是否能显示 selection 和 confidence，且点击放大仍可用。

## 下一步建议

- 用 5 到 10 段真实投篮视频记录 release 选择结果，和人工标注帧做对比。
- 如果 release 偏早，降低 wrist_y 权重，提高 ball_moving_away 或 elbow 权重。
- 如果 release 偏晚，提高 wrist_y 权重，并限制 release 不能离最高腕点太远。
- 后续可以把采样结果作为 debug 数据单独返回给开发模式，但不建议现在扩大前端结构。

## 真实视频验证结果

验证日期：2026-06-18

后端启动方式：

```powershell
.venv310\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8020 --reload
```

说明：

- 系统默认 `python` 指向 Python 3.14，未安装 `ultralytics`，直接运行 `python -m uvicorn ...` 会失败。
- 使用项目 `.venv310` 后端可以正常启动，`/api/health` 返回模型文件存在。
- 本次使用本地已有样例视频，不下载新数据。

### 视频 1

- 文件：`datasets/samples/basketball51/2p0_2p0_v165_010726_x264.mp4`
- `/api/analyze`：成功
- metadata：180 帧，29.97 fps
- frames：`setup,dip,release,follow_through,landing`
- release frame_index：119
- release timestamp：3.971s
- release selection_method：`pose_trajectory_score`
- release confidence：0.72
- release evidence：

```json
{
  "wrist_y": 90.5,
  "elbow_angle": 149.3,
  "visible_keypoints": 17,
  "ball_detected": false,
  "ball_wrist_distance": null,
  "ball_moving_away": false,
  "score": 0.72
}
```

人工观察：

- 远景比赛转播，人物和球较小。
- release 帧接近投篮出手/球离手区间。
- 判断：基本准确，但由于画面太远，精确偏差无法严格判断。

### 视频 2

- 文件：`datasets/samples/basketball51/3p0_3p0_v121_004247_x264.mp4`
- `/api/analyze`：成功
- metadata：151 帧，25.00 fps
- frames：`setup,dip,release,follow_through,landing`
- release frame_index：119
- release timestamp：4.760s
- release selection_method：`pose_trajectory_score`
- release confidence：0.89
- release evidence：

```json
{
  "wrist_y": 143.7,
  "elbow_angle": 172.2,
  "visible_keypoints": 12,
  "ball_detected": true,
  "ball_wrist_distance": 49.6,
  "ball_moving_away": true,
  "score": 0.89
}
```

人工观察：

- 远景比赛转播，人物和球较小。
- release 帧落在出手后的随球阶段附近。
- 判断：略偏晚，粗略估计偏晚约 6 帧，约 0.24 秒。

### 前端验证

- `http://127.0.0.1:8020/` 可以打开。
- 初始页面控制台未发现 error/warn。
- 当前浏览器自动化接口不支持 `setInputFiles`，系统文件选择框也没有把路径成功带回页面。
- 因此本次未完成真实前端上传验收，不能声称“前端上传完整通过”。
- 后端 `/api/analyze` 已用 2 个真实视频完整跑通，新增 release 字段可正常返回。

人工补充验收：

- 用户已在 in-app browser 中手动完成真实前端上传验收。
- 页面能打开，能选择真实投篮视频，能生成分析报告。
- 关键帧大图正常显示，点击放大正常。
- `setup`、`dip`、`release`、`follow_through`、`landing` 都正常显示。
- `selection_method`、`confidence`、`evidence` 已在关键帧信息区可见。
- 指标区正常显示。

### 当前结论

- 后端 release 识别改动可运行，且不会破坏 `frames` 五阶段返回。
- 多信号打分字段和 evidence 返回正常。
- 第二个视频出现轻微偏晚，后续需要更多真实样例对权重做小步校准。
- 前端真实上传展示已经由用户人工验收通过。
- 当前仍存在问题：第二个测试视频 release 略偏晚约 6 帧，后续需要继续优化。
- 当前建议：可以本地 commit，暂不 push。

补充记录：

- 2026-06-18 已完成 8 个 `basketball51` 样例验证。
- `/api/analyze` 全部跑通。
- 统计结果：1 个准确，0 个偏早，2 个偏晚，5 个无法判断。
- 结论：暂不改算法，先收集更清晰固定机位训练视频。
- 详细记录见 `docs/release-validation-2026-06-18.md`。

Bilibili 素材验证补充：

- 2026-06-18 已完成 8 个 Bilibili 裁剪片段 release 验证。
- `/api/analyze` 全部跑通。
- 统计结果：3 个准确，0 个偏早，2 个偏晚，3 个无法判断。
- 推荐 MVP 标准样例：`BILI_001_A`、`BILI_003_A`、`BILI_005_A`、`BILI_006_A`。
- 结论：暂不改 release 算法，先围绕这些标准样例继续验证。
- 详细记录见 `docs/bilibili-release-validation-2026-06-18.md`。

Release 定义修正补充：

- contact sheet 人工复核后，项目正式区分 `release_pose_frame` 和 `ball_release_frame`。
- 当前算法输出的 release 更接近 `release_pose_frame`。
- 严格 `ball_release_frame` 定义为球最后脱离投篮手指尖后的第一帧。
- `BILI_001_A`：系统预测 55，人工 `ball_release_frame` 55，准确。
- `BILI_003_A`：系统预测 513，人工 `ball_release_frame` 约 517±1，严格口径偏早约 3-5 帧。
- `BILI_005_A`：系统预测 218，人工 `ball_release_frame` 约 223-224，严格口径偏早约 5-6 帧。
- `BILI_006_A`：无法可靠判断，不作为严格真值样例。
- 结论：暂不改算法；后续若优化严格 `ball_release_frame`，应围绕 `ball-hand separation`，而不是简单加时间惩罚。

Ball release debug experiment 补充：

- 2026-06-18 进行了一次只读诊断实验，尝试用现有 ball detection + shooting wrist 在 release 附近窗口推断严格 `ball_release_frame`。
- 验证样例：`BILI_001_A`、`BILI_003_A`、`BILI_005_A`。
- 结果：3 个样例均未得到有效 `candidate_ball_release_frame`。
- 主要原因：ball detection 在 release 附近窗口不连续，`BILI_003_A` 和 `BILI_005_A` 窗口内完全没有检测到 ball。
- 结论：当前瓶颈是 ball track continuity，不是 wrist keypoint。
- 决策：不把 debug helper 合入主线，暂不改 release 算法；后续如果优化严格 `ball_release_frame`，应优先解决 ball tracking。

Ball ROI scan experiment 补充：

- 2026-06-18 完成 `BILI_003_A` / `BILI_005_A` 的本地 ROI 参数扫描。
- 每个 clip 测试 120 个 ROI 配置。
- 两个 clip 均为 `0/120` 非零配置。
- ROI debug 图显示 ROI 未明显裁偏，球在很多帧内位于 ROI 中，但 detector 仍然 `0` 召回。
- 结论：当前瓶颈是 detector 对 release 邻域小球召回不足，不是 ROI 参数。
- 决策：不继续 ROI 参数路线，不改主线代码；下一步优先换更清晰样例或准备 release 邻域小球标注集。

Release ball annotation plan 补充：

- 2026-06-18 已完成第一批 release 邻域小球标注候选筛选。
- 推荐 10 个候选：`BILI_001_A`、`BILI_001_C`、`BILI_002_A`、`BILI_003_A`、`BILI_005_A`、`BILI_006_A`、`BILI_008_A`、`BILI_010_A`、`BILI_010_B`、`BILI_010_D`。
- 严格 `ball_release_frame` 真值优先样例：`BILI_001_A`、`BILI_003_A`、`BILI_005_A`。
- detector 训练补充样例：`BILI_001_C`、`BILI_002_A`、`BILI_006_A`、`BILI_008_A`、`BILI_010_A`、`BILI_010_B`、`BILI_010_D`。
- 当前素材足够做第一批约 10 clip / 约 310 帧的小标注集。
- 决策：建议开始人工标注；暂不训练 detector；暂不改主线代码。
- 详细计划见 `docs/release-ball-annotation-plan-2026-06-18.md`。

## Release ball annotation batch 001 supplement

- Completed manual release-neighborhood ball annotation for the first batch of 3 core clips on 2026-06-18.
- Annotated clips: BILI_001_A, BILI_003_A, BILI_005_A.
- Total: 93 frames, including 71 frames with a target shooter ball box and 22 frames without a target ball.
- Strict `ball_release_frame` recorded:
  - BILI_001_A = 55, high
  - BILI_003_A = 517, medium
  - BILI_005_A = 224, high
- Annotation data saved to `datasets/annotations/release_ball_batch_001/labels.csv`.
- Decision: keep this as a small annotation set first; do not train the detector yet; do not change the mainline algorithm.
## Release ball detector eval batch 001 supplement

- Completed an offline failure analysis on `release_ball_batch_001` using the current detector logic.
- Reused `backend.main.detect_frame()` from the production code path.
- Recall on the 71 manual target-shooter ball frames was 0.
- The three strict release frames were all missed:
  - BILI_001_A = 55
  - BILI_003_A = 517
  - BILI_005_A = 224
- Among the 22 `ball_visible=no` frames, 4 still produced detections, which look more like background balls or false positives.
- Conclusion: the blocker is detector recall on small balls near release, not release scoring, wrist keypoints, ROI tuning, or tracking.
- Decision: do not change `backend/main.py`; do not keep tuning release weights; next step is to annotate the remaining 7 clips or prepare detector fine-tuning.
