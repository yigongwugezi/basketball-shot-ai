# Release Ball Detector 与 Release Frame 选择融合方案

## 1. 当前状态

当前仓库已经完成以下能力：

- 后端可选 `release_ball_evidence` 接入
- 前端 `Release Ball Detector Evidence` 卡片展示
- 产品级验收文档 [release-ball-detector-evidence-validation-2026-06-21.md](/C:/Users/20825/Documents/Codex/2026-05-21/1-2-ai-app-3-ai/docs/release-ball-detector-evidence-validation-2026-06-21.md)

基于 [backend/main.py](/C:/Users/20825/Documents/Codex/2026-05-21/1-2-ai-app-3-ai/backend/main.py) 的真实代码，当前 `release ball detector` 还没有参与最终 `release frame` 选择。当前最终 `release` 仍由旧的姿态/轨迹逻辑决定，`release_ball_evidence` 只是在该帧已经选出之后额外生成并返回，因此它目前只是 evidence，不是 decision maker。

## 2. 当前旧 release 逻辑

当前 `release frame` 的核心选择在 [backend/main.py](/C:/Users/20825/Documents/Codex/2026-05-21/1-2-ai-app-3-ai/backend/main.py) 的 `candidate_frame_indices()` 和 `estimate_release_from_sampled_frames()` 中完成。

旧逻辑的主要输入包括：

- `YOLO pose` 产出的关键点
- `shooting_side`
- `shooting_wrist_y`
- `shooting_elbow_angle`
- `min_knee_angle`
- `visible_keypoints`
- `detect_frame()` 得到的篮球检测结果
- `closest_ball_distance()` 计算的球心到投篮手腕距离

当前 release 评分是一个多信号打分，而不是单点规则。代码里对 release candidate 计算了：

- `wrist_score`
  - 手腕越高越接近 release
- `elbow_score`
  - 肘角越大越接近 release
- `visible_score`
  - 有效关键点越多越可信
- `ball_score`
  - 球越靠近投篮手腕越加分
  - 如果下一采样帧球开始远离手腕，还会额外加一点分

当前总分权重为：

- `wrist_score * 0.38`
- `elbow_score * 0.27`
- `visible_score * 0.20`
- `ball_score * 0.15`

此外还有时序约束：

- `release` 不能早于 `dip`
- `release` 后必须还能找到 follow-through 候选
- `setup -> dip -> release -> follow_through -> landing` 顺序必须成立，否则 fallback

fallback 逻辑在 `fallback_frame_indices()`：

- 按 `FRAME_PLAN` 的固定比例直接回退
- `release.selection_method = "fallback_ratio"`
- `release.confidence = 0.3`
- `release.evidence = "有效骨架帧不足，使用固定比例兜底"`

当前 `selection_method` 含义：

- `release`
  - 正常轨迹逻辑：`pose_trajectory_score`
  - fallback：`fallback_ratio`
- 其他关键帧
  - 正常轨迹逻辑：`pose_sequence`
  - fallback：`fallback_ratio`

当前 `confidence` 含义：

- `release`
  - 来自 `release["score"]`，并被截断到 `0.35 ~ 0.95`
- 其他关键帧
  - 正常情况固定为 `0.5`
  - fallback 为 `0.2`

当前 `evidence` 含义：

- 非 release 关键帧
  - 通常是字符串：`由 release 前后有效骨架帧顺序推断`
- release 关键帧
  - 会被替换成结构化对象，包含：
    - `wrist_y`
    - `elbow_angle`
    - `visible_keypoints`
    - `ball_detected`
    - `ball_wrist_distance`
    - `ball_moving_away`
    - `score`

## 3. 当前 release_ball_evidence 逻辑

当前 `release_ball_evidence` 逻辑位于 [backend/main.py](/C:/Users/20825/Documents/Codex/2026-05-21/1-2-ai-app-3-ai/backend/main.py) 的：

- `release_ball_detector_enabled()`
- `configured_release_ball_model_path()`
- `get_release_ball_model()`
- `best_release_ball_detection()`
- `build_release_ball_evidence()`
- `analyze_video()`

启用方式基于环境变量：

- `ENABLE_RELEASE_BALL_DETECTOR`
  - 通过 `env_truthy()` 解析，支持 `1/true/yes/on`
- `RELEASE_BALL_MODEL_PATH`
  - 通过 `os.getenv()` 读取
  - 再转成 `Path(...).expanduser()`

当前状态可能包括：

- `ok`
- `model_missing`
- `disabled_by_missing_model`
- `no_detection`
- `read_failed`
- `error`
- `no_release_frame`

字段含义如下：

- `frames`
  - release 邻域窗口内逐帧 evidence，当前窗口半径为 `3`
  - 每帧包含：
    - `frame_index`
    - `distance_to_release`
    - `confidence`
    - `bbox`
    - `has_detection`
    - `status`
- `best_frame`
  - 当前窗口内最高置信度的篮球检测结果
- `confidence`
  - 指该帧 release-ball detector 的检测置信度，不是旧 release 主逻辑的 `confidence`
- `distance_to_release`
  - 相对旧 release frame 的帧偏移，`0` 表示正好落在旧 release 上

它的生成阶段是在 `analyze_video()` 里：

1. 先调用 `candidate_frame_indices()` 得到旧的关键帧选择结果
2. 再把这批 `frame_selection` 组装成 `frames`
3. 然后仅当 `release_ball_detector_enabled()` 为真时，找到 `key == "release"` 的帧
4. 以这个已经选出的 `release_frame_index` 为中心，调用 `build_release_ball_evidence()`
5. 把结果挂到响应顶层 `response["release_ball_evidence"]`

因此当前 `release_ball_evidence` 和最终 `release frame` 的关系是：

- 它依赖旧 release frame 先存在
- 它围绕旧 release frame 做检测
- 它当前不反向修改 `frame_indices`
- 它当前不覆盖 `selection_method`
- 它当前不覆盖旧 `confidence`
- 它当前只是 evidence

## 4. 为什么不能直接替换旧逻辑

从产品级角度看，当前不适合直接让 detector 替代旧 release 主逻辑，原因包括：

- detector 训练数据量还小
  - 当前 release-ball 训练集规模仍然有限，泛化能力还没有经过大样本验证
- release-ball detector 解决的是“球在哪里/窗口内是否命中”
  - 这并不等于完整动作理解，也不等于已经准确识别了动作阶段
- 视频角度、遮挡、画质、帧率都会明显影响 detector
  - 特别是低帧率、远景、小球、遮挡、跟拍镜头
- 旧 pose-based 逻辑仍有价值
  - 它能利用手腕、肘角、膝角、随球时序这些动作线索
- 直接替换风险较高
  - 一旦 detector 在某些视频里失灵，系统会失去当前姿态逻辑提供的保底稳定性

## 5. 推荐融合策略 v1

推荐的 v1 融合策略应保持保守：

- 默认仍以旧 release 逻辑为主
- 当 `release_ball_evidence.status = ok` 且 `best_frame.confidence` 达到阈值时，只把 detector 当作校准信号
- 如果 detector `best_frame.frame_index` 与旧 `release_frame_index` 距离很近，例如 `±1` 或 `±3` 帧：
  - 不改帧
  - 只提高旧 release 的可信度说明
  - 或在 evidence 中增加 agreement 说明
- 如果 detector 与旧 release 明显冲突：
  - 不直接覆盖
  - 标记 `disagreement`
  - 把冲突暴露给前端或日志
- 只有在以下条件同时满足时，才考虑让 detector 提供 candidate release frame：
  - 旧逻辑 `confidence` 很低
  - detector `best_frame.confidence` 很高
  - `frames` 命中稳定，不是单帧偶然命中
  - 冲突距离不大，且时序仍然合理
- 所有融合结果必须可解释
  - 必须让前端或调试输出明确知道：最终为什么仍选 pose，或者为什么采纳 detector candidate

## 6. 建议新增字段

建议新增但这一步不实现的字段如下：

- `release_fusion`
- `release_fusion.status`
- `release_fusion.final_source`
- `release_fusion.pose_release_frame_index`
- `release_fusion.detector_release_frame_index`
- `release_fusion.frame_delta`
- `release_fusion.agreement_level`
- `release_fusion.reason`
- `release_fusion.risk_flags`

这些字段的设计目标是：

- 不破坏现有 `frames[].selection_method`
- 不隐藏 pose 结果
- 不隐藏 detector 结果
- 用单独的融合对象表达最终决策与风险

## 7. 建议实现步骤

建议按低风险顺序推进：

1. 第一步：只增加 fusion diagnostic，不改变旧 release 输出
2. 第二步：前端展示 pose vs detector 对比
3. 第三步：扩大验证集
4. 第四步：再决定是否让 fusion 影响最终 `release_frame`
5. 第五步：形成可回滚的 feature flag

这样可以保证每一步都可解释、可回滚、可验收，而不是一次性把 detector 硬塞进主路径。

## 8. 验收标准

后续实现融合时，验收标准建议明确为：

- 默认关闭不影响旧行为
- 模型缺失不报错
- detector 与旧逻辑一致时能显示 agreement
- detector 与旧逻辑冲突时能显示 disagreement
- 前端能解释为什么最终选择某一帧
- 不提交训练产物
- `py_compile` / `node --check` 通过

如果未来让 fusion 参与最终决策，还应补充：

- fallback 仍然存在
- 低置信 detector 不得覆盖 pose 结果
- 所有覆盖行为必须可审计

## 9. 当前建议结论

当前更稳妥的产品级路线是：

- 下一阶段先做 `fusion diagnostic`
- 不要立即替换 `release frame` 主逻辑
- 先把 pose 结果、detector 结果、agreement/disagreement 机制做清楚
- 在扩大验证集和人工验收之后，再决定是否让 fusion 真正影响最终 `release frame`

这条路线更符合当前代码事实，也更符合产品级稳定性要求。
