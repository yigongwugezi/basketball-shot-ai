# LocateAnything Research-only 预标注小实验结果

## 1. 实验范围

本次实验只用于学生研究和比赛阶段的 research-only 预标注验证，不产生 v2 trusted 数据，不进入训练集或独立 test，也不作为产品级泛化结论。

测试视频：

`E:\BasketballShotAI\raw\bilibili\自用库里完整罚球慢动作赏析投篮手侧后方视角\1-自用库里完整罚球慢动作赏析投篮手侧后方视角-480P 标清-AVC.mp4`

临时抽帧目录：

`tmp/locateanything_sample_frames/curry_free_throw_side_back/`

测试 prompt：

`basketball`

## 2. 已测试帧

本次人工测试到 frame 528，包含以下 7 张出手附近帧：

| frame_index | 结果 | 备注 |
|---:|---|---|
| 341 | 可用 | basketball bbox 无明显问题 |
| 372 | 可用 | basketball bbox 无明显问题 |
| 403 | 可用 | basketball bbox 无明显问题 |
| 434 | 可用 | basketball bbox 无明显问题 |
| 466 | 略偏小 | 可能因为球员手部遮挡，导致 bbox 没有完整包住篮球 |
| 497 | 可用 | basketball bbox 无明显问题 |
| 528 | 可用 | basketball bbox 无明显问题 |

## 3. 初步结论

LocateAnything 对清晰、近景、出手附近的 basketball bbox 有 research-only 预标注可用性。当前观察到的主要问题是遮挡场景下 bbox 可能偏小，需要人工审核和修正。

本次样本量较小，只能说明它适合继续作为辅助预标注工具做小规模研究验证，不能证明模型具有稳定泛化能力。

## 4. 停止原因

继续测试需要进入付费流程，当前 research-only 验证目的已经达到，因此不建议支付费用继续扩大本次实验。

## 5. 使用边界

- `label_source` 必须标记为 `locateanything_auto`
- `trusted_status` 只能是 `quarantine` 或 `research_only`
- `commercial_use=no`
- 不进入 v2 trusted train
- 不进入独立 test
- 不作为产品级效果结论
- 后续商业化前必须剔除、重标或使用合规链路替换

## 6. 下一步建议

下一步不训练 detector。建议先用 Supervision 做最小 QA/contact sheet 工具，帮助人工快速查看 bbox 是否准确，再决定是否值得继续扩展 research-only 预标注流程。
