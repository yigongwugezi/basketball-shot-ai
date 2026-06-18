# Datasets

本目录用于保存 AI 投篮动作分析项目的数据集方案、manifest、原始视频、抽帧结果、标注和模型文件。

当前最重要的文件：

- `basketball-shot-dataset-plan.md`：数据集总方案。
- `manifest.example.json`：自建投篮视频 manifest 示例。

建议本地目录：

```text
datasets/
  raw/
    public/
    self_collected/
  processed/
    frames/
    clips/
    annotations/
  models/
    yolo_ball_rim/
    pose/
  splits/
```

注意：

- 不要把大视频文件直接提交到 Git。
- 未授权公开视频不要直接用于商业训练。
- 自建视频要记录授权、拍摄角度、帧率、命中结果和关键事件。
