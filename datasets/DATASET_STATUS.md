# 数据集状态

更新日期：2026-06-15

## 已下载

### 1. basketball-and-hoop-detection

本地路径：

`datasets/raw/public/roboflow/basketball-and-hoop-detection`

来源：

https://universe.roboflow.com/amrita-hlhw6/basketball-and-hoop-detection/dataset/1

格式：

- YOLOv8

许可证：

- MIT

类别：

- `Basketball Court`
- `Basketball`
- `Net`
- `No ball`

用途：

- 篮球检测。
- 篮网/篮筐相关区域检测。
- 第一版 ball/net 检测 baseline。

### 2. basketball-hoop-ball-and-player

本地路径：

`datasets/raw/public/roboflow/basketball-hoop-ball-and-player`

来源：

https://universe.roboflow.com/personal-project-effcw/basketball-hoop-ball-and-player-5axdt/dataset/1

格式：

- YOLOv8

许可证：

- CC BY 4.0

类别：

- `3pt_area`
- `ball`
- `court`
- `hoop`
- `number`
- `paint`
- `player`

用途：

- 篮球检测。
- 篮筐检测。
- 球员检测。
- 后续统一标签训练。

## 当前统计

本地 `datasets/raw/public/roboflow` 下：

- 图片总数：10,435
- YOLO txt 标注文件总数：10,439

## 注意事项

两个数据集类别体系不同，不能直接粗暴合并训练。后续需要做标签统一：

| 原始类别 | 统一类别建议 |
| --- | --- |
| `Basketball` | `ball` |
| `ball` | `ball` |
| `Net` | `net` 或 `rim_area` |
| `hoop` | `rim` |
| `player` | `shooter_or_player` |
| `Basketball Court` / `court` / `3pt_area` / `paint` | 暂时忽略或后续单独做场地检测 |
| `number` | 暂时忽略 |
| `No ball` | 暂时忽略 |

第一版 YOLO baseline 推荐先训练三个核心类：

- `ball`
- `rim`
- `player`

`net` 可作为后续补充类。

## 下一步

1. 写标签统一脚本，把两个数据集转换到统一类别。
2. 生成 `datasets/processed/yolo_ball_rim_player`。
3. 安装 Ultralytics。
4. 训练 YOLO baseline。
5. 在投篮原型里接入检测结果。

## 已处理数据集

已生成统一标签数据集：

`datasets/processed/yolo_ball_rim_player`

统一类别：

- `ball`
- `rim`
- `player`

统计：

- train：8,874 张图片
- valid：1,031 张图片
- test：530 张图片
- 总图片：10,435
- 总标注框：18,863
- 空标签图片：4,976

说明：

- 空标签图片主要来自原始数据集中被忽略的场地类、无球类，暂时作为负样本保留。
- 后续如果训练效果不好，可以过滤空标签或降低负样本比例。

## 训练冒烟测试

已完成一次 Ultralytics YOLO 冒烟训练：

```powershell
.venv310\Scripts\yolo.exe detect train model=yolo11n.yaml data=datasets\processed\yolo_ball_rim_player\data.yaml epochs=1 imgsz=320 batch=4 workers=0 fraction=0.03 project=runs name=ball_rim_player_smoke exist_ok=True
```

结果目录：

`runs/detect/runs/ball_rim_player_smoke`

说明：

- 当前机器使用 CPU，未检测到 CUDA。
- 这次训练从 `yolo11n.yaml` 随机初始化，且只用 3% 数据跑 1 epoch。
- 指标低是正常的，本次目标只是验证数据格式、标签转换和训练链路。
- 原计划使用 `yolo11n.pt` 预训练权重，但 GitHub 下载时发生 TLS 连接失败，后续可手动下载权重后再正式训练。
