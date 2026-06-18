# YOLO 训练记录

## 环境

使用 Python 3.10 虚拟环境：

```powershell
py -3.10 -m venv .venv310
.venv310\Scripts\python -m pip install ultralytics
```

检查环境：

```powershell
.venv310\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

当前状态：

- PyTorch：2.12.0 CPU
- CUDA：False

## 数据集

统一 YOLO 数据集：

```text
datasets/processed/yolo_ball_rim_player/data.yaml
```

类别：

- `ball`
- `rim`
- `player`

生成命令：

```powershell
python scripts/prepare_yolo_ball_rim_player.py
```

## 冒烟训练

用于验证训练链路：

```powershell
.venv310\Scripts\yolo.exe detect train model=yolo11n.yaml data=datasets\processed\yolo_ball_rim_player\data.yaml epochs=1 imgsz=320 batch=4 workers=0 fraction=0.03 project=runs name=ball_rim_player_smoke exist_ok=True
```

注意：

- `yolo11n.yaml` 是随机初始化，不适合正式效果评估。
- 正式训练应该使用预训练权重 `yolo11n.pt` 或更合适的检测模型。
- 当前 GitHub 下载预训练权重失败，后续可手动下载到项目根目录再训练。

## 正式训练建议

CPU 训练会很慢。若有 NVIDIA GPU，优先使用 GPU 环境。

有预训练权重后可尝试：

```powershell
.venv310\Scripts\yolo.exe detect train model=yolo11n.pt data=datasets\processed\yolo_ball_rim_player\data.yaml epochs=50 imgsz=640 batch=8 workers=0 project=runs name=ball_rim_player_yolo11n
```

如果仍然只有 CPU，建议先用：

```powershell
.venv310\Scripts\yolo.exe detect train model=yolo11n.pt data=datasets\processed\yolo_ball_rim_player\data.yaml epochs=10 imgsz=416 batch=4 workers=0 fraction=0.2 project=runs name=ball_rim_player_cpu_trial
```

