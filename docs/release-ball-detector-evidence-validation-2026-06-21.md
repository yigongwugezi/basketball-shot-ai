# Release Ball Detector Evidence 前后端验收记录

## 1. 验收目标

本次验收目标是确认 fine-tuned release ball detector 已经以可选 evidence 的形式接入后端，并且前端可以展示 Release Ball Detector Evidence 卡片。

## 2. 当前提交

- 后端可选 evidence 接入已完成
- 前端 evidence 卡片展示已完成
- 最新前端展示提交：`16a7819 frontend: show release ball detector evidence`

## 3. 启用方式

- 默认关闭时保持旧 release 逻辑
- 启用环境变量：
  - `ENABLE_RELEASE_BALL_DETECTOR=true`
  - `RELEASE_BALL_MODEL_PATH=<本地 best.pt 路径>`
- 模型文件 `best.pt` 不提交到仓库

## 4. 本地验收环境

- 后端入口：`backend/main.py`
- 常用启动方式：

```powershell
..\venv310\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8020
```

- 由于 `8020/8021` 曾有旧 uvicorn/Python 进程占用，本次临时使用过 `8899` 并成功验收
- 成功访问地址：`http://127.0.0.1:8899`

## 5. 人工验收结果

页面展示的 Release Ball Detector Evidence 卡片结果如下：

- `status = ok`
- `detector_type = release_ball_yolo`
- `release_frame_index = 218`
- `best_frame.frame_index = 218`
- `best_frame.confidence = 0.956`
- `best_frame.distance_to_release = 0`
- `frames` 总数 = 7
- 命中帧数 = 7

这表明后端 `release_ball_evidence` 返回成功，前端 Evidence 卡片展示成功。

## 6. 不提交内容

以下内容不提交：

- `tmp/`
- `runs/`
- `best.pt`
- `last.pt`
- `datasets/processed/`
- 生成图片
- 生成视频
- 训练产物

## 7. 当前结论

- `release ball detector` 已完成从训练、离线评估、后端 evidence 接入、前端展示到人工验收的最小闭环
- 当前 detector 只作为 evidence，不替代旧 release 选择逻辑
- 该能力已经具备展示和答辩说明价值
- 后续可以继续做“融合 release ball detector 结果参与 release frame 决策”，但应作为下一阶段增强，不混入本次验收提交
