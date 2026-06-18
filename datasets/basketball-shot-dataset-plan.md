# AI 投篮动作分析数据集方案

生成日期：2026-06-15

本文件是本项目的数据集总方案。结论先写在前面：

> 目前公开网上没有一个能直接支撑“专业投篮动作分析”的完整数据集。项目应该采用“公开数据集打底 + 自建投篮视频数据集”的路线。

公开数据集主要用于：

- 篮球检测。
- 篮筐/篮网检测。
- 球员/人体检测。
- 命中/未命中粗分类。
- 生物力学指标参考。

自建数据集用于：

- 投篮关键事件标注。
- 出手瞬间识别。
- 手型分析。
- 球轨迹。
- 命中与未命中动作差异。
- 个人动作稳定性分析。

## 1. 数据集建设原则

### 1.1 不依赖单一公开数据集

投篮动作分析需要同时识别：

- 人体姿态。
- 手型。
- 篮球。
- 篮筐。
- 球轨迹。
- 命中结果。
- 投篮动作阶段。

公开数据集通常只覆盖其中一部分。因此必须组合使用。

### 1.2 第一阶段先服务 MVP

第一阶段不追求手型和 3D 轨迹，只服务：

- 固定机位。
- 单人投篮。
- 关键帧抽取。
- 人体姿态。
- 球/篮筐粗检测。
- 基础报告。

### 1.3 从第一天开始积累自有数据

公开数据只能帮你启动。真正的壁垒来自自有数据：

- 自己拍摄的投篮视频。
- 自己标注的 release、follow-through、landing。
- 不同角度、不同水平、不同命中结果。
- 后续用户上传授权数据。

## 2. 推荐公开数据源

### 2.1 Roboflow：Basketball and Hoop Detection

链接：

https://universe.roboflow.com/amrita-hlhw6/basketball-and-hoop-detection

用途：

- 训练/验证篮球和篮网检测。
- 第一版 YOLO ball/net 检测底座。

公开信息：

- 约 3,809 张图片。
- 任务：Object Detection。
- 类别包括 Basketball、Basketball Court、Net、No ball、Player1、Player2。
- 页面显示 mAP@50 约 79.5%，Precision 约 93.0%，Recall 约 78.6%。
- 许可证：MIT。

项目用途优先级：

> 高。适合第一批球/篮筐检测实验。

注意：

- 类别命名不完全符合本项目，需要统一成自己的标签体系。
- 可能包含非固定机位、非标准投篮场景，需要筛选。

### 2.2 Roboflow：Basketball Hoop, Ball and Player

链接：

https://universe.roboflow.com/personal-project-effcw/basketball-hoop-ball-and-player-5axdt

用途：

- 检测 ball、hoop、player。
- 训练“人 + 球 + 框”的初始检测模型。

公开信息：

- 约 1,312 张图片。
- 类别包括 ball、player、number、3pt_area、court、hoop、objects、paint。
- 页面显示 mAP@50 约 87.9%，Precision 约 89.6%，Recall 约 80.8%。
- 许可证：CC BY 4.0。

项目用途优先级：

> 高。适合补充 ball/hoop/player 检测。

注意：

- CC BY 4.0 需要署名。
- 类别较杂，训练前要清洗。

### 2.3 Roboflow：Basketball Video Analysis

链接：

https://universe.roboflow.com/computer-vision-project-v2zmg/basketball-video-analysis/model/1

用途：

- 参考篮球视频分析中的检测类别和效果。
- 可作为视频帧检测模型参考。

公开信息：

- 页面显示模型训练于 basketball-video-analysis。
- 约 3,152 张训练图片。
- mAP@50 约 82.9%，Precision 约 87.3%，Recall 约 77.3%。
- Roboflow 页面支持 API/边缘部署。

项目用途优先级：

> 中高。可用于参考，不建议完全依赖 hosted API。

注意：

- 需要确认具体类别和导出权限。
- 项目正式训练仍应导出数据或自训模型。

### 2.4 Roboflow Sports

链接：

https://github.com/roboflow/sports

用途：

- 参考 Roboflow 官方体育视觉数据组织方式。
- 其中有 basketball court keypoint detection、basketball jersey numbers OCR 等数据入口。

项目用途优先级：

> 中。不是投篮动作核心数据，但对后续球场/空间定位有价值。

### 2.5 Kaggle：Biomechanical Basketball Shooting Dataset

链接：

https://www.kaggle.com/datasets/ziya07/biomechanical-basketball-shooting-dataset

用途：

- 参考投篮生物力学指标。
- 可用于训练/测试 tabular 指标和建议生成。

公开搜索信息：

- 包含 shot speed、release angle、follow-through angle、distance 等特征。

项目用途优先级：

> 中。适合作为指标参考，不适合直接训练视觉模型。

注意：

- 需要登录 Kaggle 确认数据字段、质量和许可证。

### 2.6 Kaggle：Basketball Object Tracking Dataset

链接：

https://www.kaggle.com/datasets/trainingdatapro/basketball-tracking-dataset

用途：

- 训练/验证篮球 tracking。
- 球框检测和跟踪算法调试。

公开搜索信息：

- 来自篮球比赛视频截图。
- 标注了篮球 bounding box。

项目用途优先级：

> 中高。适合球追踪模块，但不一定是投篮固定机位。

注意：

- 需要确认许可证。
- 比赛视角和个人投篮训练视角差异较大。

### 2.7 Kaggle：NBA Player Shooting Motions

链接：

https://www.kaggle.com/datasets/paultimothymooney/nba-player-shooting-motions

用途：

- 球星风格参考。
- 3D shooting motion 数据参考。
- 后续做“风格匹配”或专业动作模板时可参考。

公开搜索信息：

- 包含球员典型投篮动作的三维轨迹数据。

项目用途优先级：

> 中。适合后期球星风格模块，不适合第一版 MVP。

### 2.8 Kaggle：Projectile Motion in Basketball

链接：

https://www.kaggle.com/datasets/energyinvestigators/projectile-motion-in-basketball

用途：

- 参考篮球抛物线、release angle、release velocity。
- 训练/验证球轨迹指标解释。

项目用途优先级：

> 中。偏物理模拟/表格，不是视觉核心数据。

### 2.9 Basketball-51 Activity Recognition Dataset

论文链接：

https://aircconline.com/csit/papers/vol11/csit110712.pdf

用途：

- 篮球动作识别。
- 命中/未命中、远中近距离投篮类型参考。

公开信息：

- 论文描述该数据集包含篮球比赛中的得分尝试片段。
- 分为 8 个类别。
- 每个 clip 包含远距离/中距离/近距离和 made/miss 信息。

项目用途优先级：

> 中。更适合动作分类，不是个人投篮姿势细分。

注意：

- 需要进一步确认数据是否可直接下载。

### 2.10 Stanford/课程项目参考：Basketball Shooting Analysis via 3D Pose Estimation

链接：

https://web.stanford.edu/class/cs231a/prev_projects_2022/lCS231a_Project_Final_Report.pdf

用途：

- 参考系统设计。
- 参考 teacher-student 动作对齐。
- 参考 3D pose analysis 思路。

公开信息：

- 项目目标是输入 student 和 teacher 投篮视频，重建 3D pose，计算前臂/上臂等角度，做动作对齐和对比。

项目用途优先级：

> 高。不是数据集，但对项目方法很有参考价值。

### 2.11 GitHub：chonyy/AI-basketball-analysis

链接：

https://github.com/chonyy/AI-basketball-analysis

用途：

- 参考早期篮球投篮分析系统架构。
- 参考 OpenPose + YOLO 类路线。

公开信息：

- 项目用于分析 basketball shots 和 shooting pose。
- 使用 OpenPose 计算 body keypoints。
- 项目 README 提醒 OpenPose 许可仅适合非商业研究。

项目用途优先级：

> 中。可以参考思路，不建议直接沿用 OpenPose 作为商业主线。

## 3. 本项目标准标签体系

为了避免不同公开数据集类别混乱，本项目统一标签如下。

### 3.1 目标检测标签

基础：

- `ball`
- `rim`
- `backboard`
- `net`
- `shooter`
- `other_player`

可选：

- `court_line`
- `free_throw_line`
- `three_point_line`
- `paint_area`

### 3.2 投篮事件标签

每次投篮必须逐步标注：

- `setup`：准备姿势。
- `dip`：下沉。
- `jump_start`：起跳开始。
- `release`：球离手。
- `follow_through`：随球。
- `landing`：落地。
- `rim_approach`：球接近篮筐。
- `result`：结果帧。

### 3.3 结果标签

- `made`
- `miss_front_rim`
- `miss_back_rim`
- `miss_left`
- `miss_right`
- `airball`
- `unknown`

### 3.4 拍摄标签

- `side`
- `front`
- `back`
- `forty_five_degree`
- `unknown`

### 3.5 质量标签

- `full_body_visible`
- `ball_visible_at_release`
- `rim_visible`
- `stable_camera`
- `good_lighting`
- `high_fps`
- `occluded`

## 4. 自建数据集路线

### 4.1 V0 数据集

目标：

- 支撑第一版人体姿态 + 关键帧报告。

数量：

- 20 到 50 段视频。

要求：

- 单人投篮。
- 固定机位。
- 侧面或 45 度。
- 全身入镜。
- 3 到 8 秒。
- 尽量 60fps。

标注：

- 拍摄角度。
- 是否命中。
- setup/dip/release/follow_through/landing 帧号。
- 视频质量标签。

### 4.2 V1 数据集

目标：

- 支撑篮球/篮筐检测和基础事件识别。

数量：

- 200 到 500 段视频。
- 从视频抽帧 5,000 到 20,000 张图片。

标注：

- ball bbox。
- rim bbox。
- net bbox。
- shooter bbox。
- release 帧。
- made/miss。

### 4.3 V2 数据集

目标：

- 支撑多次投篮稳定性分析。

数量：

- 50 到 100 名投篮者。
- 每人 20 到 50 次投篮。

标注：

- 每次投篮结果。
- 出手帧。
- 是否明显偏左/偏右/短/长。
- 拍摄角度。
- 投篮距离。

### 4.4 V3 数据集

目标：

- 支撑手型分析。

要求：

- 高清。
- 120fps 或 240fps。
- 近距离或 45 度前侧。
- 手和球尽量清楚。

标注：

- 手腕关键点。
- 辅助手离开帧。
- release 帧。
- follow-through 手型。

### 4.5 V4 数据集

目标：

- 支撑双机位/3D 专业版。

要求：

- 侧面 + 正面/45 度双机位。
- 同步拍摄。
- 相机标定。
- 篮筐完整入镜。

标注：

- 双机位同一投篮 ID。
- release。
- ball 2D bbox。
- rim 2D bbox。
- made/miss。
- 相机参数。

## 5. 标注工具建议

推荐：

- CVAT：视频标注、bbox、关键帧、轨迹。
- Label Studio：事件标签和文本元数据。
- Roboflow：目标检测数据管理和 YOLO 导出。

第一阶段最简单：

- 用 CVAT 标视频关键帧和 ball/rim bbox。
- 用 CSV/JSON 记录投篮事件和结果。

## 6. 本地数据目录规范

建议项目使用以下目录：

```text
datasets/
  README.md
  basketball-shot-dataset-plan.md
  manifest.json
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
    train.txt
    val.txt
    test.txt
```

## 7. manifest 字段规范

每段视频记录为：

```json
{
  "video_id": "self_000001",
  "source": "self_collected",
  "license": "private_research",
  "path": "raw/self_collected/self_000001.mp4",
  "fps": 60,
  "resolution": [1920, 1080],
  "camera_view": "forty_five_degree",
  "shot_distance": "free_throw",
  "made": true,
  "quality": {
    "full_body_visible": true,
    "ball_visible_at_release": true,
    "rim_visible": true,
    "stable_camera": true,
    "good_lighting": true
  },
  "events": {
    "setup": 15,
    "dip": 31,
    "jump_start": 42,
    "release": 55,
    "follow_through": 63,
    "landing": 78,
    "rim_approach": 96,
    "result": 104
  }
}
```

## 8. 第一批数据建设任务

### 8.1 公开数据

先做：

1. 注册 Roboflow。
2. Fork 或下载 Basketball and Hoop Detection。
3. Fork 或下载 Basketball Hoop, Ball and Player。
4. 统一标签为 ball/rim/net/shooter。
5. 训练一个 YOLO11 baseline。

### 8.2 自建数据

立刻做：

1. 自己或找同学拍 20 段固定机位投篮视频。
2. 每段视频记录是否命中。
3. 手动标 release、follow-through、landing。
4. 放入 `datasets/raw/self_collected/`。
5. 建立 `manifest.json`。

### 8.3 不建议现在做

暂时不要：

- 大规模爬 NBA/抖音/B站视频。
- 用未授权视频训练商业模型。
- 一开始标手指级关键点。
- 一开始做 3D 双机位数据。

## 9. 当前推荐数据集组合

第一阶段组合：

1. Roboflow Basketball and Hoop Detection：球/篮网检测。
2. Roboflow Basketball Hoop, Ball and Player：球/框/人检测。
3. 自建 20 到 50 段固定机位投篮视频：项目核心动作数据。
4. Stanford 3D Pose 项目报告：方法参考。
5. Kaggle Biomechanical Basketball Shooting Dataset：指标参考。

第二阶段组合：

1. Kaggle Basketball Object Tracking Dataset：球追踪补充。
2. Basketball-51：命中/未命中和投篮类型参考。
3. 自建 200 到 500 段投篮视频。

第三阶段组合：

1. NBA Player Shooting Motions：球星风格参考。
2. 自建高帧率手型数据。
3. 自建双机位 3D 数据。

## 10. 关键结论

本项目的数据路线不是“找一个现成完美数据集”，而是：

> 公开数据训练 ball/rim/shooter 检测，自建数据训练投篮事件、手型、轨迹和稳定性分析。

最应该马上做的是：

1. 用 Roboflow 数据训练球/篮筐检测 baseline。
2. 自己拍 20 到 50 段标准投篮视频。
3. 建立 manifest。
4. 手动标注关键事件。
5. 让系统先能基于这些视频输出报告。

## 11. 当前本地下载状态

已通过 Roboflow API 下载两个公开数据集到本地：

- `datasets/raw/public/roboflow/basketball-and-hoop-detection`
- `datasets/raw/public/roboflow/basketball-hoop-ball-and-player`

当前本地总量：

- 图片：10,435
- YOLO 标注 txt：10,439

注意：

- 两个数据集都是 YOLOv8 格式。
- 类别体系不一致，训练前必须统一标签。
- 详细状态见 `datasets/DATASET_STATUS.md`。
