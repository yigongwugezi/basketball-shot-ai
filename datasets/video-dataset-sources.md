# 投篮视频数据集清单

更新日期：2026-06-17

## 结论

目前公开网上能找到篮球视频数据集，但真正适合“投篮姿势纠正”的公开视频很少。

本项目要分两条线走：

1. 公开视频数据集：用于视频上传测试、动作识别、投篮片段识别、球/框/人检测的早期验证。
2. 自采投篮视频：用于真正的姿势分析、出手时刻、手型、身体角度、球路和个性化建议。

也就是说：咱们现在确实需要投篮视频，但不能指望一个公开视频集直接喂进去就变成专业投篮教练。

## 当前优先级

### P0：Basketball-51 Dataset

链接：
https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset

用途：

- 视频上传流程测试
- 投篮/得分片段分类
- made/miss、1 分/2 分/3 分这类粗粒度判断参考
- 后续训练视频分类或事件识别 baseline

优点：

- 是真实篮球视频片段
- 与“投篮结果”和“得分类型”更接近
- 适合让当前系统从“只能上传视频”变成“有真实篮球视频可测”

限制：

- 多数来自比赛转播视角，不是固定机位的个人投篮训练视频
- 不适合直接做细粒度投篮姿势纠正
- 不一定能看清手型、脚步、完整身体链条

本地建议路径：

`datasets/raw/public/video/basketball-51`

当前本地状态：

- 已下载
- 本地视频数量：10,311 个 `.mp4`
- 占用空间：约 6.03 GB
- 实际目录：`datasets/raw/public/video/basketball-51/Basketball_51 dataset`
- 子目录：`2p0`、`2p1`、`3p0`、`3p1`、`ft0`、`ft1`、`mp0`、`mp1`

下载方式：

需要 Kaggle 登录，并配置新版 `KAGGLE_API_TOKEN` 或 `~/.kaggle/access_token`。配置后运行：

```powershell
.\scripts\download_kaggle_basketball51.ps1
```

旧版 Python API 脚本仍保留，用于兼容 `kaggle.json`：

```powershell
.venv310\Scripts\python.exe scripts\download_kaggle_basketball51.py
```

### P1：SpaceJam Basketball Action Recognition

链接：
https://github.com/simonefrancia/SpaceJam

Kaggle 镜像：
https://www.kaggle.com/datasets/antocommii/spacejam-action-recognition

用途：

- 篮球动作识别
- shooting / pass / dribble 等动作分类参考
- 视频 clip 和人体关节点数据结构参考

优点：

- 每个样本是短视频 clip
- 同时有 joint coordinates，适合研究“骨架 + 动作识别”的路线
- 可用于投篮动作识别的预研

限制：

- clip 较短，分辨率较低
- 重点是动作分类，不是投篮姿势纠错
- 不一定包含完整出手、球路、入框结果

本地建议路径：

`datasets/raw/public/video/spacejam`

### P1：UCF101 / Basketball 类公开视频

官方页：
https://www.crcv.ucf.edu/data/UCF101.php

Hugging Face ZIP 镜像：
https://huggingface.co/datasets/quchenyuan/UCF101-ZIP

用途：

- 从公开视频中整理出的动作识别数据
- 可作为“篮球动作视频素材池”
- 可筛选其中 Basketball / BasketballDunk 相关片段，用于动作识别、人体姿态抽取、视频质量测试

优点：

- 数据集本身就是从 YouTube 等用户上传视频整理而来
- 总量约 13,320 个视频，101 个动作类别
- Hugging Face 镜像约 6.96 GB，下载比原始站点更方便
- 比 Basketball-51 更接近“公开视频平台整理包”

限制：

- 主要任务是动作分类，不是投篮姿势纠错
- 画面分辨率多为 320x240，很多视频并不适合精细手型分析
- 类别粒度粗，需要二次筛选出“全身、近距离、固定机位、单人投篮”的片段

本地建议路径：

`datasets/raw/public/video/ucf101`

### P1：自媒体/教学视频采集候选池

来源：

- YouTube：篮球投篮教学、free throw form、shooting mechanics、side view shooting form
- B 站：投篮教学、投篮姿势、出手慢动作、库里投篮解析
- 抖音/快手：个人训练、青少年训练营、教练教学短视频

用途：

- 建立“候选视频清单”
- 人工筛选高质量片段
- 用于非公开研发测试、算法验证和标注流程设计

筛选标准：

- 单人或少人
- 全身入镜
- 篮球、篮筐尽量可见
- 侧面或 45 度
- 画面稳定
- 清晰度 720p 以上优先
- 出手前后动作完整

注意：

- 这些视频“网上很多”，但通常不是开放数据集。
- 不能默认用于商用训练或公开发布数据集。
- 更稳的做法是先做 URL 清单和人工标注流程，后续只把获得授权或用户上传授权的视频纳入正式训练集。

项目判断：

> 这是很有价值的一条路。公开视频平台适合做“候选池”和“冷启动数据来源”，但最终产品护城河仍应来自用户授权上传和自采标注数据。

### P1：BASKET Fine-Grained Skill Estimation

论文：
https://arxiv.org/abs/2503.20781

项目页：
https://sites.google.com/cs.unc.edu/basket

用途：

- 长视频篮球技能评估研究参考
- 后期“球员能力评估”“风格匹配”“综合技能评分”的方向参考

优点：

- 规模极大：论文称包含 4,477 小时视频、32,232 名篮球运动员
- 任务更接近技能评估，而不是普通动作分类
- 对本项目最终专业化方向有参考价值

限制：

- 数据规模和任务都偏研究级
- 不是第一阶段 MVP 的最佳数据源
- 不一定容易直接下载和清洗

本地建议路径：

`datasets/raw/public/video/basket`

### P2：Biomechanical Basketball Shooting Dataset

链接：
https://www.kaggle.com/datasets/ziya07/biomechanical-basketball-shooting-dataset

用途：

- 投篮生物力学指标参考
- release angle、shot speed、follow-through angle 等指标参考
- 给 AI 建议生成提供专业术语和指标结构

优点：

- 和投篮姿势指标直接相关
- 适合做报告指标体系参考

限制：

- 更偏表格/指标数据，不一定有原始视频
- 不能直接解决视频姿势检测

本地建议路径：

`datasets/raw/public/biomechanics`

### P2：Real-Time Basketball Shooting Action Recognition

论文：
https://www.diva-portal.org/smash/get/diva2%3A1890063/FULLTEXT01.pdf

用途：

- 方法参考
- 单摄像头视频 + 人体骨架 + 篮球节点的 ST-GCN 路线参考

优点：

- 方向非常贴近“投篮动作识别”
- 强调把篮球节点加入人体骨架图结构

限制：

- 主要是论文方法，不是直接可用的数据集
- 不等于姿势纠错数据

## 为什么之前下载了很多图片

因为视频分析不是只靠视频分类。

一个投篮视频系统通常要先把每一帧里的关键对象找出来：

- 人
- 篮球
- 篮筐
- 篮板
- 身体关键点

所以 Roboflow 图片数据集的价值是训练/验证 `ball/rim/player` 检测器。它们不是最终目的，但能支撑视频逐帧分析。

当前已完成的图片数据集主要服务于：

- 视频中检测篮球
- 视频中检测篮筐
- 视频中检测球员
- 抽关键帧后做可视化报告

## 项目真正需要的自采视频

公开视频只能帮我们起步。要做专业投篮教练，必须建立自己的投篮视频数据。

### V0 自采标准

每条视频：

- 单人投篮
- 手机横屏或竖屏都可以，但镜头固定
- 人全身可见
- 篮筐可见
- 出手前 1 秒到结果后 1 秒
- 尽量 60fps
- 侧面 45 度优先

每条视频至少标注：

- `setup`：准备
- `dip`：下沉
- `release`：出手
- `follow_through`：随球
- `landing`：落地
- `result`：命中/未命中

### V1 自采数量

第一批不要贪大，先采：

- 5 个人
- 每人 20 次投篮
- 总共 100 条视频

这 100 条视频比随便下载几千个比赛片段更有价值，因为它们场景一致、身体可见、能标注动作问题。

## 本项目视频数据目录约定

```text
datasets/
  raw/
    public/
      video/
        basketball-51/
        spacejam/
        basket/
    self_collected/
      videos/
      labels/
  processed/
    video_clips/
    shot_events/
```

## 近期行动

1. 先拿 Basketball-51 做视频上传和分析流程测试。
2. 当前原型支持上传视频后抽关键帧并跑 ball/rim/player 检测。
3. 配好 Kaggle token 后，运行 `scripts/download_kaggle_basketball51.py` 下载视频。
4. 没有 Kaggle token 时，可以手动从 Kaggle 下载 zip，解压到 `datasets/raw/public/video/basketball-51`。
5. 同步开始设计自采视频标注格式，因为这才是项目后续专业化的核心资产。
