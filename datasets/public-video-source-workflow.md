# 公开视频素材池工作流

更新日期：2026-06-18

## 目标

建立一个“可人工筛选、可逐步授权、可截取标注”的投篮视频候选池。

这里的核心不是马上批量下载，而是先把素材源组织起来：

- 哪些视频像固定机位
- 哪些能看到全身
- 哪些能看到篮球和篮筐
- 哪些适合截 3-8 秒片段
- 哪些可能有版权风险
- 哪些适合作为专业报告语言参考

候选清单在：

`datasets/video_sources.csv`

## 为什么不用直接批量下载

公开视频平台上的投篮教学、球星慢动作、自媒体训练视频确实很多，但大多数不是开放数据集。

因此项目里先做三层处理：

1. `候选 URL`：只保存链接、标题、平台和用途判断。
2. `人工验收`：看画面是否清晰、全身、固定机位、动作完整。
3. `授权/自采/用户上传`：只有明确可用的视频才进入正式训练集。

## 第一批筛选标准

高优先级视频应满足：

- 单人或主体清楚
- 全身入镜
- 篮球可见
- 篮筐最好可见
- 侧面、背侧、45 度优先
- 镜头不要频繁移动
- 画质 720p 以上优先
- 单次投篮动作完整
- 能截出 3-8 秒片段

## 素材类型

### 1. 教练教学视频

用途：

- 获取标准动作示范
- 获取常见错误和纠正语言
- 生成报告模板

风险：

- 多为剪辑视频，不一定每段都是完整投篮
- 授权不一定允许训练

### 2. 球星慢动作视频

用途：

- 风格匹配
- release、follow-through、手型参考
- 专业动作模板

风险：

- 常来自比赛或二创
- 不一定固定机位
- 可能只适合研究和参考，不适合正式训练

### 3. 普通人/训练营固定机位视频

用途：

- 最接近产品真实用户
- 适合做姿势纠正数据

风险：

- 来源零散
- 授权难确认
- 需要大量人工筛选

### 4. 官方/机构教学视频

用途：

- 作为可靠参考
- 用于指标体系和教学语言

风险：

- 未必允许下载或训练

## 下一步

1. 人工打开 `datasets/video_sources.csv` 里的高优先级链接。
2. 给每条视频补充视觉验收字段：
   - `full_body_visible`
   - `ball_visible`
   - `rim_visible`
   - `camera_stable`
   - `usable_clip_start`
   - `usable_clip_end`
3. 先选 10 条做手动截片段测试。
4. 每条只截 1-3 个 3-8 秒投篮片段。
5. 标注 `setup/dip/release/follow_through/landing/result`。

## 本地保存位置

候选原视频保存到：

`datasets/raw/public/video/public_candidates`

通过人工验收后截出的 3-8 秒短片段保存到：

`datasets/processed/video_clips/public_candidates`

下载少量高优先级候选视频：

```powershell
.venv310\Scripts\python.exe scripts\download_public_video_candidates.py --priority high --limit 3
```

先预览将会下载什么，不实际下载：

```powershell
.venv310\Scripts\python.exe scripts\download_public_video_candidates.py --priority high --limit 3 --dry-run
```

## 重要判断

公开视频素材可以作为项目冷启动，但不能替代自有数据。

本项目后续真正有价值的数据应来自：

- 自己拍摄的投篮视频
- 用户上传并授权的视频
- 和训练营/教练合作获得的视频
- 明确开放授权的数据集
