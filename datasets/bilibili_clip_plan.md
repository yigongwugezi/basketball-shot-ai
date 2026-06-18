# B站投篮素材截取表说明

表格文件：

`datasets/bilibili_clip_plan.csv`

## 你需要填写的列

下载完视频后，主要填这三列：

- `downloaded_file_path`：下载后的视频本地路径
- `start_time`：要截取的开始时间，比如 `00:00:12.500`
- `end_time`：要截取的结束时间，比如 `00:00:18.000`

可选补充：

- `view_angle`：侧面、正面、背面、45度等
- `full_body_visible`：yes/no
- `ball_visible`：yes/no
- `rim_visible`：yes/no
- `camera_stable`：yes/no
- `notes`：你看到的问题或备注

## 推荐先看的顺序

1. `BILI_001`：库里全身侧面/背面固定镜头
2. `BILI_002`：普通人转腕发力解析
3. `BILI_003`：库里罚球投篮手右侧方视角
4. `BILI_005`：库里完整罚球慢动作
5. `BILI_006`：利拉德双视角、手部和重心细节

## 下载位置建议

用 DownKyi 下载后，建议统一放到：

`datasets/raw/public/video/public_candidates/bilibili`

如果 DownKyi 默认下到别的目录，也没关系，把完整路径填到 `downloaded_file_path` 即可。

## 截取后保存位置

后续我会把截好的 3-8 秒片段放到：

`datasets/processed/video_clips/public_candidates/bilibili`

