# release_ball_batch_001

- Annotation date: 2026-06-18
- Annotation scope: BILI_001_A, BILI_003_A, BILI_005_A
- Each clip covers about 15 frames before and after release, 31 frames total per clip
- Total frames: 93
- 71 frames with target shooter ball box
- 22 frames without target shooter ball

## Labeling Rules

- Only label the main shooter's basketball
- Do not label balls held by background people
- Do not label irrelevant balls
- If the target ball is partially visible but still identifiable, a box may be used with `quality=poor` or `occlusion=partial/heavy`
- If the target ball cannot be seen or confirmed, set `ball_visible=no`

## Release Ground Truth

- BILI_001_A `ball_release_frame=55`, confidence=high
- BILI_003_A `ball_release_frame=517`, confidence=medium
- BILI_005_A `ball_release_frame=224`, confidence=high

## Purpose

- Strict `ball_release_frame` verification
- Failure analysis for release-neighborhood ball detection
- Small-sample basis for later basketball detector fine-tuning

## Limitations

- No image files included
- No video files included
- Frame images can be regenerated from the source clips by `frame_index`
- This is only the first version of a small annotation set and should not be treated as a final training conclusion
