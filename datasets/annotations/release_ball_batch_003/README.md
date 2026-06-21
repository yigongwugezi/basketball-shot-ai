# release_ball_batch_003

- Annotation batch: `release_ball_batch_003`
- Source scope: 9 formal candidates selected from 13 new clips in `batch_003`
- Intended use: target-shooter basketball detector and strict release evaluation data
- Assets included here: CSV annotations only
- Assets excluded here: images, frames, contact sheets, source videos, tmp files

## Included clips

- NEW_001
- NEW_002
- NEW_003
- NEW_004
- NEW_005
- NEW_006
- NEW_009
- NEW_010
- NEW_012

## Excluded clips

- NEW_007: window error, captured a dribble segment instead of a release window
- NEW_008: layup, not part of current jump-shot release data
- NEW_011: window error, captured a dribble segment instead of a release window
- NEW_013: no valid release window, not reliable for strict `ball_release_frame`

## Annotation target

- Only annotate the main shooter's basketball
- Do not annotate background balls
- Do not annotate balls in other players' hands
- Do not annotate unrelated floor balls

## Frame definitions

- `ball_release_frame`: the first frame after the ball fully leaves the shooting fingertips
- `release_pose_frame`: the system pose-based release estimate; it is not the same as `ball_release_frame`

## Special notes

- `NEW_004`: follow-ball camera with possible blank/decoder-anomaly frames; suitable for detector/release auxiliary data, not for full-motion primary analysis
- `NEW_006`: sourced from a fallback window, then manually reviewed and completed
- `NEW_012`: `label_confidence=low`; later frames contain many `ball_visible=no`, but the clip is fully closed out

## Dataset notes

- This batch is suitable for detector and release training/evaluation workflows
- `NEW_004` should not be treated as a full-motion primary sample
- `image_file` entries are relative references only; no absolute local paths are included
