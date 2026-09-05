# RELEASE_BALL_V2_TRAINING_MANIFEST

Frozen before training; do not use it to tune against R-CAM2.

## Configuration

- Architecture: YOLO11n
- Image size: 640
- Seed: 0
- Evaluation confidence threshold: 0.15
- Training: 100 epochs, batch 16, optimizer auto, pretrained initialization

## Dataset and split

- Manual reviewed rows: 341 (train 279, val 62)
- Inputs: release_ball_batch_001 and release_ball_batch_003 historical reviewed extracts
- Validation clips: BILI_003_A, NEW_012
- Training clips are source-disjoint from validation clips.

## R-CAM2 exclusion

- Excluded whole source videos: IMG_7221.MOV and IMG_7222.MP4.
- IMG_7221.MOV occurred as NEW_006 and all 31 of its labeled frames were removed.
- IMG_7222.MP4 had no rows in the formal label inputs.
- Auto-assisted batch002 frames are not training inputs.

- Dataset metadata SHA256: `6f902bd31d153ccaae5e887882f8fbce8221519a394acab0d3d82f0acaf2ec98`
