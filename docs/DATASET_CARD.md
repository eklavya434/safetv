# SafeTV Dataset Card

## Dataset Summary

The SafeTV benchmark and pilot training dataset consists of curated video recordings and extracted spatiotemporal clips designed for surveillance violence detection research.

## Dataset Structure

- **Clips Directory**: `data/real_dataset/`
- **Annotations File**: `data/real_dataset/annotations/labels.csv`
- **Format**: MP4 video files, 15–30 FPS, various resolutions resized to 112×112 during extraction.

## Class Distribution

| Class Label | Class ID | Description |
|---|---|---|
| `non_violence` | 0 | Benign human activities: walking, running, hugging, exercising, dancing, celebrating |
| `violence` | 1 | Physical combat: punching, kicking, brawling, pushing, wrestling, physical assaults |

## Split Strategy & Data Leakage Prevention

- **Train Split**: 18 clips (70%)
- **Validation Split**: 3 clips (11%)
- **Test Split**: 5 clips (19%)

### Leakage Prevention Mechanism
Clips are grouped by unique base video source (`source_id`). All clips originating from the same recording are placed strictly into the same split (verified by `test_zero_data_leakage_between_splits`). No overlapping frames exist between train, validation, and test partitions.

## Annotations Schema

Each entry in `labels.csv` contains:
```csv
clip_id,source_video_path,processed_clip_path,split,label,label_id,subcategory,source_dataset,start_frame,end_frame,start_sec,end_sec,num_frames
```

## Known Dataset Limitations

1. **Dataset Size**: The current pilot set contains 26 clips. This is suitable for pipeline validation, integration testing, and proof of concept, but insufficient for enterprise deployment.
2. **Environmental Diversity**: The majority of clips represent indoor and daylight conditions; nighttime IR surveillance is not yet comprehensively represented.
3. **Demographic Coverage**: Demographic attributes (age, gender, ethnicity) have not been balanced.
