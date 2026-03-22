# Sign Language Recognition Project

## Project Goal

This project is building a continuous sign language recognition pipeline.

Current direction:

1. Build a usable baseline with `MediaPipe + BiGRU`
2. Expand sequence data to support continuous recognition
3. Prepare data and annotations so the project can later move toward `HST-SLR`

---

## Current Technical Stack

- Landmark extraction: `MediaPipe`
- Input features:
  - relative coordinates
  - geometry features
  - velocity features
- Baseline recognition model: `Bidirectional GRU`
- Future target direction: `HST-SLR`

---

## What This Repository Is For

This repository is mainly for team collaboration on:

- sequence video recording
- video review and file organization
- sequence annotation
- glossary and gloss description writing

---

## Team Workflow

Please read these files first:

1. [`data/annotations/sequence_work_plan_300.md`](data/annotations/sequence_work_plan_300.md)
2. [`docs/TEAM_TASKS.md`](docs/TEAM_TASKS.md)
3. [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md)

---

## Core Files

### Sequence planning
- [`data/annotations/sequence_templates_12.csv`](data/annotations/sequence_templates_12.csv)
- [`data/annotations/sequence_recording_manifest_300.csv`](data/annotations/sequence_recording_manifest_300.csv)

### Review
- [`data/annotations/sequence_review_checklist_300.csv`](data/annotations/sequence_review_checklist_300.csv)

### Annotation
- [`data/annotations/sequence_annotations_300_template.csv`](data/annotations/sequence_annotations_300_template.csv)

### Glossary / descriptions
- [`data/annotations/glossary_template.csv`](data/annotations/glossary_template.csv)
- [`data/annotations/gloss_description_assignment.csv`](data/annotations/gloss_description_assignment.csv)

---

## Recording Rules

- upper body must stay fully in frame
- hands must not leave the frame
- fixed camera position
- simple background
- stable lighting
- leave `0.5 ~ 1` second blank at the beginning and end
- follow the exact gloss order of the assigned sequence template

Special note:

- `其他` is only allowed in the meaning of `其他地區`
- do not use `其他` as a catch-all unknown label

---

## Directory Notes

- `data/annotations/`: all CSV templates and task files
- `data/results/`: generated outputs, reports, videos, model outputs
- `scripts/`: processing, training, visualization scripts
- `temp/`: local working data and raw videos

---

## Next Stage

After enough sequence data is collected and annotated:

1. train a stronger sequence model
2. evaluate segmentation quality
3. convert dataset into an `HST-SLR`-compatible format
4. test `HST-SLR` on the expanded sequence dataset
