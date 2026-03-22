# Team Tasks

## Group 1: Recording

Main job:

- record all assigned sequence videos
- follow naming rules exactly
- follow the sequence template exactly
- re-record videos if review group marks them as invalid

Files to use:

- `data/annotations/sequence_templates_12.csv`
- `data/annotations/sequence_recording_manifest_300.csv`

What to update:

- update recording progress in `sequence_recording_manifest_300.csv`

---

## Group 2: Review + File Organization

Main job:

- check whether the video is usable
- verify framing, lighting, hand visibility, and gloss order
- mark whether re-recording is needed
- keep filenames and folders consistent

Files to use:

- `data/annotations/sequence_recording_manifest_300.csv`
- `data/annotations/sequence_review_checklist_300.csv`

What to update:

- `review_status`
- `need_rerecord`
- `notes`

---

## Group 3: Annotation

Main job:

- annotate gloss order for each sequence
- fill `start_sec` and `end_sec`
- keep labels consistent with glossary

Files to use:

- `data/annotations/sequence_annotations_300_template.csv`
- `data/annotations/glossary_template.csv`

What to update:

- `start_sec`
- `end_sec`
- `done`
- `notes`

Annotation rule:

- one row = one gloss segment
- fill time in seconds
- recommended precision: `0.001`

---

## Group 4: Gloss Description + Documents

Main job:

- define each gloss clearly
- maintain the glossary
- write motion descriptions for each gloss
- prepare materials that can later support `HST-SLR`

Files to use:

- `data/annotations/glossary_template.csv`
- `data/annotations/gloss_description_assignment.csv`

What to update:

- `meaning`
- `alias`
- `description`
- `notes`

Important:

- descriptions are written per gloss class, not per video
- `其他` must stay restricted to the meaning of `其他地區`
