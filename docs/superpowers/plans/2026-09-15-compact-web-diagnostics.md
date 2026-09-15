# Compact Web diagnostics — approved scope and release checklist

User approved the proposed UI: tracking, rest speed/threshold and hold progress
remain primary; one operation hint stays on the video; duration appears only
while signing/confirming; FPS, rest distance and reference revision are collapsed.
Package and publish the update; preserve the accepted recognition behavior.

- [x] Add failing Node regressions before editing the served HTML/JS.
- [x] Implement display-only changes in webservice/static/index.html.
- [x] Preserve the API and backend, models, feature contract and thresholds.
- [x] Use native details/progress; end-confirmation progress is indeterminate
  because the current API does not expose that hold's elapsed evidence.
- [x] Reset old session display on mode change without changing session logic.
- [x] Verify 24 Node tests and desktop/narrow UI, disclosure and mode switching.
- [x] Independently reproduce and fix the reviewer-found Space shortcut conflict
  on SUMMARY; preserve native disclosure without starting manual recording.
- [x] Model progress.value attribute reflection in the test DOM double.
- [ ] Recheck full Python suite, independent review, archive and published digest.
- [ ] Commit/tag web-final-20260915-ui1, update handoff and existing PR, publish
  a new ZIP/SHA-256; retain web-final-20260915 as rollback baseline.

UI verification does not claim new human/model accuracy. Browser used for visual
checks denied camera permission; no permission bypass, images or raw landmark
recording is included in the public package. Whole-package installation remains
required from older Web builds; only the exact accepted base can take HTML-only.
