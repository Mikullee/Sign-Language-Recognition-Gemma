# Knee-zone hysteresis and longer signs

User requests implementing the previously discussed stable rest-zone approach,
including timeouts while signing hunger / not understanding. Continue the local
repair branch; do not upload videos or capture new raw landmarks without consent.

Evidence: current human scalar stream 2 had 12 seed resets, 11 with geometry
false. A failed end confirmation had motion .113 but geometry false. The
shipped hard duration is 5 seconds. No recorded sentence labels establish which
event was either reported sign; do not claim sentence-specific measured success.

Implementation (TDD, inline):

- [x] Preserve raw geometric truth and expose a signed knee-region margin in
  shoulder widths. Zero reproduces the current inside/outside decision.
- [x] Add a separate stateful geometry tracker: strict valid geometry establishes
  a candidate; an exit band of at most .08 shoulder widths retains it only
  while both wrists remain within .10 shoulder widths of its anchor. No candidate
  can originate outside the strict region. Use body-relative wrist positions;
  body relocation, observation gaps, invalid torso/wrists/knees reset the tracker.
  This is spatial hysteresis, not a time-only permission to ignore lost tracking.
- [x] Keep recognition features and motion guards unchanged. Reset the rest
  motion filter on stable-region entry, not every raw geometry boundary flip.
  Update the zone anchor only after a reference revision or a new strict entry.
- [x] Set the shipped Web maximum to 10 seconds for all classes; retain bounded
  end-confirmation grace, timeout rejection, feature-buffer bounds and no EOF
  fabrication. Add current elapsed/max duration to diagnostics and UI messages.
- [x] Test boundary jitter at 10/15/17.5/30 FPS, initial outside/chest/abdomen,
  real departure, lost tracking, relocation, reset, two sentences, and 6/8-second
  generic gestures. Longer-sequence model input stays resampled to its fixed
  shape; tests demonstrate transport compatibility, not accuracy.
- [x] Run full Python/Node suite, independent review and actual-model HTTP
  normal/long-sign checks; restart local service and document limitations.

Alternatives rejected: merely increasing timeout leaves geometry resets intact;
ignoring knee tracking indefinitely or accepting timed-out segments removes the
user's end-of-sentence guarantee. Exact tolerances are candidate engineering
bounds, not empirically validated universal thresholds.

Verification: 426 Python tests + 4 subtests; 16 Node tests. Independent read-only
review found no important defects in this scoped change. Actual loaded model via
HTTP produced one result and reference revision 2 for each normal, 8-second and
late-return synthetic case. Archived v13 module hashes unchanged. Human signing
of the two named phrases is pending, not established by these synthetic checks.
