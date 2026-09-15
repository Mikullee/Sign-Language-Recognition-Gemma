# Live rest-hold follow-up

User approved direct implementation after the scalar-log diagnosis. Work stays
in the running repair branch, preserving its existing uncommitted changes.

Goal: stop small motion spikes resetting calibration, allow a return already
under confirmation to finish past the normal deadline, and clear stale warnings.
No model, archived v13 source, video upload, or display-rate change.

Design: a bounded observed-evidence hold shared by strict Web calibration,
rearming and end confirmation. Stable time only accrues between good real
observations. Visible knee/wrist/torso geometry is mandatory for tolerating
motion up to 1.5 times the normal threshold for at most 0.20 seconds. At least
80% of elapsed hold time must be stable. Invalid geometry, missing tracking,
larger movement, body relocation and observation gaps still reset the hold.
Only good observations update reference signatures or finish confirmation.
Hands still on knees cannot start a new sentence just because motion spikes.

Deadline: a hold that began before the five-second deadline may finish within
end_hold_sec + observation_gap_sec extra time. No pending confirmation means
immediate timeout; no fresh confirmation may begin after the deadline.
The gesture boundary must remain before the ordinary maximum duration.

Tasks (inline execution; already approved, no new design approval needed):

- [x] Add failing scalar-state tests in tests/test_web_rest_hold.py for isolated
  motion spikes, persistent movement, lost wrists/chest, rearm refresh count,
  deadline grace, lost evidence during grace, and no false start on knees.
- [x] Add failing DOM tests in tests/web_diagnostics_flicker.test.mjs for clearing
  an old segment warning on a new baseline or result, not on arbitrary frames.
- [x] Implement recognition/transformer/rest_hold.py and integrate only strict
  Web mode in live_trigger.py. Preserve non-knee legacy behavior.
- [x] Run Python and Node targeted tests, then all related regression tests.
- [x] Compare old scalar log hold evidence locally (not full video replay or
  accuracy), restart the exact local server, and run HTTP synthetic smoke.
- [x] Record tested changes and remaining live validation; keep videos private.

Verification commands: `python -m pytest tests/test_web_rest_hold.py -q`,
`npm test`, then the full related suite documented in the existing repair report.
Final delivery is the running local repair, not an unrequested GitHub push.
