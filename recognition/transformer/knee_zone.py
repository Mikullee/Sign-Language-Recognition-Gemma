"""Spatial hysteresis for a previously observed knee region; no model edits."""
import numpy as np


class KneeRestZone:
    def __init__(self):
        self.reset()

    def reset(self):
        self.inside = False
        self.body = self.wrists = self.last_t = None
        self.revision = None

    def update(self, timestamp, analysis, config, revision):
        trusted = (analysis.torso_valid and analysis.wrists_detected and analysis.knee_landmarks_valid
                   and analysis.body_anchor is not None and analysis.wrist_rest_signature is not None)
        if not trusted:
            self.reset()
            return False
        body = np.asarray(analysis.body_anchor)
        wrists = np.asarray(analysis.wrist_rest_signature).reshape(2,2)
        relocated = False
        if self.body is not None:
            shift = max(np.linalg.norm(body[:2]-self.body[:2]),
                        np.linalg.norm(body[2:4]-self.body[2:4]))/self.body[-1]
            scale = max(body[-1]/self.body[-1],self.body[-1]/body[-1])
            relocated = shift > config.body_shift_threshold or scale > config.body_scale_ratio_threshold
        if relocated or (self.last_t is not None and timestamp-self.last_t > config.observation_gap_sec):
            self.reset()
        self.last_t = timestamp
        if analysis.hands_on_knees:
            if not self.inside or self.revision != revision:
                self.body, self.wrists = body.copy(), wrists.copy()
            self.inside = True
        else:
            # A near miss alone never creates a candidate. Both real wrists
            # must remain near the original region, not follow a moving hand.
            self.inside = bool(self.inside and analysis.knee_region_margin is not None
                and analysis.knee_region_margin >= -config.knee_zone_exit_margin
                and np.max(np.linalg.norm(wrists-self.wrists,axis=1)) <= config.knee_zone_wrist_radius)
        if self.inside and self.revision != revision:
            self.body, self.wrists = body.copy(), wrists.copy()
        self.revision = revision
        return self.inside
