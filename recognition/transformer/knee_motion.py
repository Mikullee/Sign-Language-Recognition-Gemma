"""Causal rest-motion measurements; no filtering of recognition features."""
from collections import deque
import numpy as np
from recognition.transformer.knee_geometry import TORSO, _anchor, _normalized


class KneeRestMotion:
    def __init__(self, window_sec=.16):
        self.window_sec = window_sec
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, timestamp, vector, config):
        points = vector.reshape(75,3).copy()
        # One physical wrist trajectory persists across detector fallback.
        # Otherwise alternating missing hand/pose channels can leave every
        # group empty and incorrectly turn observed motion into zero.
        for pose,hand in ((15,33),(16,54)):
            if np.any(points[hand]):
                points[pose] = points[hand]
        anchor = _anchor(points)
        if (anchor is None or (self.samples and
                timestamp-self.samples[-1][0] > config.observation_gap_sec)):
            self.reset()
        if anchor is None or not np.any(points[[15,16]] != 0,axis=1).all():
            self.reset()
            return 0.0, False
        self.samples.append((timestamp, points.copy(), anchor))
        cutoff=timestamp-self.window_sec
        while len(self.samples)>2 and self.samples[0][0] < cutoff-1e-9:
            self.samples.popleft()
        elapsed=timestamp-self.samples[0][0]
        if len(self.samples)<2 or elapsed < .05-1e-9:
            return 0.0, False
        normalized=np.stack([_normalized(p,a) for _,p,a in self.samples])
        # Measure only actual observations, but do not erase a moving hand
        # just because one intervening detection was missing.
        observed=np.stack([np.any(p != 0,axis=1) for _,p,_ in self.samples])
        valid=observed.sum(axis=0)>=2
        times=np.asarray([t for t,_,_ in self.samples])[:,None]
        spans=np.where(observed,times,-np.inf).max(axis=0)-np.where(observed,times,np.inf).min(axis=0)
        # Excursion catches movement out-and-back even when endpoints match.
        maximum=np.where(observed[:,:,None],normalized,-np.inf).max(axis=0)
        minimum=np.where(observed[:,:,None],normalized,np.inf).min(axis=0)
        excursion=np.zeros((75,2))
        excursion[valid]=(maximum[valid]-minimum[valid])/spans[valid,None]
        def rms(indices):
            indices=np.asarray(indices)
            selected=indices[valid[indices]]
            return float(np.sqrt(np.mean(excursion[selected]**2))) if len(selected) else 0.0
        hand_speed=max(rms(range(33,54)),rms(range(54,75)),rms([13,14,15,16]))
        anchors=np.asarray([a for _,_,a in self.samples])
        body_speed=float(np.linalg.norm(np.ptp(anchors[:,:2],axis=0)))/anchors[0,-1]/elapsed
        torso_speed=max(rms(TORSO),body_speed)
        return max(hand_speed,config.torso_motion_weight*torso_speed), True
