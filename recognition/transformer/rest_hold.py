"""Bounded holds using real observations; uncertain time never counts as rest."""


class ObservedRestHold:
    def __init__(self, target, grace, *, minimum_ratio=.8):
        self.target = target
        self.grace = grace
        self.minimum_ratio = minimum_ratio
        self.clear()

    def clear(self):
        self.onset = self.last_t = self.last_good_t = None
        self.run_onset = None
        self.good_sec = 0.0
        self.previous_good = False

    def update(self, t, *, good, soft=False):
        if not good and not soft:
            self.clear()
            return False
        # A resumed good frame must not bridge a long uncertain interval.
        if self.onset is not None and not self.previous_good:
            if t - self.last_good_t > self.grace + 1e-9:
                self.clear()
        if self.onset is None:
            if not good:
                return False
            self.onset = t
        if good and not self.previous_good:
            self.run_onset = t
        if not good:
            self.run_onset = None
        if self.last_t is not None and good and self.previous_good:
            self.good_sec += max(0., t-self.last_t)
        self.last_t = t
        self.previous_good = good
        if good:
            self.last_good_t = t
        elapsed = t-self.onset
        ready = (good and self.good_sec + 1e-9 >= self.target
                 and self.good_sec + 1e-9 >= self.minimum_ratio * elapsed)
        # A clean continuous tail must always be at least as fast as the old
        # strict hold, regardless of uncertainty earlier in this window.
        run_sec = 0. if self.run_onset is None else t-self.run_onset
        if not ready and good and run_sec + 1e-9 >= self.target:
            self.onset, self.good_sec = self.run_onset, run_sec
            return True
        # Avoid collecting tiny good intervals indefinitely around real motion.
        if not ready and elapsed > self.target/self.minimum_ratio + self.grace:
            if good:
                self.onset, self.good_sec = self.run_onset, run_sec
            else:
                self.clear()
        return ready
