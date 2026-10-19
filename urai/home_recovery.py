"""Restricted recovery from a near-margin stop; ordinary motion keeps 30 mm."""
import math

class HomeClearanceRecovery:
    margin_mm = 30.
    exit_mm = 32.
    minimum_start_mm = 28.

    def __init__(self, start_mm, *, feedback=False):
        start_mm = float(start_mm)
        if not math.isfinite(start_mm) or not self.minimum_start_mm <= start_mm < self.margin_mm:
            raise ValueError('Recovery requires a positive near-margin clearance (28–30 mm)')
        self.high_water = start_mm
        self.feedback = feedback
        self.exited = False

    def check(self, clearance_mm):
        c = float(clearance_mm)
        if not math.isfinite(c):
            raise ValueError('Non-finite inter-arm clearance')
        if self.exited:
            if c < self.margin_mm:
                raise ValueError(f'Recovery re-entered inter-arm margin: {c:.3f} mm')
            self.high_water = max(self.high_water, c)
            return
        # No planned decrease. Measured encoder/geometry jitter has a bounded
        # 0.3 mm allowance against the greatest clearance already achieved;
        # repeated samples cannot accumulate a backwards drift.
        tolerance = .3 if self.feedback else 1e-7
        floor = max(self.minimum_start_mm, self.high_water - tolerance)
        if c < floor:
            raise ValueError(f'Recovery would reduce inter-arm clearance: {c:.3f} < {floor:.3f} mm')
        self.high_water = max(self.high_water, c)
        if c >= (self.margin_mm if self.feedback else self.exit_mm):
            self.exited = True

    def finish(self):
        if not self.exited or self.high_water < self.exit_mm:
            raise ValueError('Recovery must leave the inter-arm margin by at least 2 mm')
