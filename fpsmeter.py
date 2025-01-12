"""FpsMeter Module"""

import time


class FPSMeter:
    """A class to make a fps counter"""

    def __init__(self) -> None:
        self.time = time.perf_counter()
        self.fps = 0

    def update(self, img=1):
        """Update the fps counter"""
        t = time.perf_counter()
        self.fps = img / (t - self.time)
        self.time = time.perf_counter()
