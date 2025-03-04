"""Reader Module"""

import queue
import logging
from threading import Thread
from math import inf

import cv2

from analyzer import Analyser
from fpsmeter import FPSMeter
from functions import beautifultime
from static import INFOTIME

logger = logging.getLogger(__name__)

class Reader:
    """A threaded worker that read the input flux"""

    def __init__(self, url, analys: Analyser, fifo: queue.Queue):
        self.url = url  # default is 0 for primary camera
        self.analys = analys
        self.fifo = fifo
        self.fps = FPSMeter()
        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.url)
        if self.vcap.isOpened() is False:
            logger.error("[Exiting]: Error accessing webcam stream.")
            exit(0)

        self.framerate = int(self.vcap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            logger.error("[Exiting] No more frames to read")
            exit(0)
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        #                       while the program is executing
        self.i = 0

    def start(self):
        """Start the threaded worker"""
        self.stopped = False
        logger.info("Start")
        self.t.start()

    def update(self):
        """The Thread main process"""
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed:
                self.i += 1
                self.fps.update()
                logger.debug(f"Img : {self.frame.shape}")
                self.fifo.put(
                    (self.i, self.analys.recolor(self.analys.crop_scale_inferance(self.frame))), True
                )
            if self.grabbed is False:
                self.fifo.put((inf, []), True)
                logger.info("[Exiting] No more frames to read")
                self.stopped = True
                break
            if self.i % (INFOTIME * self.framerate) == 0:
                logger.info(
                    f"{self.i} images ({beautifultime(self.i//self.framerate)}) (FPS:{self.fps.fps:.2f})"
                )

        self.vcap.release()

    def read(self):
        """get the last read image"""
        return self.frame

    def stop(self):
        """Stop the worker"""
        logger.info("Stop")
        self.stopped = True

    def join(self):
        """Wait untile the worket finish"""
        self.t.join()
