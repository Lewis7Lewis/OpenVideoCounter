"""The treaded counter to maximise performances with FIFOs"""

import datetime
import queue
import time

from analyzer import Analyser
from computing import Computing
from inferance import Inferance
from reader import Reader
from loger import Loger


class ThreadCounter:
    """The treaded Detector the manages alls workers"""

    def __init__(
        self, configfile, url, logfile, size=30, net="yolov8n.onnx", show=False
    ) -> None:
        self.configfile = configfile
        self.logfile = logfile
        self.imgs = queue.PriorityQueue(60)
        self.preds = queue.PriorityQueue(120)
        self.people = queue.Queue(60)

        self.analys = Analyser(self.configfile)
        self.analys.open()
        self.cam = Reader(url, self.analys, self.imgs)
        videoinfos = self.cam.framerate, self.cam.frame_count
        self.infes = [Inferance(
            self.imgs, self.preds, self.analys, videoinfos, net, size=size,index = i+1
        ) for i in range(2)]
        self.compute = Computing(self.preds, self.people, self.analys, videoinfos, show)
        self.loger = Loger(logfile, self.people)
        self.duration = datetime.timedelta(milliseconds=1)

    def run(self):
        """the starting function"""
        starttime = datetime.datetime.now()
        self.cam.start()
        for infe in self.infes :
            infe.start()
        self.compute.start()
        self.loger.start()
        try:
            while self.compute.t.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.cam.stop()
            for infe in self.infes :
                infe.stop()
            self.compute.stop()
            self.loger.stop()

        self.loger.rec(
            self.cam.frame_count / (self.cam.framerate), self.compute.counter
        )
        self.loger.close()

        self.duration = datetime.datetime.now() - starttime

        return self.compute.counter, self.duration

    def factorspeed(self):
        """To calculatio the speed of calculus"""
        return float(
            datetime.timedelta(seconds=self.cam.frame_count / self.cam.framerate)
            / self.duration
        )
