"""The treaded counter to maximise performances with FIFOs"""

import datetime
import queue
import time

from analyzer import Analyser
from computing import Computing
from inferance import Inferance, provider
from reader import Reader
from loger import Loger
from graph import Graph

import matplotlib.pyplot as plt


class ThreadCounter:
    """The treaded Detector the manages alls workers"""

    def __init__(
        self, configfile, url, logfile, size=30, net="yolov8n.onnx", show=False
    ) -> None:
        self.configfile = configfile
        self.logfile = logfile
        self.imgs = queue.Queue(60)
        self.preds = queue.Queue(120)
        self.people = queue.Queue(60)

        self.analys = Analyser(self.configfile)
        self.analys.open()
        self.cam = Reader(url, self.analys, self.imgs)
        videoinfos = self.cam.framerate, self.cam.frame_count
        self.infes = [Inferance(
            self.imgs, self.preds, self.analys, videoinfos, net, size=size,index = i+1,providers= [p]
        ) for i, p in enumerate(provider)]
        self.compute = Computing(self.preds, self.people, self.analys, videoinfos, show)
        self.loger = Loger(logfile, self.people)
        self.duration = datetime.timedelta(milliseconds=1)

        self.graph = Graph(self.imgs, self.preds, self.people)

        self.starttime = 0

        self.runing= False

    def start(self):
        """the starting function"""
        self.runing =True
        self.starttime = datetime.datetime.now()
        self.graph.add_rec()
        self.cam.start()
        for infe in self.infes :
            infe.start()
        self.compute.start()
        self.loger.start()

    def run(self):
        self.start()
        try:
            while self.runing and self.compute.t.is_alive():
                self.graph.add_rec()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Interuption")
            time.sleep(0.1)

        self.close()

        return self.get_result()
    
    def pourcent(self):
        return self.compute.i / self.cam.frame_count
    
    def stop(self):
        print("Stop")
        self.runing = False
        self.close()

    def get_result(self):
        return self.compute.counter, self.duration
    

    def close(self):
        print("Close")
        self.runing = False
        self.cam.stop()
        for infe in self.infes :
            infe.stop()
        self.compute.stop()
        self.loger.stop()

        self.loger.rec(
            self.cam.frame_count / (self.cam.framerate), self.compute.counter
        )
        self.loger.close()
        self.duration = datetime.datetime.now() - self.starttime

    def factorspeed(self):
        """To calculatio the speed of calculus"""
        return float(
            datetime.timedelta(seconds=self.cam.frame_count / self.cam.framerate)
            / self.duration
        )

    def show_graph(self):
        fig = plt.figure('Size of Queues')
        self.graph.make_graph(fig)
        plt.show()