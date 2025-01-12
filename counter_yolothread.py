"""The treaded counter to maximise performances with FIFOs"""

import datetime
import queue
import time
from tkinter import filedialog

from anayzer import Analyser
from computing import Computing
from inferance import Inferance
from reader import Reader
from loger import Loger

last_time = time.time()


class TDetector:
    """The treaded Detector the manages alls workers"""

    def __init__(
        self, configfile, url, logfile, size=30, net="yolov8n.onnx", show=False
    ) -> None:
        self.configfile = configfile
        self.logfile = logfile
        self.imgs = queue.Queue(60)
        self.preds = queue.Queue(60)
        self.people = queue.Queue(60)

        self.analys = Analyser(self.configfile)
        self.analys.open()
        self.cam = Reader(url, self.analys, self.imgs)
        videoinfos = self.cam.framerate, self.cam.frame_count
        self.infe = Inferance(
            self.imgs, self.preds, self.analys, videoinfos, net, size=size
        )
        self.compute = Computing(self.preds, self.people, self.analys, videoinfos, show)
        self.loger = Loger(logfile, self.people)

    def run(self):
        """the starting function"""
        self.cam.start()
        self.infe.start()
        self.compute.start()
        self.loger.start()
        try:
            while self.compute.t.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.cam.stop()
            self.infe.stop()
            self.compute.stop()
            self.loger.stop()

        self.loger.rec(
            self.cam.frame_count / (self.cam.framerate), self.compute.counter
        )
        self.loger.close()

        return self.compute.counter

    def factorspeed(self, timing):
        """To calculatio the speed of calculus"""
        return float(
            datetime.timedelta(seconds=self.cam.frame_count / self.cam.framerate)
            / timing
        )


if __name__ == "__main__":
    Video_file = filedialog.askopenfilename(title="Select a VideoFile")
    csvfilename = filedialog.asksaveasfilename(
        title="Save CSV name", filetypes=(("CSV File", "*.csv"), ("all files", "*.*"))
    )

    Detectorator = TDetector(
        "config.toml",
        Video_file,
        csvfilename,
        size=32,
        net="Models/yolov8n.onnx",
        show=False,
    )
    print("Starting")
    startjob = datetime.datetime.now()
    count = Detectorator.run()
    duration = datetime.datetime.now() - startjob
    factor = Detectorator.factorspeed(duration)

    print(f"Le système à denombrer {count} Personnes entrante")
    print(f"Le job a pris {duration},(x{factor:.2f})")
