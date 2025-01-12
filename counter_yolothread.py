"""The treaded couter to maximise performances with FIFOs"""

import csv
import datetime
import queue
import time
from threading import Thread
from tkinter import filedialog

import cv2
import numpy as np
import onnxruntime as onx

from static import LIST_COLORS

from anayzer import Analyser


sess_options = onx.SessionOptions()
sess_options.enable_profiling = False
sess_options.execution_mode = onx.ExecutionMode.ORT_PARALLEL
trt_ep_options = {
    "trt_timing_cache_enable": True,
    "trt_max_workspace_size": 2147483648,
    "trt_dla_enable": True,
}
cuda_ep_option = {}


provider = [
    ("CUDAExecutionProvider", cuda_ep_option)
]  # ('TensorrtExecutionProvider',trt_ep_options),

provider = onx.get_available_providers()


def determinant(u, v):
    """Calculate the determinant of a 2scalar vector"""
    return u[0] * v[1] - u[1] * v[0]


def vect(a: tuple, b: tuple) -> tuple:
    """Calculate the vector of a 2 points"""
    return (b[0] - a[0], b[1] - a[1])


def distance(a, b):
    """calculate the euclerien distance(norm L2)"""
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5


last_time = time.time()


INFOTIME = 10


def beautifultime(s):
    """Make a string from a second time"""
    m = s // 60
    h = m // 60
    m = m % 60
    s = s % 60
    return f"{h:2}:{m:0>2d}:{s:0>2d}"


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
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        self.framerate = int(self.vcap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print("[Exiting] No more frames to read")
            exit(0)
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        #                       while the program is executing
        self.i = 0

    def start(self):
        """Start the threaded worker"""
        self.stopped = False
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
                self.fifo.put(
                    (self.i, self.analys.crop_scale_inferance(self.frame)), True
                )
            if self.grabbed is False:
                self.fifo.put(("end", []), True)
                print("[Exiting] No more frames to read")
                self.stopped = True
                break
            if self.i % (INFOTIME * self.framerate) == 0:
                print(
                    f"Reader    : {self.i} images ({beautifultime(self.i//self.framerate)}) (FPS:{self.fps.fps:.2f})"
                )

        self.vcap.release()

    def read(self):
        """get the last read image"""
        return self.frame

    def stop(self):
        """Stop the worker"""
        self.stopped = True

    def join(self):
        """Wait untile the worket finish"""
        self.t.join()


class Inferance:
    """A threaded worker to perform detection inference"""

    def __init__(
        self,
        imgsfifo: queue.Queue,
        predfifo: queue.Queue,
        analys: Analyser,
        videoinfos=None,
        net="yolov8n.onnx",
        size=30,
    ) -> None:
        self.size = size
        self.analys = analys
        self.fps = FPSMeter()
        if videoinfos is None:
            self.videoinfos = [20, 120000]
        else:
            self.videoinfos = videoinfos
        self.imgsfifo = imgsfifo
        self.predfifo = predfifo
        self.net = onx.InferenceSession(
            net, providers=provider, sess_options=sess_options
        )  # ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.input_name = self.net.get_inputs()[0].name
        self.label_name = self.net.get_outputs()[0].name
        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True
        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        # while the program is executing
        self.batch = []

    def predict(self, imgs):
        """Make the prediction from the model"""
        blob = cv2.dnn.blobFromImages(imgs, 1 / 255, (640, 640), (0, 0, 0), swapRB=True)
        return self.net.run(
            [self.label_name], {self.input_name: blob.astype(np.float32)}
        )

    def start(self):
        """Start the worker"""
        self.stopped = False
        self.t.start()

    def update(self):
        """The thread main process"""
        while not self.stopped:
            try:
                i, img = self.imgsfifo.get(True, 1)

                if i == "end":
                    self.do_batch()
                    self.predfifo.put((i, [], []), True)
                    print("[End INFERANCE]")
                    self.stop()
                else:
                    # print(img.shape)
                    self.batch.append((i, img))
            except queue.Empty:
                pass
            else:
                if len(self.batch) == self.size:
                    self.do_batch()
                    self.batch = []
                if i != "end" and i % (INFOTIME * self.videoinfos[0]) == 0:
                    print(
                        f"Inference : {i} images ({beautifultime(i//self.videoinfos[0])}) (FPS:{self.fps.fps:.2f})"
                    )

    def do_batch(self):
        """Agregate images to make batch"""
        if len(self.batch) > 0:
            self.fps.update(1)
            predictions = self.predict([img for _, img in self.batch])
            self.fps.update(len(self.batch))
            for (i, img), prediction in zip(self.batch, predictions[0]):
                self.predfifo.put((i, img, prediction), True)

    def stop(self):
        """Stop the worker"""
        self.stopped = True

    def join(self):
        """Wait until finish task"""
        self.t.join()


class Computing:
    """A threded worker to compute the infrered data"""

    def __init__(
        self,
        predfifo: queue.Queue,
        peoplefifo: queue.Queue,
        analys: Analyser,
        videoinfos=None,
        show=False,
    ) -> None:
        self.people = {}
        self.analys = analys
        self.show = show
        if videoinfos is None:
            self.videoinfos = [20, 120000]
        else:
            self.videoinfos = videoinfos
        self.last_people_id = 0
        self.counter = 0
        self.predfifo = predfifo
        self.peoplefifo = peoplefifo
        self.fps = FPSMeter()

        self.stopped = True
        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        # while the program is executing

        # predefine
        self.i = None

    def tracking(self, detect):
        """The tracking function to calculate paths"""
        trackdist = self.analys.get_max_tracking_dist()
        for x, y, w, h in detect:
            midle = (x + (w // 2), y + (h // 2))
            if len(self.people) >= 1:
                d = [
                    (ident, distance(midle, p["pose"][-1]))
                    for ident, p in self.people.items()
                    if self.people[ident]["last_seen"] + 10 >= self.i
                ]
                d.sort(key=lambda x: x[1])
                if len(d) >= 1 and d[0][1] <= trackdist:
                    i, d = d[0]
                    self.people[i]["pose"].append(midle)
                    self.people[i]["last_seen"] = self.i
                    self.people[i]["seen"] += 1
                else:
                    self.people[self.last_people_id] = {
                        "pose": [midle],
                        "last_seen": self.i,
                        "pass": False,
                        "rec": 1,
                        "seen": 1,
                    }
                    self.last_people_id += 1
            else:
                self.people[self.last_people_id] = {
                    "pose": [midle],
                    "last_seen": self.i,
                    "pass": False,
                    "rec": 1,
                    "seen": 1,
                }
                self.last_people_id += 1

    def clean_people(self):
        """Clean the tracking path to clear the memory"""
        for p in list(self.people.keys()):
            if not self.people[p]["last_seen"] + (self.videoinfos[0] * 2) >= self.i:
                del self.people[p]

    def count_people(self, img):
        """Counting the numbers of trepassing"""
        shape = img.shape
        for p in list(self.people.keys()):
            data = self.people[p]
            if (
                True
                and len(data["pose"]) > 1
                and data["pass"] is False
                and self.people[p]["last_seen"] + 10 >= self.i
            ):
                trajet = data["pose"][-2:]
                passing, error = self.analys.passing_fences(trajet, shape)
                if passing:
                    self.people[p]["pass"] = True
                    self.counter += 1

    def draw(self, img, detect):
        """The drawing function to see what's is going on"""
        out = self.analys.draw_settings(img.copy())
        shape = img.shape
        [height, width, _] = shape
        for x, y, w, h in detect:
            xy = x, y
            x1y1 = x + w, y + h
            midle = (x + (w // 2), y + (h // 2))
            cv2.circle(out, midle, 2, (0, 255, 255), 2)
            cv2.rectangle(out, xy, x1y1, tuple([255, 255, 0]), 2)

        for i, p in enumerate(list(self.people.keys())):
            data = self.people[p]
            l = iter(data["pose"])
            last = next(l)
            for pt in l:
                cv2.line(out, last, pt, color=LIST_COLORS[i % 10], thickness=2)
                last = pt

        text_size, _ = cv2.getTextSize(
            f"Compte : {self.counter} Personnes", cv2.FONT_HERSHEY_PLAIN, 1, 2
        )
        cv2.putText(
            out,
            f"Compte : {self.counter} Personnes",
            (10, 10 + text_size[1]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        text_size, _ = cv2.getTextSize(
            f"FPS : {self.fps.fps:.2f}", cv2.FONT_HERSHEY_PLAIN, 1, 2
        )
        cv2.putText(
            out,
            f"FPS : {self.fps.fps:02.2f}",
            (shape[1] - 10 - text_size[0], shape[0] - 10 - text_size[1]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return out

    def start(self):
        """Start the worker"""
        self.stopped = False
        self.t.start()

    def update(self):
        """the threaded main process"""
        while not self.stopped:
            try:
                i, img, prediction = self.predfifo.get(True, 1)
            except queue.Empty:
                pass
            else:
                if i == "end":
                    self.peoplefifo.put(("end", 0))
                    print("END Computing")
                    self.stop()
                else:
                    self.i = i
                    detect = self.analys.overlap_supression(img, prediction)
                    self.tracking(detect)
                    self.clean_people()
                    self.count_people(img)

                    # cv2.imshow("image",self.draw(img,detect))
                    # if ord("q")==cv2.waitKey(1) :
                    #    self.stop()
                    if i % (self.videoinfos[0] * 60) == 0:
                        self.peoplefifo.put(
                            (i // (60 * self.videoinfos[0]), self.counter)
                        )

                    if self.show:
                        
                        cv2.imshow("image",self.draw(img, detect))
                        cv2.waitKey(10)

                    self.fps.update()
                    if i % (INFOTIME * self.videoinfos[0]) == 0:
                        print(
                            f"Computing : {i} images ({beautifultime(i//self.videoinfos[0])}) (FPS:{self.fps.fps:.2f}) ; count {self.counter},"
                        )

    def stop(self):
        """stop the worker"""
        self.stopped = True

    def join(self):
        """Wait until finish"""
        self.t.join()


class Loger:
    """A threaded worker to Log and gat nices tables"""

    def __init__(self, filename, fifo: queue.Queue):
        self.file = open(filename, "w", newline="", encoding="utf-8")
        self.fifo = fifo
        self.csv = csv.writer(self.file)
        self.init_file()
        self.i = 0
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        # while the program is executing
        self.i = 0

    def init_file(self):
        """Init the nice table"""
        self.csv.writerow(["Timing (secondes)", "people"])

    def rec(self, timing, people):
        """Add a record"""
        self.i += 1
        self.csv.writerow([timing, people])
        if self.i % 5 == 0:
            self.flush()

    def close(self):
        """Close the file"""
        self.file.close()

    def update(self):
        """The threaded main process"""
        while not self.stopped:
            t, p = self.fifo.get(True)
            if t == "end":
                self.stop()
            else:
                self.rec(t, p)

    def flush(self):
        """Flush datas to the disk"""
        self.file.flush()

    def start(self):
        """Start the worker"""
        self.stopped = False
        self.t.start()

    def stop(self):
        """Stop the worker"""
        self.stopped = True

    def join(self):
        """Wait until finish"""
        self.t.join()


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
            self.cam.frame_count / (self.cam.framerate * 60), self.compute.counter
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
