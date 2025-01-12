"""computing module"""

import queue
from threading import Thread

import cv2

from analyzer import Analyser
from fpsmeter import FPSMeter
from functions import distance, beautifultime
from static import LIST_COLORS, INFOTIME


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
                passing, _ = self.analys.passing_fences(trajet, shape)
                if passing:
                    self.people[p]["pass"] = True
                    self.counter += 1

    def draw(self, img, detect):
        """The drawing function to see what's is going on"""
        out = self.analys.draw_settings(img.copy())
        shape = img.shape
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
                        self.peoplefifo.put((i // (self.videoinfos[0]), self.counter))

                    if self.show:
                        cv2.imshow("image", self.draw(img, detect))
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
