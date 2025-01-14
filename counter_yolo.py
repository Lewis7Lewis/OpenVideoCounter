"""The simple counter

not update

##########################
DO NOT USE
DO NOT USE
DO NOT USE
DO NOT USE
DO NOT USE
DO NOT USE
##########################"""

import csv
import time

import cv2
import numpy as np
import onnxruntime as onx

from static import LIST_COLORS

camera = cv2.VideoCapture


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


class Detector:
    """The counter main process"""

    def __init__(self, feed, loging=None, net="yolov8n.onnx") -> None:
        self.cam = feed
        self.log = loging
        self.people = {}
        self.last_people_id = 0
        self.image_index = 0
        self.counter = 0

        self.net = onx.InferenceSession(
            net, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )  # providers=[ 'CPUExecutionProvider']) #'CUDAExecutionProvider'
        self.input_name = self.net.get_inputs()[0].name
        self.label_name = self.net.get_outputs()[0].name
        self.framerate = int(self.cam.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(self.input_name,self.label_name)
        # self.net = cv2.dnn.readNetFromONNX(net)
        # cv2.dnn.readNetFrom
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

        self.fps = 0
        self.img = None
        self.predictions = None
        self.detect = None

    def predict(self, imgs):
        """Do predictions on batchs images"""
        blob = cv2.dnn.blobFromImages(imgs, 1 / 255, (640, 640), (0, 0, 0), swapRB=True)

        ##self.net.setInput(blob)
        ##self.net.forward()
        return self.net.run(
            [self.label_name], {self.input_name: blob.astype(np.float32)}
        )

    def do_batch(self, size, show=False):
        """Make batchs of images"""
        fin = True
        for _ in range(0, self.frame_count, size):
            imgs = []
            i = 0
            while self.cam.isOpened() and i < size:
                i += 1
                ret, img = self.cam.read()
                if ret:
                    img = cv2.resize(img, (1920, 1080))
                    imgs.append(self.do_pre_image(img))
                else:
                    break
            self.fps = time.time()
            predictions_batch = self.predict(imgs)
            self.fps = time.time() - self.fps
            self.fps = 1 / (self.fps / len(imgs))
            for img, d in zip(imgs, predictions_batch[0]):
                self.image_index += 1
                self.img = img
                self.predictions = d
                self.do_post_image(img, d)
                self.tracking()
                self.clean_people()
                self.count_people()

                if (
                    self.log and self.image_index % (self.framerate * 60) == 0
                ):  # 60 s on rec
                    self.log.rec((self.image_index // self.framerate), self.counter)
                if show:
                    self.draw()
                    cv2.imshow("out", self.draw())
                    if cv2.waitKey(1) == ord("q"):
                        fin = False
                else:
                    if self.image_index % (self.framerate * 10) == 0:  # each 10 s
                        s = self.image_index // self.framerate
                        m = s // 60
                        h = m // 60
                        m = m % 60
                        s = s % 60
                        s = f"{h:02d}:{m:02d}:{s:02d}"
                        print(
                            f"Image {self.image_index} ({self.fps:.2f} FPS (x{self.fps/self.framerate:.2f}) : {s} ({self.image_index/self.frame_count:.2%}) : Compteur {self.counter} personnes"
                        )
            if fin is False:
                break
        return self.counter

    def do_once(self, img, show=False):
        """Computes everything for one image"""
        img = self.do_pre_image(img)
        self.img = img
        self.fps = time.time()
        self.predictions = self.predict([img])
        self.fps = time.time() - self.fps
        self.fps = 1 / self.fps

        self.do_post_image(img, self.predictions)
        self.tracking()
        self.clean_people()
        self.count_people()
        if show:
            ## to do
            pass
        return self.draw()

    def do_pre_image(self, img):
        """Prepare the image for the inferance"""
        img = cv2.resize(img, [i // 2 for i in reversed(img.shape[:2])])
        shape = img.shape
        img = img[:, 200 : shape[1] - 300]
        return img

    def do_post_image(self, img, predictions):
        """Do post nead befor calculations"""
        shape = img.shape
        [height, width, _] = shape

        # length = max((height, width))
        predictions = np.array(cv2.transpose(predictions))
        boxes = []
        scores = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for d in predictions:
            classes_scores = d[4:]
            (_, max_score, _, (_, max_class_index)) = cv2.minMaxLoc(classes_scores)
            if max_score >= 0.25 and max_class_index == 0:
                box = np.array(
                    [
                        (d[0] - (0.5 * d[2])) * (width / 640),
                        (d[1] - (0.5 * d[3])) * (height / 640),
                        d[2] * (width / 640),
                        d[3] * (height / 640),
                    ]
                )
                if abs((height // 2) - (box[1] + box[3] // 2)) >= 75:
                    continue
                if box[2] >= 150 and box[3] >= 150:
                    continue
                boxes.append(box)
                scores.append(max_score)

        # Apply NMS (Non-maximum suppression)
        self.detect = []
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        if len(result_boxes) > 0:
            for index in result_boxes:  # pylint: disable=E1133
                # index = result_boxes[i]
                box = np.array(boxes[index], dtype=np.int16)
                self.detect.append(box)

    def tracking(self):
        """Tracking algorythme to follow people through the virtual fence"""
        for x, y, w, h in self.detect:
            midle = (x + (w // 2), y + (h // 2))
            if len(self.people) >= 1:
                d = [
                    (ident, distance(midle, p["pose"][-1]))
                    for ident, p in self.people.items()
                    if self.people[ident]["last_seen"] + 10 >= self.image_index
                ]
                d.sort(key=lambda x: x[1])
                if len(d) >= 1 and d[0][1] <= 30:
                    i, d = d[0]
                    self.people[i]["pose"].append(midle)
                    self.people[i]["last_seen"] = self.image_index
                    self.people[i]["seen"] += 1
                else:
                    self.people[self.last_people_id] = {
                        "pose": [midle],
                        "last_seen": self.image_index,
                        "pass": False,
                        "rec": 1,
                        "seen": 1,
                    }
                    self.last_people_id += 1
            else:
                self.people[self.last_people_id] = {
                    "pose": [midle],
                    "last_seen": self.image_index,
                    "pass": False,
                    "rec": 1,
                    "seen": 1,
                }
                self.last_people_id += 1

    def draw(self):
        """Graphical interface fonction"""
        out = self.img.copy()
        shape = self.img.shape
        [height, width, _] = shape
        cv2.line(out, (0, height // 2), (width, height // 2), (0, 0, 255), 2)
        for x, y, w, h in self.detect:
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
            f"FPS : {self.fps:.2f}", cv2.FONT_HERSHEY_PLAIN, 1, 2
        )
        cv2.putText(
            out,
            f"FPS : {self.fps:02.2f}",
            (shape[1] - 10 - text_size[0], shape[0] - 10 - text_size[1]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return out

    def clean_people(self):
        """Clean the memory"""
        for p in list(self.people.keys()):
            if not self.people[p]["last_seen"] + 10 >= self.image_index:
                del self.people[p]

    def count_people(self):
        """Counter function"""
        shape = self.img.shape
        for p in list(self.people.keys()):
            data = self.people[p]
            if (
                True
                and len(data["pose"]) > 1
                and data["pass"] is False
                and self.people[p]["last_seen"] + 10 >= self.image_index
            ):
                trajet = data["pose"][-2:]
                if (
                    distance(*trajet) >= 5
                    and determinant(vect(*trajet), (-1, 0)) / distance(*trajet) >= 0.80
                    and abs(shape[0] // 2 - data["pose"][-1][1]) <= 20
                ):
                    self.people[p]["pass"] = True
                    self.counter += 1


class Loger:
    """The Loger do the csv records"""

    def __init__(self, csvfilename):
        self.file = open(csvfilename, "w", newline="", encoding="utf-8")
        self.csv = csv.writer(self.file)
        self.init_file()
        self.i = 0

    def init_file(self):
        """Initalise the file"""
        self.csv.writerow(["Timing (secondes)", "people"])

    def rec(self, timing, people):
        """Add a record"""
        self.i += 1
        self.csv.writerow([timing, people])
        if self.i % 5 == 0:
            self.flush()

    def close(self):
        """Close file"""
        self.file.close()

    def flush(self):
        """Flush on the disk"""
        self.file.flush()


if __name__ == "__main__":

    from tkinter import filedialog, messagebox

    Video_file = filedialog.askopenfilename(title="Select a VideoFile")
    filename = filedialog.asksaveasfilename(
        title="Save CSV name", filetypes=(("CSV File", "*.csv"), ("all files", "*.*"))
    )

    cam = cv2.VideoCapture(Video_file)

    log = Loger(filename)
    print("Initialisation")
    Detectorator = Detector(cam, log, net="yolov8n.onnx")
    if messagebox.askyesno(
        "Interface Graphique", "Voulez vous une interfaces Graphique ?"
    ):
        prog_start = time.time()
        try:
            Detectorator.do_batch(2, show=True)
        except KeyboardInterrupt:
            pass
    else:
        prog_start = time.time()
        try:
            Detectorator.do_batch(30, show=False)
        except KeyboardInterrupt:
            pass
    cam.release()
    log.rec(Detectorator.frame_count // Detectorator.framerate, Detectorator.counter)
    log.close()
    print(
        f"Compute time of {time.time()-prog_start:.2f}s ~  {(time.time()-prog_start)/60:.1f} minutes "
    )
    print(f"Le système denombrer {Detectorator.counter} Personnes entrante")
