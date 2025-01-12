"""The analyser setting that make the specific calculation of the counter"""

import cv2
import numpy as np
import tomlkit


def determinant(u, v):
    """Calculate the determinant of a 2scalar vector"""
    return u[0] * v[1] - u[1] * v[0]


def vect(a: tuple, b: tuple) -> tuple:
    """Calculate the vector of a 2 points"""
    return (b[0] - a[0], b[1] - a[1])


def distance(a, b):
    """calculate the euclerien distance(norm L2)"""
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5


def intersects(s0, s1):
    """Check if to line intersect"""
    dx0 = s0[1][0] - s0[0][0]
    dx1 = s1[1][0] - s1[0][0]
    dy0 = s0[1][1] - s0[0][1]
    dy1 = s1[1][1] - s1[0][1]
    p0 = dy1 * (s1[1][0] - s0[0][0]) - dx1 * (s1[1][1] - s0[0][1])
    p1 = dy1 * (s1[1][0] - s0[1][0]) - dx1 * (s1[1][1] - s0[1][1])
    p2 = dy0 * (s0[1][0] - s1[0][0]) - dx0 * (s0[1][1] - s1[0][1])
    p3 = dy0 * (s0[1][0] - s1[1][0]) - dx0 * (s0[1][1] - s1[1][1])
    return (p0 * p1 <= 0) & (p2 * p3 <= 0)


class Analyser:
    """The Analyser hold all computation method link to config file"""

    def __init__(self, file=None):

        self.configfile = file
        self.config = {
            "crop": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "filters": {
                "maxsize": 150,
                "trackingDistance": 75,
                "minWalkingDistance": 5,
                "maxtrackingDistance" : 40
            },
            "fence": {"l": 50, "r": 50, "angle": 20},
        }

    def get_max_tracking_dist(self):
        return self.config["filters"]["maxtrackingDistance"]

    def open(self, file=None):
        """open and load the config file"""
        if file is None:
            file = self.configfile

        with open(file, "r", encoding="utf-8") as fic:
            self.config = dict(tomlkit.load(fic))

    def save(self, file=None):
        """Save the config file"""
        if file is None:
            file = self.configfile

        with open(file, "w", encoding="utf-8") as fic:
            tomlkit.dump(self.config, fic)

    def check(self):
        """Check all the paramters"""
        conf = self.config
        crop = conf["crop"]
        ok = True
        if crop["left"] + crop["right"] >= 100:
            ok = False
            print("Error : too much crop on left right")
        if crop["top"] + crop["bottom"] >= 100:
            ok = False
            print("Error : too much crop on top bottom")

        return ok

    def crop_scale_inferance(self, img):
        """Prepare the image to inference by croping and scaling"""
        h, w, _ = img.shape
        l, r, t, b = [
            self.config["crop"][v] for v in ("left", "right", "top", "bottom")
        ]
        img = img[int(h * t / 100) : h - int(h * b / 100)]
        img = img[:, int(w * l / 100) : w - int(w * r / 100)]
        # Scale for the inference
        f = 640 / min(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f)
        return img

    def overlap_supression(self, img, predictions):
        """Do the post need befor calculating"""
        shape = img.shape
        [height, width, _] = shape

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
                x, y, w, h = box
                mx, my = (x + (w // 2), y + (h // 2))
                fl = self.config["fence"]["l"]
                fr = self.config["fence"]["r"]
                fy = ((mx / width) * (fr - fl) + fl) * height/100
                if (
                    abs(my - fy) >= self.config["filters"]["trackingDistance"]
                ):  ##position filtering
                    continue
                if (
                    w >= self.config["filters"]["maxsize"]
                    and h >= self.config["filters"]["maxsize"]
                ):  # size fitlers
                    continue
                boxes.append(box)
                scores.append(max_score)

        # Apply NMS (Non-maximum suppression)
        detect = []
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        if len(result_boxes) > 0:
            for index in result_boxes:  # pylint: disable=E1133
                box = np.array(boxes[index], dtype=np.int16)
                detect.append(box)

        return detect

    def passing_fences(self, trajet, shape):
        """Check with the config if a traject path pass the fence"""
        dt = distance(*trajet)
        if not dt >= self.config["filters"]["minWalkingDistance"]:
            return False, "minWalkingDistance"

        [height, width, _] = shape
        fl = self.config["fence"]["l"]
        fr = self.config["fence"]["r"]
        fp = np.array([(0, fl * height / 100), (width, fr * height / 100)])

        if not intersects(trajet, fp):
            return False, "No intersection"

        fv = np.array((-np.sqrt(1 - ((fl - fr) / 100) ** 2), (fl - fr) / 100))
        if not determinant(vect(*trajet), fv) / distance(*trajet) >= np.cos(
            np.deg2rad(self.config["fence"]["angle"]) / 2
        ):
            return False, "Not in the aceptence cone"

        return True, "Perfect"

    def draw_settings(self, img):
        """Draw the fence"""
        [height, width, _] = img.shape
        fl = self.config["fence"]["l"]
        fr = self.config["fence"]["r"]
        fp = np.array([(0, fl * height / 100), (width, fr * height / 100)])
        img = cv2.line(img, *fp.astype(np.int16), (0, 0, 255),2)

        return img


if __name__ == "__main__":
    analys = Analyser("config.toml")
    analys.open()
    print(analys.config)
    shape = np.array([640, 820])  # width,height
    import matplotlib.pyplot as plt

    midle = shape // 2
    fig = plt.figure("Travels graph")
    plt.axline(
        (0, analys.config["fence"]["l"] * shape[1] / 100),
        (shape[0], shape[1] * analys.config["fence"]["r"] / 100),
    )
    for angle in np.linspace(0, np.pi * 2, 4 * 6, endpoint=False):
        trajet = np.array([[np.cos(angle), np.sin(angle)], [-np.cos(angle), -np.sin(angle)]])
        trajet *= analys.config["filters"]["minWalkingDistance"]
        trajet += midle
        passing, er = analys.passing_fences(trajet, np.concat([shape[::-1], [0]]))
        print("state :", passing, er, "points :", trajet, "vector :", (trajet[1] - trajet[0]))

        if passing:
            plt.quiver(
                *trajet[0],
                *(trajet[1] - trajet[0]),
                color="g",
                scale=0.2,
                angles="xy",
                scale_units="xy"
            )
        else:
            plt.quiver(
                *trajet[0],
                *(trajet[1] - trajet[0]),
                color="r",
                scale=0.2,
                angles="xy",
                scale_units="xy"
            )
    # plt.axis('equal')
    plt.xlim(0, shape[0])
    plt.ylim(shape[1], 0)
    plt.show()
