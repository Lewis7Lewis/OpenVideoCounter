"""The analyser setting that make the specific calculation of the counter"""

import time
import cv2
import numpy as np
import tomlkit


from functions import distance, intersects, determinant, vect, non_max_suppression_fast




class Crops:
    """Saving parameters of croping"""
    left = 0
    right = 0
    top = 0
    bottom = 0

    def to_dict(self):
        """to dict for saving"""
        return {"left": self.left, "right": self.right, "top": self.top, "bottom": self.bottom}
    
    def from_dict(self,dic:dict):
        """From dic to load"""
        self.left = dic.get("left",self.left)
        self.right = dic.get("right",self.right)
        self.top = dic.get("top",self.top)
        self.bottom = dic.get("bottom",self.bottom)

class Filters:
    """Saving Filters parameters"""
    maxsize =150
    tracking_distance = 75
    min_walking_distance =5
    max_tracking_distance = 40

    def to_dict(self):
        """To dict for saving"""
        return {
                "maxsize": self.maxsize,
                "trackingDistance": self.tracking_distance,
                "minWalkingDistance": self.min_walking_distance,
                "maxTrackingDistance": self.max_tracking_distance,
            }
    
    def from_dict(self,dic:dict):
        """From dict to init"""
        self.maxsize = dic.get("maxsize",self.maxsize)
        self.tracking_distance = dic.get("trackingDistance",self.tracking_distance)
        self.min_walking_distance = dic.get("minWalkingDistance",self.min_walking_distance)
        self.max_tracking_distance = dic.get("maxTrackingDistance",self.max_tracking_distance)

class Fence :
    """Fence parameters class"""
    l = 50
    r = 50
    angle = 20

    def to_dict(self):
        """To dict parameters"""
        return {"l": 50, "r": 50, "angle": 20}
    
    def from_dict(self,dic:dict):
        """Load from dict"""
        self.l = dic.get("l",self.l)
        self.r = dic.get("r",self.r)
        self.angle = dic.get("angle",self.angle)

class Analyser:
    """The Analyser hold all computation method link to config file"""

    def __init__(self, file=None):

        self.configfile = file
        self.config = {}
        self.filters = Filters()
        self.crops = Crops()
        self.fence = Fence()
        self.to_dict()

    def to_dict(self):
        """To save parameters"""
        self.config = {
            "crop": self.crops.to_dict(),
            "filters": self.filters.to_dict(),
            "fence": self.fence.to_dict(),
        }
        return self.config
    
    def from_dict(self,dic:dict):
        """Import parameters"""
        self.crops.from_dict(dic.get("crop",{}))
        self.filters.from_dict(dic.get("filters",{}))
        self.fence.from_dict(dic.get("fence",{}))

    def get_max_tracking_dist(self):
        """Return the maxtracking distance parameter for the tracking"""
        return self.filters.max_tracking_distance

    def open(self, file=None):
        """open and load the config file"""
        if file is None:
            file = self.configfile

        with open(file, "r", encoding="utf-8") as fic:
            self.config = dict(tomlkit.load(fic))

        self.from_dict(self.config)

    def save(self, file=None):
        """Save the config file"""
        if file is None:
            file = self.configfile

        self.to_dict()

        with open(file, "w", encoding="utf-8") as fic:
            tomlkit.dump(self.config, fic)

    def check(self):
        """Check all the paramters"""
        crop = self.crops
        ok = True
        if crop.left + crop.right >= 100:
            ok = False
            print("Error : too much crop on left right")
        if crop.top + crop.bottom >= 100:
            ok = False
            print("Error : too much crop on top bottom")

        return ok

    def crop_scale_inferance(self, img):
        """Prepare the image to inference by croping and scaling"""
        h, w, _ = img.shape
        l, r, t, b = self.crops.left,self.crops.right,self.crops.top,self.crops.bottom
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
                    ],dtype=np.int64
                )
                x, y, w, h = box
                mx, my = (x + (w // 2), y + (h // 2))
                fl = self.fence.l
                fr = self.fence.r
                fy = ((mx / width) * (fr - fl) + fl) * height / 100
                if (
                    abs(my - fy) >= self.config["filters"]["trackingDistance"]
                ):  # position filtering
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
        #result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        
        if False and len(result_boxes) > 0:
            for index in result_boxes:  # pylint: disable=E1133
                box = np.array(boxes[index], dtype=np.int16)
                detect.append(box)

        result_boxes = non_max_suppression_fast(np.array(boxes), 0.45)
        return detect

    def passing_fences(self, trajet, shape):
        """Check with the config if a traject path pass the fence"""
        dt = distance(*trajet)
        if not dt >= self.filters.min_walking_distance:
            return False, "minWalkingDistance"

        [height, width, _] = shape
        fl = self.fence.l
        fr = self.fence.r
        fp = np.array([(0, fl * height / 100), (width, fr * height / 100)])

        if not intersects(trajet, fp):
            return False, "No intersection"

        fv = np.array((-np.sqrt(1 - ((fl - fr) / 100) ** 2), (fl - fr) / 100))
        if not determinant(vect(*trajet), fv) / distance(*trajet) >= np.cos(
            np.deg2rad(self.fence.angle) / 2
        ):
            return False, "Not in the aceptence cone"

        return True, "Perfect"

    def draw_settings(self, img):
        """Draw the fence"""
        [height, width, _] = img.shape
        fl = self.fence.l
        fr = self.fence.r
        fp = np.array([(0, fl * height / 100), (width, fr * height / 100)])
        img = cv2.line(img, *fp.astype(np.int16), (0, 0, 255), 2)
        return img
