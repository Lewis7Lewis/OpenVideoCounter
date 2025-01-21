"""Check allthe config on images"""

import cv2
import numpy as np

from analyzer import Analyser


def load_img(url, video=False):
    """Load an image to test"""
    if video:
        cam = cv2.VideoCapture(url)
        ok, img = cam.read()
        assert ok, "empty img"
        cam.release()
    else:
        img = cv2.imread(url)
    return img


def show(title, img, infos=True):
    """Show an image and show details"""
    if infos:
        height, width, _ = img.shape
        print(f"{title} : H : {height}, W:{width}")
    cv2.imshow(title, img)


if __name__ == "__main__":
    analys = Analyser("config.toml")
    analys.open()

    ok,error = analys.check()
    assert ok, error 

    imgurl = input("Image url :")
    print(imgurl.split(".")[-1])
    image = load_img(imgurl, imgurl.split(".")[-1] == "mp4")
    show("Base_image", image)
    resize = analys.crop_scale_inferance(image)
    show("inference image", analys.draw_settings(resize))
    cv2.waitKey(0)
