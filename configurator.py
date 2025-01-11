import cv2
import numpy as np


from anayzer import Analyser



def load_img(url,video=False):
    if video :
        cam = cv2.VideoCapture(url)
        ok , img = cam.read()
        assert ok, "empty img"
        cam.release()
    else :
        img = cv2.imread(url)
    return img


def show(title,img,infos=True):
    if infos :
        height,width,_ = img.shape
        print(f"{title} : H : {height}, W:{width}")
    cv2.imshow(title,img)



if __name__ == "__main__" :
    analys = Analyser("config.toml")
    analys.open()

    assert analys.check()

    url = input("Image url :")
    print(url.split(".")[-1])
    img = load_img(url,url.split(".")[-1] == "mp4")
    show("Base_image",img)
    show("inference image",analys.crop_scale_inferance(img))
    cv2.waitKey(0)