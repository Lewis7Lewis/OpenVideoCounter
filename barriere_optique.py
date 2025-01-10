import cv2
import numpy as np

cam = cv2.VideoCapture(0)
fin = True

size = np.array([40,60])
base = np.zeros(np.concatenate([size,[3]]),dtype=np.uint8)
old= False
counter  = 0
while cam.isOpened() and fin :
    ok, img = cam.read()
    if ok :
        out = img.copy()
        pt1 = np.array([x//2 for x in reversed(img.shape[:2])])-(size[::-1]//2)
        pt2 = np.array([x//2 for x in reversed(img.shape[:2])])+(size[::-1]//2)
        out= cv2.rectangle(out,pt1,pt2,(0,0,255),thickness= 2)
        cible = img[pt1[1]:pt1[1]+(size[0]),pt1[0]:pt1[0]+(size[1])]

        diff = np.array(np.abs(np.array(cible,dtype = np.int8)- np.array(base,dtype=np.int8)),dtype=np.uint8)
        diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        mean = diff.mean()
        if mean <= 255*0.05 : #5%
            out = cv2.rectangle(out,pt1,pt2,(0,255,0),thickness= 4)
            if old == False :
                counter +=1
                print(counter)
            old = True
        else :
            old = False
        diff = cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)

        cv2.imshow("out",out)
        cv2.imshow("Base",cv2.hconcat([base,cible,diff]))
        key = cv2.waitKey(1)
        if key == ord("q") : 
            fin = False
        
        elif key == ord("c"):
            base = img[pt1[1]:pt1[1]+(size[0]),pt1[0]:pt1[0]+(size[1])]




        