import cv2
import imageio

yuz_cascade = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")


def tespit(frame):
    gri_yap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_cascade.detectMultiScale(gri_yap,1.3,5)
    for (x,y,w,h) in yuzler:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    return frame

fotograf = imageio.imread("istediginizfotografınadı.uzantısı")
fotograf = tespit(frame=fotograf)
imageio.imwrite("istediğinizfotografınyuzutanınmışhalininadı.uzantısı",fotograf)
