import cv2

import os

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

img=cv2.imread('dev.png')

imgGray=cv2.cvtColor\
(img,cv2.COLOR_BGR2GRAY)

faces=faceCascade.\
detectMultiScale(imgGray,1.1,4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("dev",img)

cv2.waitKey(0)

