import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import  Classifier
import numpy as np
import math
import os

imgsize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
classifier=Classifier("model/keras_model.h5","model/labels.txt")
counter = 0
labels=["A","C"]
# Ensure the folder exists


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgsize / h
            wCal = int(k * w)
            imgresize = cv2.resize(imgcrop, (wCal, imgsize))
            wGap = (imgsize - wCal) // 2
            imgwhite[:, wGap:wGap + wCal] = imgresize
            pred,index=classifier.getPrediction(imgwhite)



        else:
            k = imgsize / w
            hCal = int(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hCal))
            hGap = (imgsize - hCal) // 2
            imgwhite[hGap:hGap + hCal, :] = imgresize
            pred, index = classifier.getPrediction(imgwhite)


        cv2.imshow("Image1", img)
        cv2.imshow("Imagewhite", imgwhite)

        key = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()