import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

imgsize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20

counter = 0
folder = "Data/A"

# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

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
        else:
            k = imgsize / w
            hCal = int(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hCal))
            hGap = (imgsize - hCal) // 2
            imgwhite[hGap:hGap + hCal, :] = imgresize

        cv2.imshow("Image1", img)
        cv2.imshow("Imagewhite", imgwhite)

        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            # Save the image with a .jpg extension
            cv2.imwrite(f'{folder}/Image_{int(time.time())}.jpg', imgwhite)
            print(counter)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()