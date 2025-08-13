import cv2
import time
import os
import HandTrackingMinModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Start webcam capture
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height

# adding picture of fingers to project
folderPath = "fingers"
myList = os.listdir(folderPath)  # Get list of image filenames
overlayList = []
for imgPath in myList:
    img = cv2.imread(folderPath + "/" + imgPath)
    overlayList.append(img)  # Add image to overlay list

# Initialize frame rate variables
pTime = 0
# initializing a hand detector with 0.75 confidence
detector = htm.handDetector(detectionCon=0.75)

# index of  Thumb, Index, Middle, Ring, Pinky tips
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList is not None:
        if len(lmList) != 0:
            finger = []
            # detecting Thumb
            if detector.handType == "Right":
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    finger.append(1)
                else:
                    finger.append(0)
            else:
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    finger.append(1)
                else:
                    finger.append(0)
            # detecting other 4 fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    finger.append(1)
                else:
                    finger.append(0)

            totalFingers = finger.count(1)

            h, w, c = overlayList[totalFingers - 1].shape
            img[0:h, 0:w] = overlayList[totalFingers - 1]
            # putting the number of fingers on screen
            cv2.rectangle(img, (0, 300), (150, 480), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (30, 440), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 25)
    # finding fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # showing the fps
    cv2.putText(img, f"fps{int(fps)}", (530, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow('img', img)
    # using word q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
