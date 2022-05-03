import mediapipe as mp
import cv2
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    sucesss, img = cap.read()
    immRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(immRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    print(results.multi_hand_landmarks)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
