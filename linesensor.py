import numpy as np
import cv2

cap = cv2.VideoCapture(0)
li = np.r_[100:120]
bgsub = cv2.createBackgroundSubtractorKNN()
while True:
    ret, frame = cap.read()
    if (not ret) or cv2.waitKey(25) & 0xFF == ord('q'):
        break
    gray = cv2.cvtColor(frame[:, li, :], cv2.COLOR_BGR2RGB)
    fg = bgsub.apply(gray)
    if np.sum(fg) > 2000:
        frame[:, li, :2] = 0
    else:
        frame[:, li, 1:] = 0
    cv2.imshow("fg", fg)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
