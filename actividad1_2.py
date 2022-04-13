# import the opencv library
import cv2
import cv2 as cv
import numpy as np
from random import randint

# define a video capture object
vid = cv.VideoCapture(0)

while (True):

    ret, frame = vid.read()
    imgContour = frame.copy()

    imgBlur = cv.GaussianBlur(frame,(7,7),1)
    gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 100, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow("Canny", edges)
    cv.imshow("Hough Lines in Building", frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
