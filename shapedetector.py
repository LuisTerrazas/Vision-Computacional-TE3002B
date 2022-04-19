import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

def nada():
    pass

cv.namedWindow("Trackbars")
cv.createTrackbar("Lower", "Trackbars",0,255,nada)
cv.createTrackbar("Upper", "Trackbars",0,255,nada)

while True:
    _, frame = cap.read()
    frame = cv.resize(frame,(500,500))

    l = cv.getTrackbarPos("Lower","Trackbars")
    u = cv.getTrackbarPos("Upper", "Trackbars")

    print(l,u)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(gray,(3,3),0)
    canny = cv.Canny(imgBlur,l,u)

    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(gray,contours,-1,(255,0,0),3)

    cv.imshow("Canny edges",canny)
    cv.imshow("Output", gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()