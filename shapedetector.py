import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing():
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower", "Trackbars",0,255,nothing)
cv2.createTrackbar("Upper", "Trackbars",0,255,nothing)

while True:
    _, frame = cap.read()

    frame = cv2.resize(frame,(500,500))

    l = cv2.getTrackbarPos("Lower","Trackbars")
    u = cv2.getTrackbarPos("Upper", "Trackbars")

    print(l,u)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray,(3,3),0)
    canny = cv2.Canny(imgBlur,l,u)

    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(gray,contours,-1,(255,0,0),3)


    cv2.imshow("Binary video",canny)
    cv2.imshow("Input", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()