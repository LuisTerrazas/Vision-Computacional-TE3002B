import cv2 as cv
import numpy as np

for i in range(5):
    img = cv.imread(str(f'Images_Building/img{i}.jpg'))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray,50,150,apertureSize=3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100, maxLineGap=10)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv.imwrite("houghLines.jpg", img)
    cv.imshow("Hough Lines in Building", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

