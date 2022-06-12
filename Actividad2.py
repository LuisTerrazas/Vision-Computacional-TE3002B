import numpy as np
import cv2
from skimage.filters import threshold_otsu

for i in range(10):
    img = cv2.imread(str(f'Images_sign/img{i}.png'))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(twoDimage, 2, None, criteria, 50, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape)

    cv2.imshow("Kmeans", result_image)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


    def filter_image(image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask
        return np.dstack([r, g, b])


    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh
    filtered = filter_image(img, img_otsu)
    cv2.imshow("FilteredOTSU", filtered)

    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    light_blue = (90, 70, 50)
    dark_blue = (128, 255, 255)

    light_green = (40, 40, 40)
    dark_green = (70, 255, 255)

    light_red = (160, 100, 100)
    dark_red = (179, 255, 255)

    light_red2 = (0, 100, 20)
    darK_red2 = (10, 255, 255)

    maskBlue = cv2.inRange(hsv_img, light_blue, dark_blue)

    maskRed1 = cv2.inRange(hsv_img, light_red, dark_red)
    maskRed2 = cv2.inRange(hsv_img, light_red2, darK_red2)
    maskRed = cv2.bitwise_or(maskRed1, maskRed2)

    resultBlue = cv2.bitwise_and(img, img, mask=maskBlue)
    resultRed = cv2.bitwise_and(img, img, mask=maskRed)

    img_maskBlue_Red = cv2.hconcat([resultRed, resultBlue])

    cv2.imshow("mask Blue Red", img_maskBlue_Red)

    cv2.waitKey()
