import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

for i in range(10):
	img = cv.imread(str(f'Images_sign/img{i}.png'))
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	_,th1 = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)
	
	cv.imshow(str(f"Original Image {i}"), img)
	cv.imshow(str(f"Grayscale Image{i}"), gray)
	cv.imshow(str(f"Binary Image{i}"),th1)
	


	noise = np.random.normal(loc=0, scale=1, size=img.shape)

	noisy = np.clip((img + noise * 500), 0, 1)
	noisy2 = np.clip((img + noise * 500), 0, 1)

	img2 = img * 2
	n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2, 0, 1)
	n4 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.4)), (1 - img2 + 1) * (1 + noise * 0.4) * -1 + 2) / 2, 0, 1)

	cv.imshow("Noise", np.hstack((noisy,noisy2)))

	cv.waitKey(0)
	cv.destroyAllWindows()






