# import cv2 as cv
# import numpy as np
#
# img = cv.imread('Mask_data_collector/Data/0_fingers/M_0.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# print(img.shape)
# img_flat = np.reshape(img, (-1,1))
# cv.imshow("Image", img)
# print(img_flat.shape)
#
# img = cv.resize(img, None, fx=1/5, fy=1/5)
# cv.imshow("Small Image", img)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
