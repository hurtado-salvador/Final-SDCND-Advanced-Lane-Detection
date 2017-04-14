import cv2
import numpy as np

image = cv2.imread('../CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg')

source_points = np.array([[[578, 460], [264, 670],[1043, 670], [704, 460]]])

image = cv2.fillPoly(image,source_points,(0,255,0))

cv2.imwrite('../CarND-Advanced-Lane-Lines/test_images/straight_1.jpg', image)
cv2.imshow('Imagen original',image)
cv2.waitKey(0)
cv2.destroyAllWindows()