import cv2


class PerspectiveTransform:
    _src = []
    _dst = []


    def __init__(self):
        self._src = []
        self._dst = []

    def setSourcePoints(self, src, dst):
        self._src = src
        self._dst = dst


    def warpedTrans(self, image):
        self._M = cv2.getPerspectiveTransform(self._src, self._dst)
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image,self._M, img_size)
        return warped, self._M