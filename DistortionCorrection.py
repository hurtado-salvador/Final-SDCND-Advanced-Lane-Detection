import numpy as np
import glob
import matplotlib.image as mpimg
import cv2
import pickle

class DistortionCorrection:

    _nx = 0
    _ny = 0
    _imgpts = []
    _objpts = []
    _imgPath = ""
    _pfilePath = ""
    _img = ""

    def __init__(self):
        self._nx = 0
        self._ny = 0
        self._imgPath = ""
        self._imgpts = []
        self._objpts = []
        self._images = []
        self._corners = []
        self._ret = False
        self._pfilePath = ""
        self._img = ""

    def setvars(self, nx, ny, imgPath, pFilePath, img):
        self._nx = nx
        self._ny = ny
        self._imgPath = imgPath
        self._pfilePath = pFilePath
        self._img = img

    def setobjp(self):
        self._objp = np.zeros((self._nx * self._ny, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)
        return self._objp

    def getimages(self):
        self._images = glob.glob(self._imgPath)
        return self._images

    def getpoints(self):
        self._images = self.getimages()
        self._objp = self.setobjp()
        for fname in self._images:
            image = mpimg.imread(fname)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self._ret, self._corners = cv2.findChessboardCorners(gray, (self._nx, self._ny), None)
            if self._ret == True:
                self._imgpts.append(self._corners)
                self._objpts.append(self._objp)
        return self._objpts, self._imgpts

    def cal_undistort(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self._objpts, self._imgpts, (self._img.shape[0], self._img.shape[1]), None, None)
        undist = cv2.undistort(self._img, mtx, dist, None, mtx)
        return undist, mtx, dist

    def savepick(self):
        self._objpts, self._imgpts = self.getpoints()
        self._dist_pickle = {}
        self._dst = []
        self._dst, self._dist_pickle["mtx"], self._dist_pickle["dist"] = self.cal_undistort()
        pickle.dump(self._dist_pickle, open(self._pfilePath, "wb"))

    def loadpick(self):
        distp = pickle.load(open(self._pfilePath, "rb"))
        return distp