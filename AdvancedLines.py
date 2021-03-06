import cv2
from DistortionCorrection import DistortionCorrection
from PerspectiveTransform import PerspectiveTransform
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from moviepy.editor import VideoFileClip

def color_filter(img):
    '''
    This function filter the white and yellow in a color image with the objective to find lane lines in a road
    change the input image to HSV color space and using cv2.inRange extract the yellow and white lines
    
    :param img: RGB image (after bird view transformation)
    :return: a layer with the yellow and white lines only (X,Y,1 channel)
    
    '''
    imagen_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(imagen_BGR, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (17, 76, 178), (30, 200, 255))

    sensitivity_1 = 68
    white = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(imagen_BGR, cv2.COLOR_BGR2HLS)
    white_2 = cv2.inRange(HSL, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    bit_layer = yellow | white | white_2 | white_3
    return bit_layer

def draw_area(color_filtered, M):
    '''
    This function get the detected lines and fit a second degree polynomial based on the pixels detected.
    Fill the area in the lane as an overlay 
    transform the overlay from bird eyes view to original perspective
    calculate and return the curvature and position with respect to the center of the lane in world space.
    
    :param color_filtered: 1 channel image with detected lines
    :param M: Matrix needed to make the transformation to the original perspective
    :return: newwarp, image with the lane overlay transformed to the original perspective
             curvatrue, return calculated curvature value
             desviacion, return calculated deviation from center of lane.
    '''

    # Set out_image
    out_img = np.zeros_like(color_filtered)
    out_img = np.dstack((out_img, out_img, out_img))
    nwindows = 9
    margin = 100
    minpix = 50

    #Histograma
    histogram = np.sum(color_filtered[int(color_filtered.shape[0] / 2):, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(color_filtered.shape[0] / nwindows)
    nonzero = color_filtered.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = color_filtered.shape[0] - (window + 1) * window_height
        win_y_high = color_filtered.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(start=0, stop=color_filtered.shape[0] - 1, endpoint=color_filtered.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    ym_per_pix = 3/103 # meters per pixel in y dimension
    xm_per_pix = 3.7/792 # meters per pixel in x dimension
    lane_width = 792#px
    # Fit new polynomials to x,y in world space

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    ## MODIFIED TO WORLD SPACE ###
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    ########
    warp_zero = np.zeros_like(color_filtered).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    Minv = inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    curvature = ((left_curverad + right_curverad)/2)* 0.001
    #desviacion = (leftx_current-rightx_current)/100
    #print(leftx_current, rightx_current, image.shape[1] )
    #desviacion_L = image.shape[1]/2 - leftx_current
    #desviacion_R = image.shape[1] - rightx_current
    #desviacion = (desviacion_L - desviacion_R) * xm_per_pix

    #calculate the average of the 30 that are part of the lane lines at the bottom of the image

    left_bottom_position = np.average(leftx[-10])
    right_bottom_position = left_bottom_position + lane_width
    desviacion = ((left_bottom_position + right_bottom_position) / 2 - image.shape[1] / 2) * xm_per_pix

    #print(desviacion, left_bottom_position, right_bottom_position)
    return newwarp, curvature, desviacion


# 1.- Distortion Correction
#  variables
nx = 9
ny = 5
imgPath = '../CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg'
pfilePath = '../CarND-Advanced-Lane-Lines/camera_cal/wide_dist_pickle.p'
#image = cv2.imread('imagenes/test1.jpg')
image = plt.imread('imagenes/test5.jpg')

# 2.- Call class to correct camera distortion
camera = DistortionCorrection()
camera.setvars(nx, ny, imgPath, pfilePath, image)

# 3.- Save Image Matrix,  and Distortion Matrix to pickle file
# Needs to be run only once to calculate the distortion correction matrix
# camera.savepick()

# 4.- Load saved values from pickle file
distp = camera.loadpick()

### Perspective Transform
source_points = np.float32([[578, 460], [264, 670],[1043, 670], [704, 460]])
destination_points = np.float32([[250,0], [250, 720],[1000, 720], [1000, 0]])
perspTrans = PerspectiveTransform()
perspTrans.setSourcePoints(source_points,destination_points)


def procesar_imagen(image):
    '''
    This function is the pipeline to transform an input image 
    Correct distortion
    Transform to bird eyes view
    Extract white and yellow lines
    Calculate curvature and vehicle position
    Compose the final image
    :param image: 
    :return: 
    '''
    undistort_image = cv2.undistort(image, distp["mtx"], distp["dist"], None, distp["mtx"])
    warped, M = perspTrans.warpedTrans(undistort_image)
    color_filtered = color_filter(warped)
    lane, curvature, desv = draw_area(color_filtered, M)
    text = "Lane Curvature:  " + str(round(curvature, 2)) + "km"
    text2 = "Deviation from center: " + str(round(desv,3)) + "mts"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(undistort_image, text, (10, 200), font, 1, (255, 5, 5), 2)
    cv2.putText(undistort_image, text2, (10, 300), font, 1, (5, 5, 255), 2)
    result = cv2.addWeighted(undistort_image, 1, lane, 0.3, 0)

    return result
'''
asd = procesar_imagen(image)
cv2.imshow('Imagen Carril',asd)
#cv2.imwrite('video/result.jpg', asd)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
# Apply process image to video.
white_output = 'D:/aaSDCNDJ/CarND-Advanced-Lane-Lines/result_Project6.mp4'
clip1 = VideoFileClip("D:/aaSDCNDJ/CarND-Advanced-Lane-Lines/project_video.mp4")
white_clip = clip1.fl_image(procesar_imagen)#.subclip(1,3) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
