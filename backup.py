import cv2
from DistortionCorrection import DistortionCorrection
from PerspectiveTransform import PerspectiveTransform
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from moviepy.editor import VideoFileClip


def color_filter(img):
    imagen_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (17, 76, 178), (30, 200, 255))

    sensitivity_1 = 68
    white = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_2 = cv2.inRange(HSL, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    bit_layer = yellow | white | white_2 | white_3
    return bit_layer


def draw_area(color_filtered):
    out_img = np.zeros_like(color_filtered)
    out_img = np.dstack((out_img, out_img, out_img))
    nwindows = 9
    margin = 100
    minpix = 50

    # Histograma

    histogram = np.sum(color_filtered[int(color_filtered.shape[0] / 2):, :], axis=0)
    '''
    plt.plot(histogram)
    #plt.savefig('../CarND-Advanced-Lane-Lines/histogram1.jpg')
    plt.show()
    '''

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
        # Identify window boundaries in x and y (and right and left)
        win_y_low = color_filtered.shape[0] - (window + 1) * window_height
        win_y_high = color_filtered.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each

    if len(leftx) == 0:
        left_fit = [.000246, -0.37811, 0.0376]
        # print('left fit',left_fit)
    else:
        # print('lefty',lefty)
        # print('leftx',leftx)
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit = [.000246, -0.37811, 0.0376]
        # print('left fit',left_fit)
    else:
        # print('lefty',lefty)
        # print('leftx',leftx)
        right_fit = np.polyfit(righty, rightx, 2)
    # print(right_fit)
    # Ventanas

    ploty = np.linspace(start=0, stop=color_filtered.shape[0] - 1, endpoint=color_filtered.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # plot images
    '''
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    #plt.savefig('../CarND-Advanced-Lane-Lines/windows1.jpg')
    plt.show()
    '''
    # nonzero
    '''
    nonzero = color_filtered.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    '''
    margin = 100
    # print('LLI',left_lane_inds)
    '''
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    '''

    '''
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, color_filtered.shape[0]-1, color_filtered.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((color_filtered, color_filtered, color_filtered))*255
    '''
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.savefig('../CarND-Advanced-Lane-Lines/track.jpg')
    # plt.show()


    ### Determine Curvature and position with respect to center
    '''
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
                                  for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    '''

    # Plot up the fake data
    '''
    mark_size = 3
    plt.plot(left_fitx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(right_fitx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    '''
    # plt.gca().invert_yaxis() # to visualize as we do the images
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    # left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    # right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    # left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    # right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # Example values: 632.1 m    626.2 m

    ### Warp Back to original Image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(color_filtered).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    Minv = inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image

    # cv2.imshow('image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return newwarp


# 1.- Distortion Correction
#  variables
nx = 9
ny = 5
imgPath = '../CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg'
pfilePath = '../CarND-Advanced-Lane-Lines/camera_cal/wide_dist_pickle.p'
# image = cv2.imread('../CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg')
image = cv2.imread('../CarND-Advanced-Lane-Lines/test_images/undist1.jpg')

# 2.- Call class to correct camera distortion
camera = DistortionCorrection()
camera.setvars(nx, ny, imgPath, pfilePath, image)

# 3.- Save Image Matrix,  and Distortion Matrix to pickle file
# Needs to be run only once to calculate the distortion correction matrix
# camera.savepick()

# 4.- Load saved values from pickle file
distp = camera.loadpick()

# 5.- Apply undistorted transformation to image
undistort_image = cv2.undistort(image, distp["mtx"], distp["dist"], None, distp["mtx"])

'''
### Imagen de Muestra para correccion de distorsion
cv2.imshow('Imagen con Distorsion',image)
cv2.waitKey(0)
cv2.imshow('Imagen sin Distorsion',undistort_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
####
'''
### Perspective Transform
source_points = np.float32([[578, 460], [264, 670], [1043, 670], [704, 460]])
destination_points = np.float32([[250, 0], [250, 720], [1000, 720], [1000, 0]])

perspTrans = PerspectiveTransform()
perspTrans.setSourcePoints(source_points, destination_points)
warped, M = perspTrans.warpedTrans(undistort_image)
'''
### Imagen de Muestra para vista de pajaro
cv2.imshow('Imagen original',image)
cv2.waitKey(0)
cv2.imshow('Imagen Transformada',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
####
'''

### Color Transform
color_filtered = color_filter(warped)

### Imagen de Muestra para vista de pajaro
cv2.imshow('Imagen original',warped)
cv2.waitKey(0)
cv2.imshow('Imagen Transformada',color_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

### Detect Lines Pixels
lane = draw_area(color_filtered)
cv2.imshow('Imagen Carril',lane)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Output visual display
'''


def procesar_imagen(image):
    undistort_image = cv2.undistort(image, distp["mtx"], distp["dist"], None, distp["mtx"])
    warped, M = perspTrans.warpedTrans(undistort_image)
    color_filtered = color_filter(warped)
    lane = draw_area(color_filtered)
    result = cv2.addWeighted(image, 1, lane, 0.3, 0)

    return undistort_image



asd = procesar_imagen(image)
cv2.imshow('Imagen Carril',asd)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Apply process image to video.
white_output = 'D:/aaSDCNDJ/CarND-Advanced-Lane-Lines/result_challenge.mp4'
clip1 = VideoFileClip("D:/aaSDCNDJ/CarND-Advanced-Lane-Lines/challenge_video.mp4")
white_clip = clip1.fl_image(procesar_imagen) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
'''