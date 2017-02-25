from __future__ import division     # to force float division (version compatibility issue)
import numpy as np
import cv2
import math
from moviepy.editor import *


# ======================================variables========================================
# =======for project videos

kernel_size = 7
low_threshold = 20
high_threshold = 90
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 50                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 10            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1
_counter = 0
_output_lines = []
_prevous_lines = []

#========my videos
"""
kernel_size = 5
low_threshold = 30
high_threshold = 120
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 50                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 10            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1
_counter = 0
_output_lines = []
_prevous_lines = []
"""
# =========================points for masks depending on input videos===================
# ===================================point = [column, row]==============================
# ======================difference between points must be at least 200==================
#       white
_bottom_left = [60, 540]
_top_left = [447, 325]
_top_right = [513, 320]
_bottom_right = [960, 540]

#      yellow
#_bottom_left = [60, 540]
#_top_left = [450, 325]
#_top_right = [515, 325]
#_bottom_right = [960, 540]

#      tbay_training_3
# _bottom_left = [60, 450]        # if the mask is too small, you loose points to generate the smooth lines
# _top_left = [225, 225]
# _top_right = [400, 225]
# _bottom_right = [900, 450]

#       challenge
# _bottom_left = [400, 540]
# _top_left = [650, 425]
# _downwards_from_top_left = [650, 450]
# _downwards_to_bottom_left = [500, 510]
# _sideways_to_bottom_right = [750, 510]
# _upwards_to_top_right = [710, 450]
# _top_right = [710, 425]
# _bottom_right = [960, 540]

_mask_points = [np.array([_bottom_left, _top_left, _top_right, _bottom_right], dtype=np.int32)]       #solid yellow left
# _mask_points = [np.array([_bottom_left, _top_left, _downwards_from_top_left, _downwards_to_bottom_left, _sideways_to_bottom_right, _upwards_to_top_right, _top_right, _bottom_right], dtype=np.int32)]       #solid yellow left

# ======================================my functions========================================
def canny_edge_det(_input_image):                                                                       #generates the final edges (includes grey, Blurr, and Canny)
    first_step_grey = cv2.cvtColor(_input_image, cv2.COLOR_BGR2GRAY)
    second_step_blur = cv2.GaussianBlur(first_step_grey, (kernel_size, kernel_size), 0)
    third_step_canny = cv2.Canny(second_step_blur, low_threshold, high_threshold)
    return third_step_canny


# cv2 is not saving the video correctly, so I had to change my code to fit moviePy (and 99% of the samples online)
def fancy_lane_detector(_input_image):
    # return generate_lines(_input_image)
    return generate_smooth_lanes(_input_image)
    # return _input_image   # to check the basic feature still works (open and save a video)


def generate_lines(_input_copy):
    # filtering according to what was covered: canny edge, ROI, and HoughLines
    _edges = canny_edge_det(_input_copy)
    _masked_edges = region_of_interest(_edges, _mask_points)
    _embedded_image, _hlines = hough_lines(_masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    _output_image = weighted_img(_embedded_image, _input_copy, _a=0.8)
    return _output_image


def generate_smooth_lanes(_input_copy):
    _edges = canny_edge_det(_input_copy)
    _masked_edges = region_of_interest(_edges, _mask_points)
    _embedded_image, _hlines = hough_lines(_masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    _weighted_image = weighted_img(_embedded_image, _input_copy, _a=0.8)

    _hlines = np.squeeze(_hlines)       # corrects:: IndexError: index 1 is out of bounds for axis 1 with size 1

    _x1_values, _y1_values = _hlines[:, 0], _hlines[:, 1]     # starts
    _x2_values, _y2_values = _hlines[:, 2], _hlines[:, 3]     # ends

    _input_slopes = (_y2_values - _y1_values) / (_x2_values - _x1_values)     # m values from all lines
    _input_magnitude = np.sqrt(np.square(_x2_values - _x1_values) + np.square(_y2_values - _y1_values))   # A values from all lines

    # There's some values that may be a straight line, giving me a nan in some of the values
    # Reducing the slope range it might do the trick
    _threshold_slope = 0.5
    _hlines = _hlines[np.abs(_input_slopes)>_threshold_slope]
    _input_magnitude = _input_magnitude[np.abs(_input_slopes)>_threshold_slope]
    _input_slopes = _input_slopes[np.abs(_input_slopes)>_threshold_slope]
    """
    Because, in the end, we want just 2 lines, we need to collect all x and y points and find common y-intercepts
    slopes, magnitudes and such. Using the slopes as reference, separate lines from left ( < 0 ) and right ( > 0 )
    """
    #   separate data
    _input_positive_slopes = _input_slopes[_input_slopes > 0]
    _input_negative_slopes = _input_slopes[_input_slopes < 0]
    _input_positive_lines = _hlines[_input_slopes > 0, :]
    _input_negative_lines = _hlines[_input_slopes < 0, :]

    _input_positive_x_values = np.concatenate([_input_positive_lines[:, 0], _input_positive_lines[:, 2]])   # to be used by the y-intercept
    _input_negative_x_values = np.concatenate([_input_negative_lines[:, 0], _input_negative_lines[:, 2]])
    _input_positive_y_values = np.concatenate([_input_positive_lines[:, 1], _input_positive_lines[:, 3]])
    _input_negative_y_values = np.concatenate([_input_negative_lines[:, 1], _input_negative_lines[:, 3]])

    _input_positive_magnitude = np.argsort(_input_magnitude[_input_slopes > 0])    #sorting for easier tracking
    _input_negative_magnitude = np.argsort(_input_magnitude[_input_slopes < 0])     #sort didnt' work, sorted didn't work, argsort does

    #   collect the last 10 elements of the magnitudes and y-intercepts, and collect the averages
    _input_positive_slopes = np.mean(_input_positive_slopes[_input_positive_magnitude][-10::])
    _input_negative_slopes = np.mean(_input_negative_slopes[_input_negative_magnitude][-10::])

    _input_positive_yIntercept = _input_positive_y_values - _input_positive_slopes * _input_positive_x_values
    _input_positive_yIntercept = np.mean(_input_positive_yIntercept[_input_positive_magnitude][-10::])

    _input_negative_yIntercept = _input_negative_y_values - _input_negative_slopes * _input_negative_x_values
    _input_negative_yIntercept = np.mean(_input_negative_yIntercept[_input_negative_magnitude][-10::])

    #    Having all the information required, time to generate the X and Y values for the mean lines
    _output_positive_x1 = (_weighted_image.shape[0] - _input_positive_yIntercept) / _input_positive_slopes
    _output_positive_y1 = _weighted_image.shape[0]
    _output_positive_x2 = (_weighted_image.shape[0]/1.5 - _input_positive_yIntercept) / _input_positive_slopes
    _output_positive_y2 = _weighted_image.shape[0]/1.5

    _output_negative_x1 = (_weighted_image.shape[0] - _input_negative_yIntercept) / _input_negative_slopes
    _output_negative_y1 = _weighted_image.shape[0]
    _output_negative_x2 = (_weighted_image.shape[0] / 1.5 - _input_negative_yIntercept) / _input_negative_slopes
    _output_negative_y2 = _weighted_image.shape[0] / 1.5

    #    Pack the lines to an array, ready to be drawn over the video

    _blank = np.zeros_like(_input_copy)
    try:
        _output_lines = np.array([[[_output_positive_x1,_output_positive_y1,_output_positive_x2,_output_positive_y2],[_output_negative_x1,_output_negative_y1,_output_negative_x2,_output_negative_y2]]], dtype=np.int32)
        draw_lines(_blank, _output_lines)
    except:
        pass

    # draw_lines(_blank, _output_lines)
    _output = weighted_img(_blank, _input_copy, _a=0.8)

    # return _edges
    # return _masked_edges
    # return _embedded_image      # returns the Hough's transformation
    # return _weighted_image    # returns Houghs mixed with original input
    # return _blank
    return _output

# =========================================project functions================================
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 153, 0], thickness= 8):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines      # modification to have access to both the lines and the merge


def weighted_img(img, initial_img, _a=0.8, _b=1., _l=0.):
    return cv2.addWeighted(initial_img, _a, img, _b, _l)

# ======================================execution========================================

# switch to moviePy to work on the video
# _input_video = VideoFileClip('solidWhiteRight.mp4')                 # A
_input_video = VideoFileClip('solidYellowLeft.mp4')                 # B
# _input_video = VideoFileClip('challenge.mp4')

_output_video = _input_video.fl_image(fancy_lane_detector)

#_output_video.write_videofile('white.mp4', audio=False)              # A
_output_video.write_videofile('yellow.mp4', audio=False)            # B
# _output_video.write_videofile('challenge_output.mp4', audio=False)