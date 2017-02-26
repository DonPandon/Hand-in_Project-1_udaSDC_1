from __future__ import division     # to force float division (version compatibility issue)
import numpy as np
import cv2
import math
from moviepy.editor import *

"""
Project 1 - Lane detector
Prepared by Daniel Garcia Gonzalez
"""
# ====================================================================================================
# ===============================================Variables============================================
# ==== For white and yellow videos
# """
kernel_size = 7
low_threshold = 20
high_threshold = 90
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 50                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 10            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1
_alphaValue = 0.8               # Default value: 0.8
_betaValue = 1.                 # leave decimal point at the end to convert to float. Default value: 1
_lambdaValue = 0.               # Default value: 0
#   Smooth lines' function
_threshold_slope = 0.5         # Crashes above 1. Above 0.5 the lines dissapaear more often. This threshold is used to remove  horizontal (and close to horizontal) lines.
_numberOfLinesToConsider = 8   # A larger value (and 0) increases time to process. This affects some videos, but the difference is hard to tell.
# """
# ==== For challenge video
"""
kernel_size = 7
low_threshold = 30
high_threshold = 70
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 5                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 15            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1
_alphaValue = 0.8               # Default value: 0.8
_betaValue = 1.                 # leave decimal point at the end to convert to float. Default value: 1
_lambdaValue = 0.               # Default value: 0
#   Smooth lines' function
_threshold_slope = 0.45         # Crashes above 1. Above 0.5 the lines dissapaear more often. This threshold is used to remove  horizontal (and close to horizontal) lines.
_numberOfLinesToConsider = 5   # A larger value (and 0) increases time to process. This affects some videos, but the difference is hard to tell.
"""
# == My videos (Thunder Bay)
"""
kernel_size = 5
low_threshold = 30
high_threshold = 120
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 50                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 10            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1
_alphaValue = 0.8               # Default value: 0.8
_betaValue = 1.                 # leave decimal point at the end to convert to float. Default value: 1
_lambdaValue = 0.               # Default value: 0

# Smooth lines function
_threshold_slope = 0.5         # Crashes above 1. Above 0.5 the lines dissapaear more often. This threshold is used to remove  horizontal (and close to horizontal) lines.
_numberOfLinesToConsider = 8   # A larger value (and 0) increases time to process. This affects some videos, but the difference is hard to tell.
"""
# == GENERAL VARIABLES
#   Lines: characteristics
_outputLineColors = [0, 153, 0]
_thicknessOfOutputLines = 8

# =========================points for masks depending on input videos===================
# ===================================point = [column, row]==============================
# ======================difference between points must be at least 200==================
#       white
_bottom_left = [60, 540]
_top_left = [447, 325]
_top_right = [513, 320]
_bottom_right = [960, 540]

#      yellow
# _bottom_left = [60, 540]
# _top_left = [450, 325]
# _top_right = [515, 325]
# _bottom_right = [960, 540]

#      tbay_training_3      NOTE: if the mask is too small, you loose points to generate the smooth lines (+- 200 span)
# _bottom_left = [60, 450]
# _top_left = [225, 225]
# _top_right = [400, 225]
# _bottom_right = [900, 450]

#       challenge
# _bottom_left = [400, 540]
# _top_left = [650, 425]
# _downwards_from_top_left = [650, 450]
# _downwards_to_bottom_left = [500, 530]
# _sideways_to_bottom_right = [750, 530]
# _upwards_to_top_right = [710, 450]
# _top_right = [710, 425]
# _bottom_right = [940, 540]

#      This mask works for most cases - Choose only 1
_mask_points = [np.array([_bottom_left, _top_left, _top_right, _bottom_right], dtype=np.int32)]       #solid yellow left
#      The following mask for the challenge video (still not working as good as expected)
# _mask_points = [np.array([_bottom_left, _top_left, _downwards_from_top_left, _downwards_to_bottom_left, _sideways_to_bottom_right, _upwards_to_top_right, _top_right, _bottom_right], dtype=np.int32)]       #solid yellow left

# ===========================================================================================
# ======================================My Functions=========================================
def canny_edge_det(_input_image):
    """generates Canny edges (includes grey, Blurr, and Canny)"""
    first_step_grey = cv2.cvtColor(_input_image, cv2.COLOR_BGR2GRAY)
    second_step_blur = cv2.GaussianBlur(first_step_grey, (kernel_size, kernel_size), 0)
    third_step_canny = cv2.Canny(second_step_blur, low_threshold, high_threshold)
    return third_step_canny


def hough_lines_only(img, rho, theta, threshold, min_line_len, max_line_gap):
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                         maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    #return lines      # modification to have access to both the lines and the merge
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)

# cv2 is not saving the video correctly, so I had to change my code to fit moviePy (and 99% of the samples online)
def fancy_lane_detector(_input_image):
    """function to generate smoooth lines ;)"""
    return generate_smooth_lanes(_input_image)  # smoooth...
    # return _input_image   # to check the basic feature still works (open and save a video)


def generate_smooth_lanes(_input_copy):
    """Based on the equation for a line:
                                             y = mx + b
       this function will split the data into segments, identify the mean values of all the points processed in Hough's,
       and obtain a mean final set of values that will be used to draw the left (negative slopes) and right (positive slopes)
       lines over the lanes. Most of the time.

       Because, in the end, we want just 2 lines, we need to collect all x and y points and find common y-intercepts
       slopes, magnitudes and such, using the input slopes (mean) as reference.
    """
    _edges = canny_edge_det(_input_copy)
    _masked_edges = region_of_interest(_edges, _mask_points)
    _embedded_image = hough_lines(_masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    _weighted_image = weighted_img(_embedded_image, _input_copy, _alphaValue)

    _hlines = hough_lines_only(_masked_edges, rho, theta, threshold, min_line_length, max_line_gap)         # the lines are needed for the calculations afterwards
    _hlines = np.squeeze(_hlines)  # corrects:: IndexError: index 1 is out of bounds for axis 1 with size 1

    # modified function to draw the smooth lines and write them right away on the output image
    _blank = draw_lines(_weighted_image, _hlines)

    # draw_lines(_blank, _output_lines)
    _output = weighted_img(_blank, _input_copy, _alphaValue)

    # == troubleshooting and end of function (leave only 1 uncommented; arranged in order of "relevance")
    # return _edges               # To return only the Canny Edges on video (VLC shows 9 screens at a time; perhaps the sqeeze did something_)
    # return _masked_edges        # Edges filtered by the mask (VLC shows 9 screens at a time)
    # return _blank               # Smooth lines over blank
    return _output              # final output, with smooth lines and original image in place


def draw_lines(_weighted_image, _hlines, color=_outputLineColors, thickness=_thicknessOfOutputLines):

    _x1_values, _y1_values = _hlines[:, 0], _hlines[:, 1]  # starts
    _x2_values, _y2_values = _hlines[:, 2], _hlines[:, 3]  # ends

    # m values from all lines
    _input_slopes = (_y2_values - _y1_values) / (_x2_values - _x1_values)
    # A values from all lines
    _input_magnitude = np.sqrt(np.square(_x2_values - _x1_values) + np.square(_y2_values - _y1_values))

    # There's some values that may be a straight line, giving me a nan in some of the values
    # Reducing the slope range it might do the trick
    _hlines = _hlines[np.abs(_input_slopes) > _threshold_slope]
    _input_magnitude = _input_magnitude[np.abs(_input_slopes) > _threshold_slope]
    _input_slopes = _input_slopes[np.abs(_input_slopes) > _threshold_slope]

    #   separate data
    _input_positive_slopes = _input_slopes[_input_slopes > 0]
    _input_negative_slopes = _input_slopes[_input_slopes < 0]
    _input_positive_lines = _hlines[_input_slopes > 0, :]
    _input_negative_lines = _hlines[_input_slopes < 0, :]

    # to be used by the y-intercept
    _input_positive_x_values = np.concatenate([_input_positive_lines[:, 0], _input_positive_lines[:, 2]])
    _input_negative_x_values = np.concatenate([_input_negative_lines[:, 0], _input_negative_lines[:, 2]])
    _input_positive_y_values = np.concatenate([_input_positive_lines[:, 1], _input_positive_lines[:, 3]])
    _input_negative_y_values = np.concatenate([_input_negative_lines[:, 1], _input_negative_lines[:, 3]])

    # sorting to choose faster a specific group; sort() didnt' work, sorted didn't work(), argsort works
    _input_positive_magnitude = np.argsort(_input_magnitude[_input_slopes > 0])
    _input_negative_magnitude = np.argsort(_input_magnitude[_input_slopes < 0])

    #   collect the last elements of the magnitudes and y-intercepts, and collect the averages
    # suggested by another peer; but it doesn't make too much of a difference with these videos, to my perspective.
    _input_positive_slopes = np.mean(_input_positive_slopes[_input_positive_magnitude][-_numberOfLinesToConsider::])
    _input_negative_slopes = np.mean(_input_negative_slopes[_input_negative_magnitude][-_numberOfLinesToConsider::])

    _input_positive_yIntercept = _input_positive_y_values - _input_positive_slopes * _input_positive_x_values
    _input_positive_yIntercept = np.mean(
        _input_positive_yIntercept[_input_positive_magnitude][-_numberOfLinesToConsider::])

    _input_negative_yIntercept = _input_negative_y_values - _input_negative_slopes * _input_negative_x_values
    _input_negative_yIntercept = np.mean(
        _input_negative_yIntercept[_input_negative_magnitude][-_numberOfLinesToConsider::])

    #    Having all the information required, time to generate the X and Y values for the mean lines by using
    #    again y = mx + b. Values to use suggested by peer
    # right lane
    _output_positive_x1 = (_weighted_image.shape[0] - _input_positive_yIntercept) /\
                          _input_positive_slopes  # top
    _output_positive_y1 = _weighted_image.shape[0]
    _output_positive_x2 = (_weighted_image.shape[0] / 1.5 - _input_positive_yIntercept) /\
                          _input_positive_slopes  # bottom
    _output_positive_y2 = _weighted_image.shape[0] / 1.5
    # left lane
    _output_negative_x1 = (_weighted_image.shape[0] - _input_negative_yIntercept) /\
                          _input_negative_slopes  # top
    _output_negative_y1 = _weighted_image.shape[0]
    _output_negative_x2 = (_weighted_image.shape[0] / 1.5 - _input_negative_yIntercept) /\
                          _input_negative_slopes  # bottom
    _output_negative_y2 = _weighted_image.shape[0] / 1.5

    #    Pack the lines to an array, ready to be drawn over the video
    _blank = np.zeros_like(_weighted_image)
    try:
        """In some videos (my videos) the array has issues storing not valid results;
        that's why I am using this try-except"""
        _output_lines = np.array([[[_output_positive_x1, _output_positive_y1, _output_positive_x2, _output_positive_y2],
                                   [_output_negative_x1, _output_negative_y1, _output_negative_x2, _output_negative_y2]]],
                                 dtype=np.int32)
        for line in _output_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(_blank, (x1, y1), (x2, y2), color, thickness)
    except:
        pass  # possibly if I store the previous values somewhere, I can just use those "good values" to avoid blank frames.

    # _previous_lines = _output_lines

    return _blank

# ===========================================================================================
# =========================================Project Functions=================================


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

"""
# Substituded by my version
def draw_lines(img, lines, color=_outputLineColors, thickness=_thicknessOfOutputLines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
"""

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)                                                   # modification to counter the modified function
    return line_img


def weighted_img(img, initial_img, _a=0.8, _b=1., _l=0.):
    return cv2.addWeighted(initial_img, _a, img, _b, _l)

# =======================================================================================
# ======================================Execution========================================

_input_video = VideoFileClip('solidWhiteRight.mp4')                 # A
# _input_video = VideoFileClip('solidYellowLeft.mp4')                 # B
# _input_video = VideoFileClip('challenge.mp4')                       # C

_output_video = _input_video.fl_image(fancy_lane_detector)      # editing

_output_video.write_videofile('white.mp4', audio=False)             # A
# _output_video.write_videofile('yellow.mp4', audio=False)            # B
# _output_video.write_videofile('challenge_output.mp4', audio=False)  # C