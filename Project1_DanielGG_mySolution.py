import numpy as np
import cv2

# ===================================input and output====================================
#_input_video = cv2.VideoCapture('solidWhiteRight.mp4')
#_input_video = cv2.VideoCapture('solidYellowLeft.mp4')
_input_video = cv2.VideoCapture('challenge.mp4')
# _codec = cv2.VideoWriter_fourcc(*'XVID')
# _output_video = cv2.VideoWriter('output.avi', _codec, 20.0, (640,480))

# ======================================variables========================================
kernel_size = 7
low_threshold = 20
high_threshold = 90
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 50                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 10            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1
_counter = 0
#               |bot_l|  |top_l |  |top_r  | |bot_r |
#_mask_points = [60, 540, 447, 325, 513, 320, 960, 540]      #solid white right
_mask_points = [78, 540, 447, 324, 510, 324, 960, 540]       #solid yellow left

# ======================================mis funciones========================================
def canny_edge_det(_img_):                                                                       #generates the final edges (includes grey, Blurr, and Canny)
    first_step_grey = cv2.cvtColor(_img_, cv2.COLOR_BGR2GRAY)
    second_step_blur = cv2.GaussianBlur(first_step_grey, (kernel_size, kernel_size), 0)
    third_step_canny = cv2.Canny(second_step_blur, low_threshold, high_threshold)
    return third_step_canny

def generate_ROI(canny, x, y, poly_points):                                                     # generates region of interest, returns image
    _mask = np.zeros_like(canny)
    #if len(canny.shape) > 2:
    #    channel_count = canny.shape[2]                                                          # i.e. 3 or 4 depending on your image
     #   ignore_mask_color = (255,) * channel_count
    #else:
    ignore_mask_color = 255
    _mask_vertices = np.array(
        [[(poly_points[0], poly_points[1]), (poly_points[2], poly_points[3]), (poly_points[4], poly_points[5]), (poly_points[6], poly_points[7])]],
        dtype=np.int32)
    cv2.fillPoly(_mask, _mask_vertices, ignore_mask_color)
    _masked_edges = cv2.bitwise_and(edges, _mask)
    return _masked_edges

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, _a=0.5, _b=1., _l=0.):
    return cv2.addWeighted(initial_img, _a, img, _b, _l)

# ======================================ejecucion========================================
while _input_video.isOpened():
    successful_frame, original = _input_video.read()
    if successful_frame == True:    # si es que se pudo leer el video....

        # copia frame original y empalma lineas (mas adelante)
        _original_copy = np.copy(original)

        edges = canny_edge_det(_original_copy)

        _masked_edges = generate_ROI(edges, 540, 960, _mask_points)
        # mask = np.zeros_like(edges)
        # ignore_mask_color = 255
        # vertices = np.array([[(60, 540), (450, 315), (495, 315), (910, 540)]],
        #                     dtype=np.int32)
        # cv2.fillPoly(mask, vertices, ignore_mask_color)
        # _masked_edges = cv2.bitwise_and(edges, mask)

        _blank = np.copy(original) * 0  # creates a black canvas
        #===============the linear regresion to remove noise should be here===============

        _lines = cv2.HoughLinesP(_masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        #===========================before drawing the lines==============================
        for line in _lines:                                                         # Iterate over the output "lines" and draw lines on a blank image
            for x1, y1, x2, y2 in line:
                cv2.line(_blank, (x1, y1), (x2, y2), (95, 206, 31), 5)

        color_edges = np.dstack((edges, edges, edges))

        lines_edges = cv2.addWeighted(_original_copy, 0.8, _blank, 1, 0)            # "fuses" the images together

        #cv2.imshow("testing outputs", edges)           #test edges (after canny)
        #cv2.imshow("testing outputs", _masked_edges)   #test mask (to check borders)
        cv2.imshow("testing outputs", lines_edges)      #final result (crossing fingers)
        #cv2.imwrite("Output/frame%d.jpg" % _counter, lines_edges)

        _counter = _counter + 1
        if cv2.waitKey(25) & 0xFF == ord('q') & _counter >= 300:                   # tiempo entre frames y tecla para terminar programa (esq y taches no funcionan)
            break
    else:
        break

# cerrar streams (asumo hay que cerrar output primero)

_input_video.release()
# _output_video.release()