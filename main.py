
import cv2
import numpy as np
import math
import center_curvature

print('Program initiating.');
print('OpenCV-Python version: %s' % cv2.__version__)


# Method below adapted from https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/
def rgb_to_hsv(input):

    (r, g, b) = (input[0], input[1], input[2])
 
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
 
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
 
    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
     
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
 
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
 
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
 
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
 
    # compute v
    v = cmax * 100
    return np.array([h/2, s * 255 / 100, v * 255 / 100])

# Method below adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def hex_to_hsv(hex):
    return rgb_to_hsv(hex_to_rgb(hex))

def read_image(path):
    print('Loading image at %s' % path)
    return cv2.imread(path, cv2.IMREAD_COLOR)

def combine_masks(*args):
    start_mask = args[0]
    for i in range(1, len(args)):
        start_mask = cv2.bitwise_or(args[i], start_mask)
    return start_mask

# Takes in an image and returns a mask which contains all grass-like regions of the image, by color
# Image must already be in HSV color space
def locate_grass(img):
    # Grass thresholds in HSV space
    # We use HSV space for color boundaries because it's better at identifying ranges of colors
    print('Locating grass')
    lower_grass = (30, 20, 20)
    upper_grass = (75, 255,255)
    return cv2.inRange(img, lower_grass, upper_grass)

# func should take the pixel value and location as arguments
def custom_mask(img, func):
    (w, h, _) = img.shape
    mask = np.zeros((w, h), dtype=np.uint8)
    for r in range(w):
        for c in range(h):
            pix = img[r][c]
            include = func(pix, (r, c))
            if include:
                mask[r][c] = 255
    return mask

def soccer_ball_mask_func(pix, _):
    (_, s, v) = pix
    """if v < 20:
        return True
    ex_val = min(v, 255 - v)
    inclusion_score = (s + ex_val*0.5) / (255 + 255*0.5)
    return inclusion_score < 0.4"""
    return v < 20 or s < 80


def locate_soccer_ball(img, background=None):
    # When finding white colors, we replace the background with a black color so it isn't selected
    """for_white = img
    if background:
        for_white = remove_color(img, background[0], background[1], replace=(0, 0, 0))
    lower_white = (0, 0, 120)
    upper_white = (180, 30, 255)
    white_mask = cv2.inRange(for_white, lower_white, upper_white)
    print(white_mask.shape)
    # Similarly, when finding black colors, we replace the background with white
    for_black = img
    if background:
        for_black = remove_color(img, background[0], background[1], replace=(0, 0, 255))
    lower_black = (0, 0, 0)
    upper_black = (180, 180, 120)
    black_mask = cv2.inRange(for_black, lower_black, upper_black)
    cv2.imshow('White mask', white_mask)
    cv2.imshow('Black mask', black_mask)
    combined = combine_masks(white_mask, black_mask)
    return combined"""
    return custom_mask(img, soccer_ball_mask_func)

def remove_color(img, lower, upper, replace=(0, 0, 0)):
    removed = img.copy()
    mask = cv2.inRange(removed, lower, upper)
    mask = invert_mask(mask)
    removed[np.where(mask==[0])] = replace
    return removed

def image_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def invert_mask(mask):
    return cv2.bitwise_not(mask)

# Input image must be grayscale
def find_circles(img, dp=1, minDist=1, param1 = 50, param2 = 30, minRadius=1, maxRadius=400):
    # Blur image so circles are easier to identify
    blurred = cv2.blur(img, (7, 7), 0)
    cv2.imshow('Blurred', blurred)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist) #param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is None:
        return np.array([])
    print('Has circles or some shit?')
    # Convert parameters to integers
    res = np.uint16(np.around(circles))
    return res

# Method below adapted from https://stackoverflow.com/questions/43841210/how-to-detect-circle-in-a-binary-image
def find_contours(img):
    blurred = cv2.blur(img, (7, 7), 0)
    cv2.imshow('Actual blurred', blurred)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)

    contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours

    contour_list = []
    i = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        # Filter based on length and area
        #if (7 < len(approx) < 18) & (900 >area > 200):
        if True:
            # print area
            contour_list.append(contour)
    return contours
    #return contour_list

def largest_circle(circles):
    if len(circles) == 0:
        print('Cannot find largest circle in empty array.')
        return -1
    largest_index = 0
    largest_rad = circles[0]
    for i in range(1, len(circles)):
        rad = circles[i][2]
        if rad > largest_rad:
            largest_index = i
            largest_rad = rad
    return largest_index

def largest_contour(cons):
    if len(cons) == 0:
        print('Cannot find largest contour in empty array.')
        return -1
    largest_index = 0
    largest_area = cv2.contourArea(cons[0])
    for i in range(1, len(cons)):
        area = cv2.contourArea(cons[i])
        if area > largest_area:
            largest_index = i
            largest_area = area
    return largest_index

def most_circular_contour(cons, min_area=0, max_area=1000000000):
    if len(cons) == 0:
        print('Cannot find most circular contour in empty array.')
        return -1
    best_index = -1
    best_aspect = 0
    for (i, con) in enumerate(cons):
        if len(con) == 1:
            continue
        area = cv2.contourArea(con)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(con)
        aspect_ratio = float(w)/h
        if aspect_ratio < 1:
            aspect_ratio = 1/aspect_ratio
        if best_index == -1 or aspect_ratio < best_aspect:
            best_index = i
            best_aspect = aspect_ratio
    return best_index

# Finds magnitude of a vector in two-dimensional coordinate space
def mag(x, y):
    return math.sqrt(x*x + y*y)

def contour_center(con):
    mom = cv2.moments(con)
    center_x = int(mom["m10"] / mom["m00"])
    center_y = int(mom["m01"] / mom["m00"])
    return (center_x, center_y)

def contour_avg_dist(con, center):
    sum = 0
    num = 0
    for point in con:
        point = point[0]
        x = point[0]
        y = point[1]
        dist = mag(x - center[0], y - center[1])
        sum += dist
        num += 1
    return sum / num

def approx_contour_as_circle(con):
    center = contour_center(con)
    avg_rad = contour_avg_dist(con, center)
    return [center[0], center[1], avg_rad]

def greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_sv(hsv):
    mean = cv2.mean(hsv)
    (w, h, _) = hsv.shape
    s_mult = 128 / mean[1]
    v_mult = 128 / mean[2]
    for r in range(w):
        for c in range(h):
            pix = hsv[r][c]
            pix[1] = int(pix[1] * s_mult)
            pix[2] = int(pix[2] * v_mult)

img = read_image('sample2.jpg')

# We convert to HSV to locate the grass, but don't use it other than that right now
hsv = image_to_hsv(img)
#normalize_sv(hsv)
#mask = locate_grass(hsv)
mask = locate_soccer_ball(hsv)
#inv_mask = invert_mask(mask)
inv_mask = mask
applied = apply_mask(img, inv_mask)

cv2.imshow('Original', img)
#cv2.imshow('Masked', applied)
cv2.imshow('Mask', mask)

contours = find_contours(inv_mask)
print('Contour')
img_with_contours = img.copy()
cv2.drawContours(img_with_contours, contours,  -1, (255,0,0), 2)
cv2.imshow('Contours', img_with_contours)
largest_ind = largest_contour(contours)
#largest_ind = -1
#largest_ind = 0
largest_ind = most_circular_contour(contours)
if largest_ind == -1:
    print('Done.')
else:
    largest = contours[largest_ind]

    curv = center_curvature.approx_center_curvature(img, largest)
    curv = center_curvature.tup_float_to_int(curv)
    with_points = img.copy()
    center_curvature.draw_points(with_points, [curv], radius=5)
    cv2.imshow('Curvature', with_points)

    """img_cp = img.copy()
    print('Best contour area: %f' % cv2.contourArea(largest))
    cv2.drawContours(img_cp, [largest], -1, (255,0,0), 2)
    cv2.imshow('Largest contour', img_cp)"""

    (x, y, r) = approx_contour_as_circle(largest)

    print('Contour diameter: %f' % (2.0*r))

    r -= 5
    r = int(round(r))

    r = int(contour_avg_dist(largest, curv)) - 5

    img_with_circle = img.copy()
    #cv2.circle(img_with_circle, (x, y), r, (0, 255, 0), 2)
    cv2.circle(img_with_circle, curv, r, (0, 255, 0), 2)
    cv2.imshow('Final', img_with_circle)

hough_circles = find_circles(inv_mask, dp=1, minDist=1, minRadius=0, maxRadius=100000000000)
print(hough_circles)

cv2.waitKey(0)
cv2.destroyAllWindows()
