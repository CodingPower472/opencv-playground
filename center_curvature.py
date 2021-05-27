
import cv2
import numpy as np
import csv
from sklearn.cluster import AffinityPropagation

def read_image(path):
    print('Loading image at %s' % path)
    return cv2.imread(path, cv2.IMREAD_COLOR)

# img must be in RGB
def mask_black(img):
    return cv2.inRange(img, (0, 0, 0), (0, 0, 0))

# takes in a mask
def find_contours(img):
    blurred = cv2.blur(img, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)

    contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Gives (m, b) in the equation y = mx + b of the perpendicular bisector of the line connecting these two points
# If line is vertical, will instead give (None, x) where x is the x-coordinate of the line
def perp_bis(p1, p2):
    mid = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    perp_slope = None
    y_int = p1[0]
    if delta_y != 0:
        perp_slope = -delta_x / delta_y
        y_int = mid[1] - perp_slope*mid[0]
    return (perp_slope, y_int)

def plug(line, x):
    if line[0] == None:
        return None
    return line[0]*x + line[1]

def find_intersection(l1, l2):
    if l2[0] == None:
        if l1[0] == None:
            return None
        a = l1
        l1 = l2
        l2 = a
    if l1[0] == None:
        return (l1[1], plug(l2, l1[1]))
    if l1[0] == l2[0]:
        return None
    """
    b2 + x*m2 = b1 + x*m1
    x*m1 - x*m2 = b2 - b1
    x(m1 - m2) = b2 - b1
    x = (b2 - b1) / (m1 - m2)
    """
    x = (l2[1] - l1[1]) / (l1[0] - l2[0])
    y = plug(l1, x)
    return (x, y)

def contour_perp_bis(contour, dens=5, gap=15):
    res = []
    for i in range(0, len(contour) - gap, dens):
        p1 = contour[i][0]
        p2 = contour[i + gap][0]
        perp = perp_bis(p1, p2)
        res.append(perp)
    return res

"""def intersection_points(lines, gap=5):
    res = []
    for i in range(0, len(lines) - gap):
        l1 = lines[i]
        l2 = lines[i + gap]
        inter = find_intersection(l1, l2)
        res.append(inter)
    return res"""

def intersection_points(lines, dens=5):
    res = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines), dens):
            l1 = lines[i]
            l2 = lines[j]
            inter = find_intersection(l1, l2)
            res.append(inter)
    return res

def tup(arr):
    return (arr[0], arr[1])

def tup_to_arr(tup):
    return [tup[0], tup[1]]

def tup_float_to_int(tup):
    return (int(tup[0]), int(tup[1]))

def draw_lines(img, lines, width):
    a = img
    for line in lines:
        if line[0] == None:
            #print('Found vertical line.  Skipping.')
            continue
        start = (0, int(line[1]))
        end = (width, int(line[0]*width + line[1]))
        a = cv2.line(a, start, end, (0,255,0), 2)
    return a

def draw_points(img, points, radius=2, color=(0,0,255), is_tup=True):
    print('Draw points')
    a = img
    for point in points:
        if point is None:
            continue
        if not is_tup:
            point = tup(point)
        a = cv2.circle(a, tup_float_to_int(point), radius, color, -1)
    return a

def find_largest_cluster(predict, size):
    if size == 0:
        return 0
    arr = np.zeros(size, np.int32)
    for i in predict:
        arr[i] += 1
    largest = arr[0]
    ind = 0
    for (i, val) in enumerate(arr):
        if val > largest:
            ind = i
            largest = val
    print(arr)
    print(ind)
    return ind

def find_point_clusters(points):
    kmeans = AffinityPropagation(damping=0.9)
    arr = []
    for i in range(len(points)):
        if points[i] == None:
            continue
        arr.append(tup_to_arr(points[i]))
    data = np.array(arr)
    print('About to fit')
    kmeans.fit(data)
    print('Done fitting')
    return (kmeans.cluster_centers_, kmeans.fit_predict(data))

# img should be a black/white mask
def approx_center_curvature(img, contour, gap=15, dens=5):
    perp = contour_perp_bis(contour, gap=gap, dens=dens)
    print(len(perp))
    points = intersection_points(perp, dens=dens)
    print(len(points))
    with_lines = img.copy()
    with_lines = draw_points(with_lines, points)
    cv2.imshow('With lines', with_lines)
    print('Step 1')
    clusters = find_point_clusters(points)
    print('Step 2')
    largest_cluster_ind = find_largest_cluster(clusters[1], len(clusters[0]))
    largest_cluster = clusters[0][largest_cluster_ind]
    return tup(largest_cluster)

"""image = read_image('arc.png')
mask = mask_black(image)
contours = find_contours(mask)

with_contours = image.copy()
cv2.drawContours(with_contours, contours, -1, (255,0,0), 2)
cv2.imshow('With contours', with_contours)

perp = contour_perp_bis(contours[0])
points = intersection_points(perp, gap=30)

with_lines = draw_lines(with_contours, perp, with_contours.shape[0])
with_lines = draw_points(with_lines, points)
clusters = find_point_clusters(points)
largest_cluster_ind = find_largest_cluster(clusters[1], len(clusters[0]))
largest_cluster = clusters[0][largest_cluster_ind]
print(largest_cluster)
with_lines = draw_points(with_lines, [largest_cluster], radius=10, color=(0,255,255), is_tup=False)
cv2.imshow('With lines', with_lines)



with open('test.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for point in points:
        row = tup_to_arr(point)
        writer.writerow(row)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""