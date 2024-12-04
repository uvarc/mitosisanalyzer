import cv2
import numpy as np
from math import atan2, degrees


def get_centers(contour):
    """Extract center x/y coordinates from a contour object"""
    M = cv2.moments(contour)
    if M["m00"] == 0.0:
        return (0, 0)
    else:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def get_rect_points(contour):
    """Get corner point of smallest rectangle enclosing a contour"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.intp(box)


def euclidian(edge=None, p1=None, p2=None):
    """Calculates the euclidian distance between two points"""
    if edge is not None:
        p1 = edge[0]
        p2 = edge[1]
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def oscillation(ref_p1, ref_p2, points, pixel_res=1.0):
    """Calculates the distance for each point in points to a reference line defined by ref_p1 and ref_p2"""
    print(
        f"Calculating oscillations with ref_points {ref_p1}, {ref_p2} and ref_axis={ref_p2-ref_p1}, points.shape={points.shape}"
    )
    ref_axis = ref_p2 - ref_p1
    osc = pixel_res * np.cross(ref_axis, points - ref_p1) / np.linalg.norm(ref_axis)
    return osc


def get_edges(corners, length_sort=True):
    """Creates list of edges defined by pairs of consecutive vertices. Optional: the edges
    may be sorted by length (ascending)"""
    edges = [(corners[i], corners[i + 1]) for i in range(len(corners) - 1)]
    edges.append((corners[len(corners) - 1], corners[0]))
    edges.sort(key=euclidian)
    return edges


def center(p1, p2):
    """Get the geometric center of two points"""
    cx = int(0.5 * (p1[0] + p2[0]))
    cy = int(0.5 * (p1[1] + p2[1]))
    return (cx, cy)


def square_edges(edges):
    sedges = [e for e in edges]
    scorners = []
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        p3 = (p2[0] + dy, p2[1] - dx)
        dx = p1[0] - p3[0]
        dy = p1[1] - p3[1]
        p4 = (p2[0] + dy, p2[1] - dx)
        sedges.append((p2, p3))
        sedges.append((p3, p4))
        sedges.append((p4, p1))
        points = np.array([p1, p2, p3, p4])
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        # region = img[xmin:xmax+1,ymin:ymax+1]
        # print (region.max())
        # centerx,centery = np.unravel_index(np.argmax(region, axis=None), region.shape)
        # centerx = int(0.5 * (xmin + xmax))
        # centery = int(0.5 * (ymin + ymax))
        # sedges.append((p1,(centerx,centery)))

        scorners.append([p1, p2, p3, p4])
    return sedges, scorners


def profile_endpoints(p1, p2, center, length):
    """Returns pair of coordinate tuples that define a line of length <length> that crosses through p1[0]/p1[1] and p2[0]/p2[1] and with center[0]/center[1] as its center)"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    l = np.sqrt(dx * dx + dy * dy)
    if l > 0.0:
        dx = dx / l
        dy = dy / l
        xoffset = int(dx * length * 0.5)
        yoffset = int(dy * length * 0.5)
        end1 = (center[0] + xoffset, center[1] + yoffset)
        end2 = (center[0] - xoffset, center[1] - yoffset)
        return (end1, end2)
    else:
        return (0, 256), (512, 256)


def get_angle(p1, p2):
    radians = atan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = degrees(radians)
    return angle


def get_row_angle(r):
    p1 = (r.iloc[0], r.iloc[1])
    p2 = (r.iloc[2], r.iloc[3])
    a = get_angle(p1, p2)
    if a < 0:
        a = a + 360
    return a


def get_row_euclidian(r, pixel_res=1.0):
    p1 = (r.iloc[0], r.iloc[1])
    p2 = (r.iloc[2], r.iloc[3])
    dist = pixel_res * euclidian(p1=p1, p2=p2)
    return dist


def intersect_line(line, p):
    deltax = line[0][0] - line[1][0]
    deltay = line[0][1] - line[1][1]
    if deltay != 0:
        orthom = -(deltax / deltay)
        line2 = np.array([p, p + orthom * np.array([1, 1])]).reshape((2, 2))
    else:
        line2 = np.array([p[0], p[1], 0 + p[0], 1 + p[0]]).reshape((2, 2))
    xdiff = (line[0][0] - line[1][0], line2[0][0] - line2[1][0])
    ydiff = (line[0][1] - line[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_orientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0],
    )
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0],
    )
    # drawAxis(img, cntr, p1, (127, 127, 0), 1)
    first_norm = np.array([p1[0] - cntr[0], p1[1] - cntr[1]])
    refline = (cntr + first_norm * 10, cntr - first_norm * 10)
    # drawAxis(img, refline[0], refline[1], (63, 63, 0), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    # textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    # cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    return angle, cntr, first_norm, refline
