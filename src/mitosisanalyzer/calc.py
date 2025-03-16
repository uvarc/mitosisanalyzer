import cv2
import numpy as np
import pandas as pd
from math import atan2, degrees
from scipy import fft
from scipy.signal.windows import blackmanharris
from scipy.signal import correlate
from typing import Optional


def get_centers(contour: np.ndarray) -> tuple[int, int]:
    """Extract center x/y coordinates from a contour object

    Parameters
        ----------
        contour: np.ndarray
            Array of points defining the contour.
    Returns
        ----------
        center: tuple
            The (x, y) coordinates of the contour's center
    """
    M = cv2.moments(contour)
    if M["m00"] == 0.0:
        center = (0, 0)
    else:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center


def get_rect_points(contour: np.ndarray) -> np.intp:
    """Get corner point of smallest rectangle enclosing a contour

    Parameters
        ----------
        contour: np.ndarray
            Array of points defining the contour.
    Returns
        ----------
        box: np.intp
            Array defining the four corners of the contour's bounding box.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.intp(box)


def euclidian(
    edge: list[tuple[int, int], tuple[int, int]] = None,
    p1: tuple[int, int] = None,
    p2: tuple[int, int] = None,
) -> float:
    """Calculates the euclidian distance between two points

    Parameters
        ----------
        edge: list
            Tuple of 2 points (x,y) defining the endpoints of a line.
            The p1 and p2 parameters are ignored when edge is not None. Default: None.
        p1: tuple
            (x,y) coordinates of the first point.
        p2: tuple
            (x,y) coordinates of the second point.
    Returns
        ----------
        dist: float
            The euclidian distance between the endpoints of the edge, or the p1 and p2 points.
    """
    if edge is not None:
        p1 = edge[0]
        p2 = edge[1]
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def extend_line(
    x1: int, y1: int, x2: int, y2: int, xlims: Optional[tuple[int, int]] = (0, 512)
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Provides the endpoints of a line extended thorugh two given points.

    Parameters
        ----------
        x1: int
            x coordinate of the first point.
        y1: int
            y coordinate of the first point.
        x2: int
            x coordinate of the second point.
        y2: int
            y coordinate of the second point.
        xlims: tuple, optional
            defines (min,max) values for the x coordinates of the extended line.
    Returns
        ----------
        eline: tuple
            tuple of endpoints of the extended line.
    """
    if x2 != x1:
        # extend lines to xlims
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x1, x2 = xlims
        y1 = slope * x1 + intercept
        y2 = slope * x2 + intercept
        eline = (x1, y1), (x2, y2)
    else:
        eline = (x1, xlims[0]), (x2, xlims[1])
    return eline


def closest_point(
    x: float,
    y: float,
    z: float,
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> tuple[float, float, ...]:
    """
    Finds the perpendicular vector from a point to a line segment in a plane z.

    Parameters
        ----------
        x: float
            The x coordinates of the point.
        y: float
            The y coordinates of the point.
        z: float
            The z coordinates of the point. May be set to None.
        line_start: tuple
            The coordinates of the start point of the line (x, y) or (x,y,z).
        line_end: tuple
            The coordinates of the end point of the line (x, y) or (x,y,z).

    Returns:
        c_point: tuple:
            The coordinates of the perpendicular vector (x, y) or (x,y,z).
    """

    # Convert points to NumPy arrays for easier calculations
    point = np.array([x, y])
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # Calculate the direction vector of the line
    line_vector = line_end - line_start

    # Calculate the vector from the line start point to the given point
    point_to_line_start = point - line_start

    # Project the point-to-line-start vector onto the line direction vector
    projection = np.dot(point_to_line_start, line_vector) / np.dot(
        line_vector, line_vector
    )

    # Calculate the closest point on the line to the given point
    c_point = line_start + projection * line_vector
    if z is None:
        return (c_point[1], c_point[0])  # yx
    else:
        return (z, c_point[1], c_point[0])  # zyx


def oscillation(
    ref_p1: tuple[float, float],
    ref_p2: tuple[float, float],
    points: np.ndarray,
    pixel_res: Optional[float] = 1.0,
) -> float:
    """Calculates the distance for each point in points to a reference line defined by ref_p1 and ref_p2

    Parameters
        ----------
        ref_p1: tuple
            First point (x,y) defining a reference line.
        ref_p2: tuple
            Second point (x,y) defining a reference line.
        points: np.ndarray
            Array of points representing the oscillating signal.
        pixel_res: float, optional
            The size of a pixel in physical units. Applied as scaling factor to the calculated osciallation amplitudes
    Returns
        ----------
        amp: np.ndarray
            Array of oscillation ammplitude for each point in points
    """
    print(
        f"Calculating oscillation amplitudes with ref_points {ref_p1}, {ref_p2} and ref_axis={ref_p2-ref_p1}, points.shape={points.shape}"
    )
    ref_axis = ref_p2 - ref_p1
    amp = pixel_res * np.cross(ref_axis, points - ref_p1) / np.linalg.norm(ref_axis)
    return amp


# def zero_crossings(signal):
#    zeroc = np.where(np.diff(np.sign(signal)))[0]
#    return zeroc


def zero_crossings(signal: np.ndarray, fs: Optional[float] = 1.0) -> float:
    """
    Estimate frequency by counting zero crossings

    Parameters
        ----------
        signal: np.ndarray
            Array of points representing the oscillating signal.
        fs: float, optional
            Frames per second
    Returns
        ----------
        crossings: float
            Frequency of corssings per second (Hz).
    """
    try:
        # Find all indices right before a rising-edge zero crossing
        indices = np.nonzero((signal[1:] >= 0) & (signal[:-1] < 0))[0]

        # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
        # crossings = indices

        # More accurate, using linear interpolation to find intersample
        # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
        crossings = [i - signal[i] / (signal[i + 1] - signal[i]) for i in indices]

        # Some other interpolation based on neighboring points might be better.
        # Spline, cubic, whatever
        diff_c = np.diff(crossings)
        if len(diff_c) == 0:
            return np.nan
        else:
            return fs / np.mean(diff_c)
    except:
        return np.nan


def dominant_freq(signal, fs: Optional[float] = 1.0) -> float:
    """
    Parameters
        ----------
        signal: np.ndarray
            Array of points representing the oscillating signal.
        fs: float, optional
            Frames per second
    Returns
        ----------
        dom_freq: float
            Frequency of oscillation per second (Hz).
    """
    try:
        spectrum = fft.fft(signal)
        freq = fft.fftfreq(len(signal), fs)
        dom_freq = freq[np.argmax(np.abs(spectrum))]
        return dom_freq
    except:
        return np.nan


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    Parameters
        ----------
        f: np.ndarray
            A vector
        x: int
            An index for the vector f.

    Returns
        ----------
        (vx, vy) tuple
            The coordinates of the vertex of a parabola that goes through
            point x and its two neighbors.

    Example
        ----------
        Defining a vector f with a local maximum at index 3 (= 6), find local
        maximum if points 2, 3, and 4 actually defined a parabola.

        In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

        In [4]: parabolic(f, argmax(f))
        Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1 / 2 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def fft_freq(signal: np.ndarray, fs: Optional[float] = 1.0):
    """
    Estimate frequency from peak of FFT

    Parameters
        ----------
        signal: np.ndarray
            Array of points representing the oscillating signal.
        fs: float, optional
            Frames per second
    Returns
        ----------
        freq: float
            Dominant frequency of oscillation per second (Hz).
    """
    try:
        # Compute Fourier transform of windowed signal
        windowed = signal * blackmanharris(len(signal))
        f = np.fft.rfft(windowed)

        # Find the peak and interpolate to get a more accurate peak
        i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
        true_i = parabolic(np.log(abs(f)), i)[0]

        # Convert to equivalent frequency
        return fs * true_i / len(windowed)
    except:
        return np.nan


def autocorr_freq(signal, fs=1.0):
    """
    Estimate frequency using autocorrelation

    Parameters
        ----------
        signal: np.ndarray
            Array of points representing the oscillating signal.
        fs: float, optional
            Frames per second
    Returns
        ----------
        px: float
            Frequency of oscillation per second (Hz).
    """
    try:
        # Calculate autocorrelation and throw away the negative lags
        corr = correlate(signal, signal, mode="full")
        corr = corr[len(corr) // 2 :]

        # Find the first low point
        d = np.diff(corr)
        start = np.nonzero(d > 0)[0][0]

        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable for long signals, due to the desired peak occurring between
        # samples, and other peaks appearing higher.
        # Should use a weighting function to de-emphasize the peaks at longer lags.
        peak = np.argmax(corr[start:]) + start
        px, py = parabolic(corr, peak)

        return fs / px
    except:
        return np.nan


# def velocity(p, pad=1):
#    print(p)
#    vel = np.sqrt((p[2] - p[0]) ** 2 + (p[3] - p[1]) ** 2)
#    return [np.nan] * pad + vel


def get_edges(
    corners: list[tuple[int, int], ...], length_sort: Optional[bool] = True
) -> list[tuple[int, int], tuple[int, int]]:
    """Creates list of edges defined by pairs of consecutive vertices. Optional: the edges
    may be sorted by length (ascending)

    Parameters
        ----------
        corners: list[tuples]
            List of corner points. Each corner defined as tuple (x,y)
        length_sort: bool, optional
            If True, edges will be sorted in descending order by the edges length. Default: True.
    Returns
        ----------
        edges: list
            list of edges, each edge defined as ((x1,y1), (x2,y2)).
    """
    edges = [(corners[i], corners[i + 1]) for i in range(len(corners) - 1)]
    edges.append((corners[len(corners) - 1], corners[0]))
    if length_sort:
        edges.sort(key=euclidian)
    return edges


def center(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    """Get the geometric center of two points

    Parameters
        ----------
        p1: tuple
           Coordinates (x,y) of first point.
        p2: tuple
           Coordinates (x,y) of second point.
    Returns
        ----------
        c: tuple
           Coordinates (x,y) of center
    """
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


def profile_endpoints(
    p1: tuple[int, int], p2: tuple[int, int], center: tuple[int, int], length: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Returns pair of coordinate tuples that define a profile line of length <length> that crosses
    through p1[0]/p1[1] and p2[0]/p2[1] and with center[0]/center[1] as its center.

    Parameter:
        p1: tuple
            First point (x,y) on new profile line (line scan).
        p2: tuple
            Second point (x,y) on newprofile line (line scan).
        center: tuple
            Defines center (x,y) of new profile line.
        length: float
            Length of new profile line.
    Returns
        endpoints: tuple
            tuple of the endpoints of the new profile line.

    """
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
        p_line = (end1, end2)
    else:
        p_line = (0, 256), (512, 256)
    return p_line


def get_angle(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Calculates the angle defined between two line segments defined
    by p1->origin (0,0) and p2->origin (0,0) linesegments.

    Parameters:
        ----------
        p1: tuple
            Coordinates (x,y) of the first point
        p2: tuple
            Coordinates (x,y) of the second point

    Returns
        ----------
        angle: float
            Angle in degrees between p1->origin (0,0) and p2->origin (0,0) linesegments.
    """
    radians = atan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = degrees(radians)
    return angle


def get_row_angle(r: pd.Series) -> float:
    """Calculates the angle defined between two line segments defined
    by p1->origin (0,0) and p2->origin (0,0) linesegments.

    Parameters:
        ----------
        r: pd.DataSeries
            Values at index 0-3 correspond in order to x of p1, y of p1, x of p2, y of p2.

    Returns
        ----------
        angle: float
            Angle in degrees between p1->origin (0,0) and p2->origin (0,0) linesegments.
    """
    p1 = (r.iloc[0], r.iloc[1])
    p2 = (r.iloc[2], r.iloc[3])
    angle = get_angle(p1, p2)
    if angle < 0:
        angle = angle + 360
    return angle


def get_row_euclidian(r: pd.Series, pixel_res: Optional[float] = 1.0) -> float:
    """Calculates the angle defined between two line segments defined
    by p1->origin (0,0) and p2->origin (0,0) linesegments.

    Parameters:
        ----------
        r: pd.DataSeries
            Values at index 0-3 correspond in order to x of p1, y of p1, x of p2, y of p2.
        pixel_res: float, optional
            The size of a pixel in physical units. Applied as scaling factor to the calculated osciallation amplitudes
    Returns
        ----------
        dist: float
            Distance between p1 and p2.
    """
    p1 = (r.iloc[0], r.iloc[1])
    p2 = (r.iloc[2], r.iloc[3])
    dist = pixel_res * euclidian(p1=p1, p2=p2)
    return dist


def intersect_line(
    line: tuple[tuple[float, float], tuple[float, float]], p: tuple[float, float]
) -> tuple[float, float]:
    """ "Determines the point (x,y) on a line that is closest to a given point p. The resulting line between (x,y) and p
    is orthoginal to the given line.

    Parameters
        ---------
        line: tuple
            The given line defined as a pair of endpoints (x,y).
        p: tuple
            A point (x,y)

    Returns
        ----------
        intersect_p: tuple
            The point (x,y) that is on the given line with the shortest distance to the given input point.
    """
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
    return (x, y)


def get_orientation(
    pts: np.ndarray,
) -> tuple[float, tuple[int, int], np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    """Uses PCA to get orientation of shape in 2D image.

    Parameters
        ----------
        pts: np.ndarray
            [[(x,y),(x,y)]]

    Returns
        ----------
        angle: float
           Angle of the refline in radians
        center: tuple[int, int]
           The center (x,y) of the points.
        first_norm: np.ndarray
           The first normal vector to the refline, i.e. the vector that is orthogonal to refline, starts at the origin (0,0) and ends on the refline intersect.
        refline: tuple[tuple[int, int], tuple[int, int]]
           The line [(x1,y1), (x2,y2)] reflecting the long axis of the object going through the center point.
    """
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
    return angle, cntr, first_norm, refline
