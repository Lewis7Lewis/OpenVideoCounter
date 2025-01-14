"""Important functions"""

import numpy as np

def beautifultime(s):
    """Make a string from a second time"""
    m = s // 60
    h = m // 60
    m = m % 60
    s = s % 60
    return f"{h:2}:{m:0>2d}:{s:0>2d}"


def determinant(u, v):
    """Calculate the determinant of a 2scalar vector"""
    return u[0] * v[1] - u[1] * v[0]


def vect(a: tuple, b: tuple) -> tuple:
    """Calculate the vector of a 2 points"""
    return (b[0] - a[0], b[1] - a[1])


def distance(a, b):
    """calculate the euclerien distance(norm L2)"""
    v = np.array(b)-np.array(a)
    return np.linalg.norm(v,2)


def intersects(s0, s1):
    """Check if to line intersect"""
    dx0 = s0[1][0] - s0[0][0]
    dx1 = s1[1][0] - s1[0][0]
    dy0 = s0[1][1] - s0[0][1]
    dy1 = s1[1][1] - s1[0][1]
    p0 = dy1 * (s1[1][0] - s0[0][0]) - dx1 * (s1[1][1] - s0[0][1])
    p1 = dy1 * (s1[1][0] - s0[1][0]) - dx1 * (s1[1][1] - s0[1][1])
    p2 = dy0 * (s0[1][0] - s1[0][0]) - dx0 * (s0[1][1] - s1[0][1])
    p3 = dy0 * (s0[1][0] - s1[1][0]) - dx0 * (s0[1][1] - s1[1][1])
    return (p0 * p1 <= 0) & (p2 * p3 <= 0)
