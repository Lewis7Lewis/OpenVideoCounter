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


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,0] + boxes[:,2]
	y2 = boxes[:,1] + boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (boxes[:,2] + 1) * (boxes[:,3] + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")