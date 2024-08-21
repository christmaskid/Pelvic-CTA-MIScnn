import numpy as np
cimport numpy as cnp
from scipy import interpolate


cdef class Point: # struct (z,y,x)
	cdef:
		public cnp.ndarray pos
		public cnp.ndarray normal
		public cnp.ndarray contour
		public Plane plane
		public double area
		public CenterlineNode belongCenterlineNode

	cdef Point copy(self)
	cpdef void set_pos(self, cnp.ndarray pos)
	cpdef void set_normal(self, list normal)
	cpdef void set_contour(self, list contour)
	cpdef void calculate_area(self)
	cpdef void smooth_and_sample_contour(self)
	cpdef void setCenterlineNode(self, CenterlineNode belongCenterlineNode)



cdef class CenterlineNode: # Tree
	cdef:
		public CenterlineNode par # parent
		public list children # list of CenterlineNode
		public list points # list of points
		public list mergedNodes
		public int name
		public list spline_pts

	cpdef void add_child(self, CenterlineNode child)
	cpdef void rem_child(self, CenterlineNode child)
	cpdef void add_point(self, Point pt, str msg=*)
	cpdef void merge_node(self, CenterlineNode otherNode)
	cpdef printNode(self)
	cpdef void smooth_points(self, bint replace=*)
	cpdef void get_normals(self, int norm_sep=*)
	# cpdef void get_contours(self, list segments, object vert_map, list list_idx_of_seg_idx, tuple label_shape)
	cpdef void get_contours_and_areas(self, cnp.ndarray labelmap, int background_value = *, double ROI_radius = *)


cdef class Segment:
	cdef:
		public cnp.ndarray origin, final
	cpdef intersect_with_plane(self, Plane plane)


cdef class Plane:
	cdef:
		public cnp.ndarray origin, normal
		public double d

cdef class VesselSeries:
	cdef:
		public CenterlineNode nodes