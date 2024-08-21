import numpy as np
cimport numpy as cnp
from scipy import interpolate
# import pandas as pd

ctypedef cnp.int64_t DTYPE_t

cdef class Point: # struct (z,y,x)

	def __init__(self, list pos):
		self.pos = np.array(pos)
		# dim: len(pos)

	cdef Point copy(self):
		cdef:
			Point new_pt = Point(self.pos)
		return new_pt

	cpdef void set_pos(self, cnp.ndarray pos):
		self.pos = np.array(pos)

	cpdef void set_normal(self, list normal):
		self.normal = np.array(normal) / np.dot(normal, normal)
		# normalized to unit vector

	cpdef void set_contour(self, list contour):
		cdef:
			cnp.ndarray np_contour, indices, _
		np_contour = np.array(contour)
		_, indices = np.unique(np_contour, axis=0, return_index=True)
		self.contour = np_contour[np.sort(indices)]
		# self.contour = pd.unique(np_contour)

	cpdef void setCenterlineNode(self, CenterlineNode belongCenterlineNode):
		self.belongCenterlineNode = belongCenterlineNode
	"""

	# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
	# https://stackoverflow.com/questions/30825123/find-area-of-3d-polygon
	cpdef void calculate_area(self):
		cdef:
			cnp.ndarray Xs, Ys
			double projected_area, cos_theta

		Xs = self.contour[:, 0]
		Ys = self.contour[:, 1]
		projected_area = 0.5 * np.abs( np.dot(Xs, np.roll(Ys, 1) ) - np.dot(Ys, np.roll(Xs, 1)) )

		# cos_theta = cos(theta), 
		# theta = the angle the plane makes with xy-plane
		#       = the angle the normal makes with z-axis
		cos_theta = self.normal[2]
		 # = np.dot(np.array(self.normal), np.array([0,0,1])) / np.sum((self.normal)**2)
		self.area = projected_area * cos_theta
	"""

	cpdef void calculate_area(self):
		cdef:
			cnp.ndarray n1, n2
			int i

		# calculate area
		self.area = 0
		for i in range(len(self.contour)):
			n1 = np.array(self.contour[i] - self.pos)
			n2 = np.array(self.contour[(i+1)%len(self.contour)] - self.pos)
			self.area += np.linalg.norm(np.cross(n1, n2))


	cpdef void smooth_and_sample_contour(self):
		cdef:
			list spline_contour
			cnp.ndarray pts, u, outx, outy, outz
			list x, y, z
			object tck, _
			double x1, y1, z1
			int i

		if len(self.contour)>3:
			x,y,z = self.contour[:, 0].tolist(), self.contour[:, 1].tolist(), self.contour[:, 2].tolist()
			# print(x,y,z,flush=True)
			spline_contour = []
			tck, _ = interpolate.splprep([x,y,z], s=5)
			u = np.linspace(0, 1, num=len(x)//2, endpoint=False)
			outx, outy, outz = interpolate.splev(u, tck)
			for x1, y1, z1 in zip(outx, outy, outz):
				spline_contour.append([x1,y1,z1])

			self.contour = np.array(spline_contour)
			# print(self.contour, flush=True)


name_cnt = 1
def get_name():
	global name_cnt
	name_cnt += 1
	return name_cnt - 1

cdef class CenterlineNode: # Tree

	def __init__(self, par):
		self.name = get_name()
		self.par = par
		self.children = []
		self.points = []
		self.mergedNodes = []
		self.spline_pts = []

	cpdef void add_child(self, CenterlineNode child):
		self.children.append(child)

	cpdef void rem_child(self, CenterlineNode child):
		child.par = None
		self.children.remove(child)

	cpdef void add_point(self, Point pt, str msg=None):
		if len(self.points)>0 and np.array_equal(self.points[-1].pos, pt.pos):
			print(msg, pt.pos, flush=True)
			exit()
		self.points.append(pt)
		pt.setCenterlineNode(self)


	cpdef void merge_node(self, CenterlineNode otherNode):
		# print("Before:")
		# self.printNode()
		# print(f"Merge [{self.name}]{len(self.points)} and [{otherNode.name}]{len(otherNode.points)}")
		# must only merge with one child
		try:
			assert otherNode.par is self
		except:
			self.printNode()
			otherNode.printNode()
			exit()

		self.points += otherNode.points
		self.children.remove(otherNode)
		for child in otherNode.children:
			self.add_child(child)
			child.par = self

		self.mergedNodes.append(otherNode.name)

		# print("After:")
		# self.printNode()

	cpdef printNode(self):
		# print(f"[{self.name}]: {len(self.points)} points ({[pt.pos for pt in self.points]}), parent [{self.par.name}], {len(self.children)} child(ren) ({[c.name for c in self.children]}).")
		print(f"[{self.name}]: {len(self.points)} points, parent [{self.par.name}], {len(self.children)} child(ren) ({[c.name for c in self.children]}), merge: {self.mergedNodes}")

	cpdef void smooth_points(self, bint replace=False):
		cdef:
			cnp.ndarray pts, u, outx, outy, outz #, x, y, z
			list x, y, z
			object tck, _
			double x1, y1, z1
			int i

		#remove adjacent duplicates; but how did they occur? (TODO)
		x,y,z=[],[],[]
		for i in range(len(self.points)):
			if i>0 and np.array_equal(self.points[i].pos, self.points[i-1].pos):
				continue
			x.append(self.points[i].pos[0])
			y.append(self.points[i].pos[1])
			z.append(self.points[i].pos[2])
		# pts = np.array([pt.pos for pt in self.points])
		# x,y,z = pts[:,0], pts[:,1], pts[:,2]
		# print(pts)

		if len(x)>3:
			self.spline_pts = []
			tck, _ = interpolate.splprep([x,y,z], s=5)
			u = np.linspace(0, 1, num=len(x), endpoint=True)
			outx, outy, outz = interpolate.splev(u, tck)
			for x1, y1, z1 in zip(outx, outy, outz):
				self.spline_pts.append(Point([x1,y1,z1]))
		else:
			self.spline_pts = self.points.copy()

		if replace:
			# self.points = self.spline_pts
			for i in range(len(self.points)):
				self.points.set_pos(self.spline_pts[i].pos)


	cpdef void get_normals(self, int norm_sep = 3):
		cdef:
			int i, j, w
			cnp.ndarray normal, pt1, pt2


		for i in range(len(self.points)):
			normal = np.zeros((3,))
			w = 0
			for j in range(norm_sep):
				if i-j-1 >= 0:
					pt1 = self.points[i-j].pos
					pt2 = self.points[i-j-1].pos
					w = 1
					normal += (pt1-pt2)*w
				if i+j+1 < len(self.points):
					pt1 = self.points[i+j+1].pos
					pt2 = self.points[i+j].pos
					w = 1
					normal += (pt1-pt2)*w

			if np.linalg.norm(normal) > 0:
				normal /= np.linalg.norm(normal)

			self.points[i].set_normal(normal.tolist())

		for i in range(len(self.points)):
			if np.isnan(self.points[i].normal).any():
				print(i, self.points[i].normal, flush=True)

	cpdef void get_contours_and_areas(self, cnp.ndarray labelmap, int background_value = 0, double ROI_radius = 30):

		cdef:
			list intersect_pts, radius_list, new_list
			list min_pos, max_pos
			Plane plane
			int idx1, idx2, i,j,k, flag
			object pt
			cnp.ndarray origin, n1, n2, circ_pt
			tuple label_shape = (labelmap.shape[0], labelmap.shape[1], labelmap.shape[2])
			double radius, theta, avg_radius, std_radius, area

		# print("Get contours", flush=True)

		for k in range(len(self.points)):
			# print(k, flush=True)
			plane = Plane(self.points[k].pos, self.points[k].normal)
			origin = self.points[k].pos
			intersect_pts = []

			min_pos = [max(0, plane.origin[i]-ROI_radius) for i in range(3)]
			max_pos = [min(label_shape[i]-1, plane.origin[i]+ROI_radius) for i in range(3)]
			# print(min_pos, max_pos, flush=True)

			n1 = np.array([plane.normal[1], -plane.normal[0], 0])
			n2 = np.array([plane.normal[2], 0, -plane.normal[0]])
			if np.linalg.norm(n1) > 0: n1 /= np.linalg.norm(n1)
			if np.linalg.norm(n2) > 0: n2 /= np.linalg.norm(n2)
			# print("n1/n2:", n1, n2)

			radius_list = []

			for theta in np.linspace(0, 2*np.pi, num=int(2*ROI_radius*np.pi)):
				radius = 1
				while radius <= ROI_radius:
					# print(k, radius, flush=True)
					circ_pt = (plane.origin + radius * (n1 * np.cos(theta) + n2 * np.sin(theta)))

					flag = 0
					# if np.isnan(circ_pt).any():
					# 	print("normal", self.points[k].normal, plane.normal, flush=True)
					# 	print(radius, n1, n2, theta, circ_pt, flush=True)
					# 	continue
					# for i in range(3):
					# 	if np.isnan(circ_pt[i]) or circ_pt[i] < 0 or circ_pt[i] > label_shape[i]-1:
					# 		flag = 1
					# 		break
					# if flag == 1:
					# 	break

					for i in range(3):
						circ_pt[i] = min(max(0, circ_pt[i]), label_shape[i]-1)


					# print(circ, flush=True)
					if labelmap[int(circ_pt[0])][int(circ_pt[1])][int(circ_pt[2])] == 0:
						radius_list.append(radius)
						intersect_pts.append(circ_pt)
						break
					radius += 1

			if len(radius_list)>0:
				avg_radius = sum(radius_list) / len(radius_list)
				std_radius = np.std(radius_list)

				new_list = []
				# Exclude wrong points
				for i in range(len(intersect_pts)):
					if i>0 and np.equal(intersect_pts[i-1], intersect_pts[i]).all():
						continue # repeated point
					# if radius_list[i] >= avg_radius + std_radius * 2:
						# continue # outlier
					new_list.append(intersect_pts[i])
				intersect_pts = new_list

			self.points[k].set_contour(intersect_pts)
			self.points[k].calculate_area()

		# print("Finish", len(self.points), flush=True)
	
	"""

	cpdef void get_contours(self, list segments, object vert_map, list list_idx_of_seg_idx, tuple label_shape):

		cdef:
			set segment_indices_set, segments_in_ROI, s
			list segment_indices, intersect_pts
			list res, min_pos, max_pos
			tuple segment_index, vert1, vert2
			Plane plane
			int idx1, idx2, i,j,k
			object pt
			cnp.ndarray origin, n1, n2, circ
			double radius, theta, cur_ROI_radius

			double ROI_radius = 30

		# print("Get contours", flush=True)

		for k in range(len(self.points)):
			# print(k, flush=True)
			plane = Plane(self.points[k].pos, self.points[k].normal)
			intersect_pts = []

			segments_in_ROI = set()
			# Find segments in the segment's neighboring ROI
			min_pos = [max(0, plane.origin[i]-ROI_radius) for i in range(3)]
			max_pos = [min(label_shape[i]-1, plane.origin[i]+ROI_radius) for i in range(3)]
			# print(min_pos, max_pos, flush=True)

			n1 = np.array([plane.normal[1], -plane.normal[0], 0])
			n2 = np.array([plane.normal[2], 0, -plane.normal[0]])
			# print("n1/n2:", n1, n2)

			# cur_ROI_radius = 0
			# for theta in np.linspace(0, 2*np.pi, num=int(ROI_radius*np.pi)):
			# 	radius = 1
			# 	while radius < ROI_radius:
			# 		# print(k, radius, flush=True)
			# 		circ = (plane.origin + radius * (n1 * np.cos(theta) + n2 * np.sin(theta))).astype(int)
			# 		for i in range(3):
			# 			circ[i] = min(max(0, circ[i]), label_shape[i]-1)
			# 		# print(circ, flush=True)
			# 		if vert_map[circ[0]][circ[1]][circ[2]] != -1:
			# 			cur_ROI_radius += radius
			# 			break
			# 		radius += 1
			# cur_ROI_radius /= int(ROI_radius*np.pi)

			cur_ROI_radius = ROI_radius

			for theta in np.linspace(0, 2*np.pi, num=int(cur_ROI_radius*np.pi)):
				radius = 1
				while radius <= cur_ROI_radius:
					# print(k, radius, flush=True)
					circ = (plane.origin + radius * (n1 * np.cos(theta) + n2 * np.sin(theta))).astype(int)
					for i in range(3):
						circ[i] = min(max(0, circ[i]), label_shape[i]-1)
					# print(circ, flush=True)
					if vert_map[circ[0]][circ[1]][circ[2]] != -1:
						s = list_idx_of_seg_idx[vert_map[circ[0]][circ[1]][circ[2]]]
						if len(s)>0:
							# print(circ, vert_map[circ[0]][circ[1]][circ[2]], s, flush=True)
							segments_in_ROI = segments_in_ROI.union(s)
							break
					radius += 1

			for i in segments_in_ROI:
				pt = segments[i].intersect_with_plane(plane)
				if pt is not None:
					intersect_pts.append(pt)
			# print(intersect_pts, flush=True)
			self.points[k].set_contour(intersect_pts)

		# print("Finish", len(self.points), flush=True)

	"""
		

cdef class Segment:

	def __init__(self, cnp.ndarray origin, cnp.ndarray final):
		# segment: (x0,y0,z0),(x1,y1,z1) 
		# => (x,y,z)=(x0+dx*t, y0+dy*t, z0+dz*t), dx = x1-x0, etc.
		# segment: origin(x_0), tangent(x_1)
		self.origin = origin
		self.final = final

	cpdef intersect_with_plane(self, Plane plane):
		# Solve: line & plane, t in [-1,1]
		# => (a*x0+b*y0+c*z0) + (a*dx+b*dy+c*dz) = d
		# => t = (d-(a*x0+b*y0+c*z0)) / (a*dx+b*dy+c*dz)
		#	  = (d- n dot x_0) / (n dot (x_1 - x_0))
		cdef:
			double base, t

		base = np.sum((self.final-self.origin) * plane.normal)
		if base == 0: # parallel
			return None
		t = (plane.d - np.sum(plane.normal * self.origin)) / base
		if t >= -1 and t <= 1:
			return self.origin + t * (self.final - self.origin) # the intersected point
		else:
			return None

cdef class Plane:

	def __init__(self, cnp.ndarray origin, cnp.ndarray normal):
		# plane: (a,b,c,d) => ax+by+cz = d
		self.origin = origin
		self.normal = normal
		self.d = np.sum(origin * normal) # inner product
