import numpy as np
cimport numpy as cnp
from skimage.morphology import skeletonize_3d
from skimage import measure
from vessel_track import find_centerline_forest_3d
from structures cimport Segment, CenterlineNode, Point

def get_all_contours(
		cnp.ndarray sample,
		bint smooth = False,#True,
		double norm_sep = 5,
	):
	cdef:
		cnp.ndarray sample_sk
		list centerline_forest
		Point centerpoint
		CenterlineNode node, child

		list intersect_pts
		list min_pos, max_pos
		int idx1, idx2, i,j,k
		object pt
		cnp.ndarray origin, n1, n2, circ
		double radius, theta

		double ROI_radius = 30

	sample_sk = skeletonize_3d(sample)
	centerline_forest = find_centerline_forest_3d(sample_sk)

	if smooth:
		def dfs_smooth(node):
			node.smooth_points()
			for child in node.children:
				dfs_smooth(child)

		for node in centerline_forest:
			# node.printNode()
			dfs_smooth(node)

	def dfs_normals(node):
		node.get_normals(norm_sep=norm_sep)
		for child in node.children:
			dfs_normals(child)

	for node in centerline_forest:
		node.printNode()
		dfs_normals(node)

	def dfs(node):
		node.get_contours_and_areas(sample)
		for child in node.children:
			dfs(child)

	for node in centerline_forest:
		# node.printNode()
		dfs(node)

	if smooth:
		def dfs_smooth(node):
			for pt in node.points:
				pt.smooth_and_sample_contour()

			for child in node.children:
				dfs_smooth(child)

		for node in centerline_forest:
			# node.printNode()
			dfs_smooth(node)

	return sample_sk, centerline_forest


def construct_mesh_and_get_contours(
		cnp.ndarray sample,
		bint smooth = False,#True,
		double norm_sep = 5,
	):

	# verts, faces: from output of skimage.measure.marching_cube
	cdef:
		cnp.ndarray sample_sk, verts, faces, normals, values
		list centerline_forest
		Point centerpoint
		CenterlineNode node, child

		set segment_indices_set, segments_in_ROI, s
		list segment_indices, segments, intersect_pts
		list min_pos, max_pos
		list list_idx_of_seg_idx
		tuple segment_index, vert1, vert2
		int idx1, idx2, i,j,k
		object pt, vert_map
		cnp.ndarray origin, n1, n2, circ
		dict vert_to_seg_idx
		double radius, theta

		double ROI_radius = 30


	sample_sk = skeletonize_3d(sample)
	centerline_forest = find_centerline_forest_3d(sample_sk)
	verts, faces, normals, values = measure.marching_cubes(sample)

	print(len(verts), len(faces))

	if smooth:
		def dfs_smooth(node):
			node.smooth_points()
			for child in node.children:
				dfs_smooth(child)

		for node in centerline_forest:
			# node.printNode()
			dfs_smooth(node)

	def dfs_normals(node):
		node.get_normals(norm_sep=norm_sep)
		for child in node.children:
			dfs_normals(child)

	for node in centerline_forest:
		node.printNode()
		dfs_normals(node)


	# verts, faces: from output of skimage.measure.marching_cube
	segment_indices_set = set() # O(Nlog(N)), N = len)faces
	for face in faces:
		for i in range(3):
			idx1 = face[i]
			idx2 = face[(i+1)%3]
			if idx1 < idx2:
				segment_indices_set.add((idx1, idx2))
			else:
				segment_indices_set.add((idx2, idx1))
	segment_indices = sorted(list(segment_indices_set))
	print(len(segment_indices), segment_indices[:3])

	segments = []
	# vert_to_seg_idx[vertex] = list of indices of segment that contain the vertex
	for segment_index in segment_indices:
		segments.append(Segment(verts[segment_index[0]], verts[segment_index[1]]))


	vert_map = np.full((sample.shape[0], sample.shape[1], sample.shape[2]), -1)
	list_idx_of_seg_idx = []

	for i in range(len(segments)):
		segment = segments[i]
		for vert in [segment.origin, segment.final]:
			vert = vert.astype(int)
			if vert_map[vert[0]][vert[1]][vert[2]] == -1:
				vert_map[vert[0]][vert[1]][vert[2]] = len(list_idx_of_seg_idx)
				list_idx_of_seg_idx.append(set())
			list_idx_of_seg_idx[vert_map[vert[0]][vert[1]][vert[2]]].add(i)

	def dfs(node):
		node.get_contours(segments, vert_map, list_idx_of_seg_idx, \
			(sample.shape[0], sample.shape[1], sample.shape[2]))
		for child in node.children:
			dfs(child)

	for node in centerline_forest:
		node.printNode()
		dfs(node)

	if smooth:
		def dfs_smooth(node):
			for pt in node.points:
				pt.smooth_and_sample_contour()

			for child in node.children:
				dfs_smooth(child)

		for node in centerline_forest:
			node.printNode()
			dfs_smooth(node)

	return sample_sk, centerline_forest
