import numpy as np
cimport numpy as cnp
from structures cimport Point, CenterlineNode

cdef class Centerline:
	cdef:
		public CenterlineNode root
		public list steps, points, normals, contours

	def __init__(self, CenterlineNode root, list steps):
		self.root = root
		self.steps = steps
		self.points = []

	cpdef create_centerline(self):
		cdef:
			int step
			CenterlineNode curNode
			Point pt

		curNode = self.root

		for step in self.steps:
			self.points += curNode.points
