import numpy as np
cimport numpy as cnp
from scipy import interpolate
from structures cimport Point, CenterlineNode

ctypedef cnp.int64_t DTYPE_t

steps = (-1, 0, 1)
directions = tuple((z,y,x) for z in steps for y in steps for x in steps if not (x==0 and y==0 and z==0))
threshold = 2


debug_pts = []#[[462, 234, 264], [462, 233, 263], [463, 233, 262], [463, 233, 261]]
debug_nodes = []#[1, 711]

def DFS(CenterlineNode curNode, Point pt, cnp.ndarray visited, cnp.ndarray labelmap):
	cdef:
		tuple drc
		Point nxt_pt
		list nxt_pos, candidates = []
		CenterlineNode newNode

	if visited[pt.pos[0]][pt.pos[1]][pt.pos[2]]:
		return curNode, visited
	visited[pt.pos[0]][pt.pos[1]][pt.pos[2]] = True

	# print("DFS", pt.pos)		
	if pt.pos in debug_pts :
		print("DFS", pt.pos, end=' ')
		curNode.printNode()
	global directions

	# print("candidates: ", end='')

	for drc in directions:
		nxt_pos = [p+d for p,d in zip(pt.pos, drc)]
		def cond(pos):
			for i in range(len(pos)):
				if pos[i]<0 or pos[i]>=labelmap.shape[i]: return False
			return (labelmap[pos[0]][pos[1]][pos[2]]>0 and not visited[pos[0]][pos[1]][pos[2]])

		if cond(nxt_pos):
			candidates.append(Point(nxt_pos))
			# print(nxt_pt.pos, end=" ")

		# if nxt_pos in debug_pts :
		# 	if cond(nxt_pos):
		# 		print(pt.pos, "--->", nxt_pos)
		# 	else:
		# 		print(pt.pos, "-x->", nxt_pos)
		# 		print(labelmap[nxt_pos[0]][nxt_pos[1]][nxt_pos[2]], visited[nxt_pos[0]][nxt_pos[1]][nxt_pos[2]])

	# print()



	if len(candidates) == 0: # End point
		# print("END")
		# curNode.printNode()
		pass

	elif len(candidates) == 1: # Mid-point: same node
		curNode.add_point(candidates[0], "mid-point")
		curNode, visited = DFS(curNode, candidates[0], visited, labelmap)

	else: # branch point: create new node as child of curNode

		for cand in candidates:
			newNode = CenterlineNode(curNode)
			newNode.add_point(cand, "branch-point")
			curNode.add_child(newNode)
			newNode, visited = DFS(newNode, cand, visited, labelmap)

			if len(newNode.points) <= threshold:
				curNode.merge_node(newNode)

				# if nxt_pos in debug_pts or pt.pos in debug_pts:
				# 	print(pt.pos, "--->", nxt_pos, 'but short appendage was merged back. (',len(newNode.points),')')

		if len(curNode.children) == 1:
			curNode.merge_node(curNode.children[0])

		# print("BRANCH")
		# curNode.printNode()

	# if pt.pos in debug_pts:
	# 	print("Finish: ", pt.pos, end='')
	# 	curNode.printNode()

	return curNode, visited



def find_centerline_forest_3d(cnp.ndarray labelmap):
	cdef:
		list forest = [] # list of CenterlineNode
		int z0, y0, x0, x, y, z
		object ys, xs
		CenterlineNode root, dummy
		Point pt
		object visited = np.full((labelmap.shape[0], labelmap.shape[1], labelmap.shape[2]), False)
		object centerline_map = np.full((labelmap.shape[0], labelmap.shape[1], labelmap.shape[2]), -1)

	# for item in debug_pts:
	# 	x,y,z = item[0], item[1], item[2]
	# 	print(item, labelmap[x][y][z])
	# 	input()

	# print(directions, flush=True)

	dummy = None
	z0, y0, x0 = labelmap.shape[0], labelmap.shape[1], labelmap.shape[2]
	z = z0 - 1
	while z>0:
		ys, xs = np.where(labelmap[z]>0)
		# print(z, ys, xs, flush=True)
		for y, x in zip(ys, xs):
			if visited[z][y][x]: 
				# print(z,y,x,"visited")
				continue

			root = CenterlineNode(dummy)
			pt = Point([z,y,x])
			root.add_point(pt, "start-point")

			# print("New DFS", z, y, x, flush=True)
			root, visited = DFS(root, pt, visited, labelmap)
			forest.append(root)
			# input()
		z -= 1

	# def traverse(node):
	# 	# node.printNode()
	# 	# input()
	# 	num = node.name
	# 	for pt in node.points:
	# 		centerline_map[pt.pos[0]][pt.pos[1]][pt.pos[2]] = num
	# 	for child in node.children:
	# 		traverse(child)

	# for tree in forest:
	# 	traverse(tree)

	# for item in debug_pts:
	# 	x,y,z = item[0], item[1], item[2]
	# 	print(item, centerline_map[x][y][z])

	return forest

cdef CenterlineNode merge_tree(CenterlineNode curNode):
	
	cdef:
		list children_list = []
	
	children_list = curNode.children[:]

	for child in children_list:
		merge_tree(child)
		curNode.merge_node(child)

	return curNode

def flatten_forest(list forest):
	for tree in forest:
		merge_tree(tree)
		input()
	return forest

def reconstruct_tree(CenterlineNode root):
	res = dict()
	root.printNode()
	res["Aorta"] = root

	# Bifur pt
	# res["R_Iliac"] = merge_tree(root.children[0])
	# res["L_Iliac"] = merge_tree(root.children[1])

	R_iliac = root.children[0]
	L_iliac = root.children[1]
	res["R_iliac"] = R_iliac
	res["L_iliac"] = L_iliac
	res["A"] = merge_tree(R_iliac.children[0])
	res["B"] = merge_tree(R_iliac.children[1])
	res["C"] = merge_tree(L_iliac.children[0])
	res["D"] = merge_tree(L_iliac.children[1])
	# print(len(R_iliac.points), len(L_iliac.points))
	# if len(R_iliac.points) < len(L_iliac.points):
	# 	res["L_Iliac"] = merge_tree(L_iliac)
	# 	R_iliac.merge_node(R_iliac.children[1])
	# 	res["median_sacral"] = R_iliac.children[0]
	# 	R_iliac.rem_child(R_iliac.children[0])
	# 	res["R_Iliac"] = merge_tree(R_iliac)
	# else:
	# 	res["R_Iliac"] = merge_tree(R_iliac)
	# 	L_iliac.merge_node(L_iliac.children[0])
	# 	res["median_sacral"] = L_iliac.children[1]
	# 	L_iliac.rem_child(L_iliac.children[1])
	# 	res["L_Iliac"] = merge_tree(L_iliac)

	return res
