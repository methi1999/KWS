"""
computes DTW for 2 given sequences
"""
import numpy as np
import matplotlib.pyplot as plt

round_f = 6

#distance function used for cost
def distance(x, y, func='euclidean'):
	# print(x.shape, y.shape)

	if func == 'euclidean':
		return np.sqrt(np.sum((x - y) ** 2))
	elif func == 'cosine':
		dot = np.dot(x, y)
		return 1-dot/(np.linalg.norm(x)*np.linalg.norm(y))
	else:
		print("Distance func not implemented")
		exit(0)

#calculate DTW path and min cost between s1 and s2
#distances[i,j] = distance between ith element of s1 and jth element of s2
#if provided, use it. We provided it durign DTW to reduce multiple calls to the distance function
def dtw_own(s1, s2, distances=None):

	m, n = len(s1), len(s2)
	
	if distances is not None:
		assert (distances.shape == (m,n))
	else:
		#calculate distance matrix
		distances = np.zeros((m,n))
		
		for i in range(m):
			for j in range(n):
				distances[i,j] = distance(s1[i], s2[j])
		
	#create cost matrix
	cost = np.ones((m,n))*np.inf
	cost[0,0] = distances[0,0]

	for i in range(1, m):
		cost[i,0] = cost[i-1,0] + distances[i,0]
	
	for j in range(1, n):
		cost[0,j] = cost[0,j-1] + distances[0,j]
	#greedily search for the least cost path
	for i in range(1, m):
		for j in range(1, n):
			left, diag, bottom = cost[i,j-1], cost[i-1,j-1], cost[i-1,j]
			cost[i,j] = min(left, diag, bottom) + distances[i,j]
	
	distances = np.around(distances, decimals=round_f)
	cost = np.around(cost, decimals=round_f)
	# print(cost, distances)
	# exit(0)
	path = []
	i, j = m-1, n-1
	#back track to find the actual path
	while i>=0 and j>=0:
		path.append((i,j))
		# print(i,j)
		if i == 0:
			j -= 1
		elif j == 0:
			i -= 1
		else:
			cur = min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1])
			# print(cur)
			if cost[i-1, j] == cur:
				i = i - 1
			elif cost[i, j-1] == cur:
				j = j-1
			elif cost[i-1,j-1] == cur:
				i -= 1
				j -= 1
			else:
				print("Kuch to gadbad hai at",i,j)
				exit(0)

	return path, cost[m-1,n-1]/(m+n)

#plot heatmap for visualisation
def distance_cost_plot(distances):
	
	im = plt.imshow(distances, interpolation='none', cmap='Reds') 
	plt.gca().invert_yaxis()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.grid()
	plt.colorbar()
#map points form sequence 1 to sequence 2
def show_sim(x, y, path):

	plt.plot(x, 'bo-' ,label='x')
	plt.plot(y, 'g^-', label = 'y')
	plt.legend();
	for (map_x, map_y) in path:
		print(map_x, x[map_x], ":", map_y, y[map_y])
		plt.plot([map_x, map_y], [x[map_x], y[map_y]], 'r')

if __name__ == '__main__':

	# a = [1,2,3,4]
	# b = [1,1,2,2,3,3,4,4]
	x = np.linspace(0, 6.28, 100)
	a, b = np.sin(x), np.cos(x)
	# print(a,b)
	path, cost = dtw_own(a, b)
	print(cost)
	show_sim(a, b, path)
	plt.show()