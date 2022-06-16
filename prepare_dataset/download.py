import sys  
import json 
from subprocess import Popen  
import mapdriver as md 
import mapbox as md2
import graph_ops as graphlib 
import math 
import numpy as np 
import scipy.misc 
from PIL import Image 
import pickle 
from skimage.morphology import skeletonize
import networkx as nx
import sknw
import tifffile
import cv2
import matplotlib.pyplot as plt
from itertools import chain
import os


tid = 0 #int(sys.argv[1])
tn = 1 #int(sys.argv[2])




if not os.path.isdir('tmp'):
	os.mkdir('tmp')
dataset_folder = f'../data/ryam'
if not os.path.isdir(dataset_folder):
	os.mkdir(dataset_folder)

zones = ['1','2','3']

for zone in zones:
	

	##############################
	# Prepare data

	im_path = f'D:/cmarseille/reseaux/datasets/sat2graph_data/zone_{zone}.tif'
	gt_path = f'D:/cmarseille/reseaux/datasets/sat2graph_data/zone_{zone}_mask.tif'


	#Load image and normalize
	img = tifffile.imread(im_path)
	img_ori_shape = img.shape
	img = cv2.resize(img[100:-101, 100:-100],(2048,2048))			#remove first line in x dimension because gt was not the same size

	img[img == 65536] = np.nan 			#set to nan the no data pixels
	img[:,:,0] = (img[:,:,0] - np.nanmin(img[:,:,0])) / (np.nanmax(img[:,:,0]) - np.nanmin(img[:,:,0]))
	img[:,:,1] = np.where(img[:,:,1]>50.0, 50.0, img[:,:,1]) / 50.0
	img[:,:,2][img[:,:,2]>=728] = np.nan
	img[:,:,2] /= np.nanmax(img[:,:,2])
	img[:,:,3] = np.where(img[:,:,3]>5.0, 5.0, img[:,:,3]) / 5.0
	img[:,:,3] = img[:,:,3]/img[:,:,3].max()

	img[np.isnan(img)] = 0
	print('min, max, mean, std')
	[print(img.min(), img.max(), img.mean(), img.std()) for img in img.T]

	#Load ground truth data and normalize
	gt = tifffile.imread(gt_path)[100:-100, 100:-100]

	if zone == '3':
		gt = 1-gt
	gt = cv2.resize(gt.astype(np.uint8),(2048,2048))
	gt = gt/gt.max()


	#Save data for dataloader
	tifffile.imsave(dataset_folder+f'/region_{c}_sat.tif', img)
	

	bands = ['mhc', 'hillshade', 'slope', 'tri']
	for i in range(img.shape[2]):
		tifffile.imsave(dataset_folder+f'/region_{c}_sat_{bands[i]}.tif', img[:,:,i])



	#Dilate original gt because ground truth network is not always closed and sknw then creates extra nodes   
	kernel = np.ones((3, 3), 'uint8')
	gt_dilated = cv2.dilate(gt, kernel, iterations=1)
	tifffile.imsave(dataset_folder+f'/region_{c}_gt.tif', gt)

	skel_dilated = skeletonize(gt_dilated).astype(np.uint8)
	gt_graph_dilated = sknw.build_sknw(skel_dilated, iso=True)


	gt_graph = gt_graph_dilated



	#Modified add path function to account for added 'pts' and 'o' key in graph nodes
	def add_path(G_to_add_to, nodes_for_path, **attr):
	    nlist = iter(nodes_for_path)
	    try:
	        first_node = next(nlist)
	    except StopIteration:
	        return
	    #G_to_add_to.add_node(first_node, pts=1, o=1)
	    G_to_add_to.add_edges_from(nx.utils.misc.pairwise(chain((first_node,), nlist)), **attr)


	#Add nodes at 20m interval along groundtruth edges

	#Original tiles have 1m resolution and different shapes. Following converts pixel to distance to compute 20m interval.
	interval = 60
	resize_shape = 2048
	scale_factor = np.sqrt(img_ori_shape[0]**2+img_ori_shape[1]**2)/resize_shape
	interval_scaled = int(interval/scale_factor)

	gt_graph_increased = nx.Graph()
	n_new_nodes = gt_graph.number_of_nodes()


	path_lengths = []
	nnids = []
	n=0
	for (start,end) in gt_graph.edges():
		ps = gt_graph[start][end]['pts']
		xs, ys = np.array([edge for edge in ps]).T
		path_length = np.sum([np.sqrt( (xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2 ) for i in range(xs.shape[0]-1)])
		path_lengths.append(int(path_length))
		n = int(path_length/interval_scaled)
		if n > 0:
			new_nodes = ps[np.linspace(0, ps.shape[0], n).astype(int)[1:-1]]
			new_nodes_ids = np.arange(n_new_nodes, n_new_nodes+len(new_nodes))
			nnids.append(new_nodes_ids)
			n_new_nodes += len(new_nodes)
			for i in range(len(new_nodes)):
				n+=1
				gt_graph_increased.add_node(new_nodes_ids[i], pts=np.array([new_nodes[i]]), o=np.array([new_nodes[i]]))
				if i == 0:
					if i == len(new_nodes)-1:
						add_path(gt_graph_increased, [start, new_nodes_ids[i], end], pts=ps, os=ps)
					else:
						add_path(gt_graph_increased, [start, new_nodes_ids[i], new_nodes_ids[i+1]], pts=ps, os=ps)
				elif i == len(new_nodes)-1:
					add_path(gt_graph_increased, [new_nodes_ids[i-1], new_nodes_ids[i], end], pts=ps, os=ps)
				else:
					add_path(gt_graph_increased, [new_nodes_ids[i-1], new_nodes_ids[i], new_nodes_ids[i+1]], pts=ps, os=ps)


	###############################
	#Plot results

	# draw node by o
	plt.imshow(img)

	nodes = gt_graph.nodes()
	ps = np.array([nodes[i]['o'] for i in nodes])
	plt.plot(ps[:,1], ps[:,0], 'r.')

	nodes = gt_graph_increased.nodes()
	ps = []
	for i in nodes:		#Remove bad nodes that did not get 'pts' and 'o' keys (??idk)
		try:
			ps.append(nodes[i]['pts'][0])
		except: continue

	ps = np.array(ps)
	plt.plot(ps[:,1], ps[:,0], 'bX')

	for (start,end) in gt_graph.edges():
		ps = gt_graph[start][end]['pts']
		plt.plot(ps[:,1], ps[:,0], 'green')

	plt.show()


	###############################
	#Save data

	#Combine both graphs
	gt_graph = nx.compose(gt_graph,gt_graph_increased)


	#create dict of nodes x,y positions and neighbors x,y positions (as is expected by dataloader.py)
	gt_graph_dict = {}
	neighbors_list = [list(gt_graph.neighbors(i)) for i in gt_graph.nodes]
	nodes = np.array(list(gt_graph.nodes))
	nodes_values = np.array(list(gt_graph.nodes.values()))

	for node in nodes:
		gt_graph_dict.update({tuple(nodes_values[node]['pts'][0]):[tuple(nodes_values[neighbor]['pts'][0]) for neighbor in neighbors_list[node]]})


	#Transform graph xy coords to lat lon (random but needed )
	gt_graph_dict_gps = graphlib.graph2RegionCoordinate(gt_graph_dict, [47.39,-78.95,47.45,-78.85])


	gt_graph_path = dataset_folder+f'/region_{zone}_refine_gt_graph.p'

	pickle.dump(gt_graph_dict_gps, open(gt_graph_path, 'wb'))