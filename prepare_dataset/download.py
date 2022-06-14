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

dataset_cfg = []
total_regions = 0 

tid = 0 #int(sys.argv[1])
tn = 1 #int(sys.argv[2])


# for name_cfg in sys.argv[1:]:
# 	dataset_cfg_ = json.load(open(name_cfg, "r"))

	
# 	for item in dataset_cfg_:
# 		dataset_cfg.append(item)
# 		ilat = item["lat_n"]
# 		ilon = item["lon_n"]

# 		total_regions += ilat * ilon 


print("total regions", total_regions)

#Popen("mkdir tmp", shell=True).wait()
#Popen("mkdir googlemap", shell=True).wait() 

dataset_folder = "../data/ryam"
folder_mapbox_cache = "ryam_cache/"
im_path = 'D:/cmarseille/reseaux/datasets/sat2graph_data/zone_1.tif'
gt_path = 'D:/cmarseille/reseaux/datasets/sat2graph_data/zone_1_mask.tif'


#Popen("mkdir %s" % dataset_folder, shell=True).wait()
#Popen("mkdir %s" % folder_mapbox_cache, shell=True).wait()

# download imagery and osm maps 

c = 0
tiles_needed = 0



# for item in dataset_cfg:
# 	#prefix = item["cityname"]
# 	ilat = item["lat_n"]
# 	ilon = item["lon_n"]
# 	lat = item["lat"]
# 	lon = item["lon"]

# 	for i in range(ilat):
# 		for j in range(ilon):
# 			print(c, total_regions)

# 			if c % tn == tid:
# 				pass
# 			else:
# 				c = c + 1
# 				continue


# 			lat = item["lat"]
# 			lon = item["lon"]

# 			lat_st = lat + 2048/111111.0 * i 
# 			lon_st = lon + 2048/111111.0 * j / math.cos(math.radians(lat))
# 			lat_ed = lat + 2048/111111.0 * (i+1)
# 			lon_ed = lon + 2048/111111.0 * (j+1) / math.cos(math.radians(lat))


# 			# download satellite imagery from google
# 			# if abs(lat_st) < 33:
# 			# 	zoom = 18
# 			# else:
# 			# 	zoom = 17

# 			# download satellite imagery from mapbox
# 			if abs(lat_st) < 33:
# 				zoom = 17
# 			else:
# 				zoom = 16

# 			print(lat_st, lon_st, lat_ed, lon_ed)
			

			# comment out the image downloading part 
			#img, _ = md2.GetMapInRect(lat_st, lon_st, lat_ed, lon_ed, start_lat = lat_st, start_lon = lon_st, zoom=zoom, folder = folder_mapbox_cache)
img = tifffile.imread(im_path)

img = cv2.resize(img[100:-101, 100:-100],(2048,2048))			#remove first line in x dimension because gt was not the same size

img[img == 65536] = np.nan 			#set to nan the no data pixels
img[:,:,0] = (img[:,:,0] - np.nanmin(img[:,:,0])) / (np.nanmax(img[:,:,0]) - np.nanmin(img[:,:,0]))
img[:,:,1] = np.where(img[:,:,1]>50.0, 50.0, img[:,:,1]) / 50.0
img[:,:,2][img[:,:,2]>=728] = np.nan
img[:,:,2] /= np.nanmax(img[:,:,2])
img[:,:,3] = np.where(img[:,:,3]>5.0, 5.0, img[:,:,3]) / 5.0

img[np.isnan(img)] = 0
print('min, max, mean, std')
[print(img.min(), img.max(), img.mean(), img.std()) for img in img.T]


gt = tifffile.imread(gt_path)[100:-100, 100:-100]
gt = cv2.resize(gt.astype(np.uint8),(2048,2048))
gt = gt/gt.max()


bands = ['mhc', 'hillshade', 'slope', 'tri']
for i in range(img.shape[2]):
	tifffile.imsave(dataset_folder+f'/region_{c}_sat_{bands[i]}.tif', img[:,:,i])
tifffile.imsave(dataset_folder+f'/region_{c}_sat.tif', img)
tifffile.imsave(dataset_folder+f'/region_{c}_gt.tif', gt)


#Dilate original gt because ground truth network is not always closed and sknw then creates extra nodes   
kernel = np.ones((3, 3), 'uint8')
gt_dilated = cv2.dilate(gt, kernel, iterations=1)

#Erode original gt because not enough nodes for algo
gt_eroded = cv2.erode(gt, kernel)

#Create skeletons and networkx graph objects
skel_dilated = skeletonize(gt_dilated).astype(np.uint8)
#skel_eroded = skeletonize(gt_eroded).astype(np.uint8)
gt_graph_dilated = sknw.build_sknw(skel_dilated, iso=True)
#gt_graph_eroded = sknw.build_sknw(skel_eroded, iso=True)

#Combine both graphs  (order matters! if 2 nodes conflict, the second called is prioritised (here dilated))
#gt_graph = nx.compose(gt_graph_eroded,gt_graph_dilated)
gt_graph = gt_graph_dilated


#Add random nodes along groundtruth
density = 5 	# Number of new nodes to add
gt_graph_increased = nx.Graph()
n_new_nodes = gt_graph.number_of_nodes()
for (start,end) in gt_graph.edges():
	ps = gt_graph[start][end]['pts']
	xs, ys = np.array([edge for edge in ps]).T
	path_length = np.sum([np.sqrt( (xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2 ) for i in range(xs.shape[0]-1)])
	density = 5
	print(path_length)
	if path_length > density:
		new_nodes = ps[np.arange(0, ps.shape[0], int(path_length/density))][1:]	
		new_nodes_ids = np.arange(n_new_nodes, n_new_nodes+len(new_nodes))
		n_new_nodes += len(new_nodes)
		for i in range(len(new_nodes)):

			# if i == len(new_nodes)-1:
			# 	nx.add_path(gt_graph_increased, [start, new_nodes_ids[i], end], pts=ps, os=ps)
			# else:
			# 	nx.add_path(gt_graph_increased, [start, new_nodes_ids[i], new_nodes_ids[i+1]], pts=ps, os=ps)
			# print('new_nodes[i]:', new_nodes[i])
			gt_graph_increased.add_node(new_nodes_ids[i], pts=new_nodes[i], o=new_nodes[i])
			print(new_nodes_ids[i])

for (start,end) in gt_graph.edges():
	ps = gt_graph[start][end]['pts']
	plt.plot(ps[:,1], ps[:,0], 'blue')

# draw edges by pts
for (start,end) in gt_graph_increased.edges():
	ps = gt_graph_increased[start][end]['pts']
	plt.plot(ps[:,1], ps[:,0], 'green')
	
# draw node by o
nodes = gt_graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])

plt.imshow(img)
plt.plot(ps[:,1], ps[:,0], 'k.')
plt.show()


# draw node by o
nodes = gt_graph_increased.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])

plt.imshow(img)
plt.plot(ps[:,1], ps[:,0], 'r.')
plt.show()


#fig, ax =plt.subplots(1, img.shape[2])
#for i in range(img.shape[2]):
#	ax[i].hist(img[:,:,i].flatten(), 100)

			# download openstreetmap 
			#OSMMap = md.OSMLoader([lat_st,lon_st,lat_ed,lon_ed], False, includeServiceRoad = False)

			# node_neighbor = {} # continuous

			# for node_id, node_info in OSMMap.nodedict.iteritems():
			# 	lat = node_info["lat"]
			# 	lon = node_info["lon"]

			# 	n1key = (lat,lon)


			# 	neighbors = []
			# 	for nid in node_info["to"].keys() + node_info["from"].keys() :
			# 		if nid not in neighbors:
			# 			neighbors.append(nid)

			# 	for nid in neighbors:
			# 		n2key = (OSMMap.nodedict[nid]["lat"],OSMMap.nodedict[nid]["lon"])

			# 		node_neighbor = graphlib.graphInsert(node_neighbor, n1key, n2key)
					
			#graphlib.graphVis2048(node_neighbor,[lat_st,lon_st,lat_ed,lon_ed], "raw.png")
			
			# interpolate the graph (20 meters interval)
			#node_neighbor = graphlib.graphDensify(node_neighbor)
			#node_neighbor_region = graphlib.graph2RegionCoordinate(node_neighbor, [lat_st,lon_st,lat_ed,lon_ed])
			


#create dict of nodes x,y positions and neighbors x,y positions (as is expected by dataloader.py)
gt_graph_dict = {}
neighbors_list = [list(gt_graph.neighbors(i)) for i in gt_graph.nodes]
nodes = list(gt_graph.nodes)
nodes_values = list(gt_graph.nodes.values())
for node in nodes:
	gt_graph_dict.update({tuple(nodes_values[node]['pts'][0]):[tuple(nodes_values[neighbor]['pts'][0]) for neighbor in neighbors_list[node]]})


#Plot densified graph
#plt.figure()
for node, nei in gt_graph_dict.items():
	plt.plot(node[1], node[0], 'r.')


gt_graph_path = dataset_folder+"/region_%d_refine_gt_graph.p" % c
#pickle.dump(node_neighbor_region, open(prop_graph, "w"))
pickle.dump(gt_graph_dict, open(gt_graph_path, "wb"))

#graphlib.graphVis2048(node_neighbor,[lat_st,lon_st,lat_ed,lon_ed], "dense.png")
#graphlib.graphVis2048Segmentation(node_neighbor, [lat_st,lon_st,lat_ed,lon_ed], dataset_folder+"/region_%d_" % c + "gt.png")

#node_neighbor_refine, sample_points = graphlib.graphGroundTruthPreProcess(node_neighbor_region)

#refine_graph = dataset_folder+"/region_%d_" % c + "refine_gt_graph.p"
#pickle.dump(node_neighbor_refine, open(refine_graph, "wb"))

#json.dump(sample_points, open(dataset_folder+"/region_%d_" % c + "refine_gt_graph_samplepoints.json", "w"), indent=2)

#c+=1

