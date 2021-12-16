import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit numpy threads to 1
import argparse
import csv

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.anomaly_detection import *
from src.dataloader import load_weighted_edgelist
from src.egonet_extractor import EgonetFeatureExtractor


def write_scores_to_file(nodes, filename):
	"""write tsv file with node and score
		[node name]<tab>[total score]

	Args:
		nodes (list): List of AnomalyScores
		filename (str): Output file name
	"""
	with open(filename, 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		for node in nodes:
			writer.writerow([node[1]['name'], node[1]['score']])
			

def get_scores_from_ids(graph, node_ids):
	"""Get scores from node ids
	
	Args:
		graph (nx.Graph): Graph
		node_ids (list): List of node ids

	Returns:
		list: List of anomaly scores for each node
	"""
	return [graph.nodes(data=True)[node_id]['score'] for node_id in node_ids]


def plot_node_and_scores(model, X, Y, node_ids, scores, nodes, xlabel, ylabel, path, no_anomalies=5):
	"""Plot results
	Hovering mechanic of plot inspired by https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-in-matplotlib

	Args:
		model (LinearRegression): Fitted linear regression model
		X (list): Sorted list of X features based on score
		Y (list): Sorted list of Y features based on score
		node_ids (list): Sorted List of node ids based on score
		scores (list): Sorted list of anomaly scores
		nodes (list): List of NetworkX nodes
		xlabel (str): X label name
		ylabel (str): Y label name
		path (str): Output path for plot
		no_anomalies (int): Number of top anomalies to plot as red dots
	"""
	alpha = model.coef_[0][0]
	beta = model.intercept_[0]
	top_scores_node_ids = node_ids[:no_anomalies+1]

	fig,ax = plt.subplots()
	fig.set_size_inches(14, 10, forward=True)

	# Plot points
	sc = plt.scatter(X, Y, cmap='viridis', c=scores, s=2, norm=matplotlib.colors.LogNorm())

	# Colour top k anomalies
	if no_anomalies > 0:
		X_top, Y_top = zip(*[(x, y) for x, y, node in zip(X, Y, node_ids) if node in top_scores_node_ids])
		plt.scatter(X_top, Y_top, s=5, color='red')

	# Plot power law line
	X = np.sort(np.array(X))
	plt.plot(X, np.exp(model.predict(np.expand_dims(np.log(X), axis=-1))), label=f"y = {alpha:.2f}x + ({beta:.2f})", color='red')

	# Enable annotation of nodes
	annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
						bbox=dict(boxstyle="round", fc="w"),
						arrowprops=dict(arrowstyle="->"))
	annot.set_visible(False)

	# Create hover function
	def update_annot(ind):
		pos = sc.get_offsets()[ind["ind"][0]]
		annot.xy = pos
	
		text = ""
		for n in ind["ind"]:
			text += f"Rank: {n} Score: {scores[n]}\n    {nodes[n]}\n"

		annot.set_text(text)
		annot.get_bbox_patch().set_alpha(0.4)

	def hover(event):
		vis = annot.get_visible()
		if event.inaxes == ax:
			cont, ind = sc.contains(event)
			if cont:
				update_annot(ind)
				annot.set_visible(True)
				fig.canvas.draw_idle()
			else:
				if vis:
					annot.set_visible(False)
					fig.canvas.draw_idle()

	fig.canvas.mpl_connect("motion_notify_event", hover)
	
	# Save and show figure
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.yscale("log")
	plt.xscale("log")
	plt.xlim(left=1)
	plt.ylim(bottom=1)
	plt.legend()
	plt.colorbar(sc, pad=0.01)

	plt.savefig(f"{path}.pdf")
	plt.show()

	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Perform OddBall anomaly detection")
	parser.add_argument("--graph", "-g", type=str, required=True, 
						help="Path to graph file")
	parser.add_argument("--out", "-o", type=str,
						help="output folder", default="./out")
	parser.add_argument("--processes", "-p", type=int, default=None,
						help="Number of processes to use")
	parser.add_argument('--lof', "-l", action="store_true", default=False,
						help='Use LOF. If left out LOF is not used.')
	parser.add_argument('--anomaly_type', "-a", type=str, required=True, choices=["sc", "hv", "de"], 
						help='Anomaly Type. sc:star_or_clique. hv:heavy_vicinity. de:dominant_edge.')
	parser.add_argument('--id_names', type=str, nargs='*', default=None,
						help='Path to labels for first edge list column in bipartite graph. If graph is unipartite, only the first file is used to match IDs')
	parser.add_argument('--group', type=int, default=-1,
						help="For bipartite graphs left/right group have id 0 and 1 respectively. -1 = analyze as unipartite")
	parser.add_argument('--force_features', action="store_true", default=False,
						help='Force feature extraction')
	parser.add_argument("--no_anomalies", "-n", type=int, default=5,
						help="Number of top anomalies to plot")
	parser.add_argument("--no_neighbors", "-k", type=int, default=5000,
						help="Number of neighbors to use for LOF")
	args = parser.parse_args()


	# Set up some files and folders
	dataset_name = os.path.splitext(os.path.basename(args.graph))[0]

	output_folder = os.path.join(args.out, dataset_name)
	output_file = os.path.join(output_folder, f"{dataset_name}_oddball_{args.anomaly_type}_lof={args.lof}_group={args.group}")

	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)

	feature_graph_path = os.path.join(output_folder, f"{dataset_name}_features.pkl")

	# Check if feature file exists. If not, extract features
	if not args.force_features and os.path.exists(feature_graph_path):
		graph = nx.read_gpickle(feature_graph_path)
	else:
		extractor = EgonetFeatureExtractor()
		graph = load_weighted_edgelist(args.graph, args.id_names)
		features = extractor.get_feature_vector(graph, args.processes)
		nx.write_gpickle(graph, feature_graph_path)

	# Choose anomaly detection method
	if not args.lof:
		print("No LOF score considered")
		if args.anomaly_type == "sc":
			print("Detecting Star/Cliques")
			detector = StarCliqueAnomalyDetection(graph)
			xlabel = "# nodes"
			ylabel = "# edges"
		elif args.anomaly_type == "hv":
			print("Detecting HeavyVicinities")
			detector = HeavyVicinityAnomalyDetection(graph)
			xlabel = "# edges"
			ylabel = "Total weight"
		elif args.anomaly_type == "de":
			print("Detecting DominantEdges")
			detector = DominantEdgeAnomalyDetection(graph)
			xlabel = "Total weight"
			ylabel = "Eigenvalue"
	else:
		print("Using LOF score")
		if args.anomaly_type == "sc":
			print("Detecting Star/Cliques")
			detector = StarCliqueLOFAnomalyDetection(graph, args.no_neighbors, args.processes)
			xlabel = "# nodes"
			ylabel = "# edges"
		elif args.anomaly_type == "hv":
			print("Detecting HeavyVicinities")
			detector = HeavyVicinityLOFAnomalyDetection(graph, args.no_neighbors, args.processes)
			xlabel = "# edges"
			ylabel = "Total weight"
		elif args.anomaly_type == "de":
			print("Detecting DominantEdges")
			detector = DominantEdgeLOFAnomalyDetection(graph, args.no_neighbors, args.processes)
			xlabel = "Total weight"
			ylabel = "Eigenvalue"

	# Run anomaly detection
	model, X, Y, node_ids = detector.detect_anomalies(args.group)
	
	# Sort everything based on total score
	node_ids, X, Y = zip(*[(sc, xs, ys) for (sc, xs, ys) in sorted(zip(node_ids, X, Y), reverse=True, key=lambda triplet : detector.graph.nodes(data=True)[triplet[0]]['score'])])
	scores = get_scores_from_ids(detector.graph, node_ids)
	nodes = [(node_id, detector.graph.nodes()[node_id]) for node_id in node_ids]

	# Write score results to tsv
	print(f"saving as{output_file}")
	write_scores_to_file(nodes, output_file + ".txt")

	# Write result graph to file
	nx.write_gpickle(detector.graph, f"{output_file}_out.pkl")
	
	# Round floats for readability
	for node in nodes:
		for key, value in node[1].items():
			if isinstance(node[1][key], float):
				node[1][key] = round(node[1][key], 2)

	# Plot results
	plot_node_and_scores(model, X, Y, node_ids, scores, nodes, xlabel, ylabel, output_file, args.no_anomalies)
