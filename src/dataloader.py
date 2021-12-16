import networkx as nx
import pandas as pd
import numpy as np


def load_weighted_edgelist(path, id_name_paths=None, allow_self_loops=False):
	"""Load graph from file in the following format:
		[src] [dst] [weight]

	Node ID names can be added by providing a list of paths to name files.
	These files must be in the format:
		[node_id] [label]

	Args:
		path (str): Path to file containing graph
		id_name_paths (list): List of paths to name files

	Returns:
		nx.Graph: Weighted undirected graph
	"""
	if id_name_paths is None:
		print("No ID labels found!")
		id_name_paths = []
	else:
		print(f"Found ID label files: {id_name_paths}")

	name_dict = {}
	for id_name_path in id_name_paths:
		name_dict.update(read_id_labels(id_name_path))

	print(f"Loading graph data from {path}...")

	data = np.loadtxt(path).astype(int)
	G = nx.Graph()
	for ite in data:
		for i in range(2):
			if not G.has_node(ite[i]): #If new source/target node --> set group (source = 0, target = 1) 
				name = name_dict.get(str(ite[i]), str(ite[i]))
				G.add_node(ite[i], group=i, name=name)

		if(G.has_edge(ite[0], ite[1])):
			G[ite[0]][ite[1]]["weight"] += ite[2]
		else:
			if allow_self_loops or ite[0] != ite[1]:
				G.add_edge(ite[0], ite[1], weight=ite[2])

	print(f"Read graph of total weight: {G.size(weight='weight')}")
	print(f"total nodes: {G.number_of_nodes()}")
	print(f"total edges: {G.number_of_edges()}")

	return G


def read_id_labels(filename):
	"""Reads ID labels from file

	Args:
		filename (str): Path to input file
	
	Returns:
		dict: Dictionary of IDs to string labels
	"""
	df = pd.read_csv(filename, header=None, usecols=[0, 1], delimiter='\t')
	df[1] = df[1].str.strip()
	df[0] = df[0].astype(str)
	return dict(df.values)
