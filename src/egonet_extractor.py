import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
from tqdm import tqdm


def init_graph(_graph):
	"""For initializing the graph object when using multithreading
	"""
	global graph
	graph = _graph


class EgonetFeatureExtractor():
	def get_feature_vector(self, G, processes):
		"""Get feature vector for each node in the graph

		Args:
			G (nx.Graph): Weighted undirected graph
			processes (int): Number of processes to use

		Returns:
			nx.Graph: Weighted undirected graph with node features
		"""
		print("Finding features")
		nodes = [[x[0]] for x in G.nodes(data=True)]
		
		results = []
		begin = time.time()
		with Pool(processes=processes, initializer=init_graph, initargs=[G]) as pool:
			results = pool.starmap(self.find_node_features, tqdm(nodes))
		
		print(f"Time used for feature finding: {time.time() - begin}")
		node_results = {k: v for x in results for k, v in x.items()}

		nx.set_node_attributes(G, node_results)

		return G
	
	def find_node_features(self, node_id):
		"""Find features of a node

		Args:
			node_id (int): Node id

		Returns:
			dict: Node features (no_edges, no_neighbors, total_weight, eigenvalue)
		"""
		egonet = self.find_egonet(graph, node_id)
		
		# Number of edges in egonet i
		Ei = egonet.number_of_edges()

		# Number of neighbors of node i
		Ni = graph.degree(node_id)
		
		# Sum of weights of egonet i
		Wi = egonet.size(weight="weight")

		# Calculate the principal eigenvalue of egonet i's weighted adjacency matrix
		egonet_adjacency_matrix = nx.adjacency_matrix(
			egonet, weight="weight").todense()
		eigenvalue, _ = np.linalg.eig(egonet_adjacency_matrix)
		Lambda_w_i = np.abs(eigenvalue).max()

		node_features = {
			node_id : {
				"edges": Ei,
				"neighbors": Ni,
				"weight": Wi,
				"eigenvalue": Lambda_w_i
			}
		}
		return node_features

	
	def find_egonet(self, G, node_id, k=1):
		"""Find egonet of node

		Args:
			G (nx.Graph): Weighted undirected graph
			node_id (int): Node id
			k (int): Number of hops away from center node. Should be > 0

		Returns:
			nx.Graph: Egonet of node (Subgraph View)
		"""
		total_ids = [node_id]
		neighbors = [node_id]

		for _ in range(k):
			neighbors = [nbr for n in neighbors for nbr in G.neighbors(n)]
			total_ids.extend(neighbors)

		egonet_sub = G.subgraph(total_ids)
		
		return egonet_sub

