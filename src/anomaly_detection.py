import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from src.utils import get_node_property_list, select_group_nodes


class BaseAnomalyDetection:
	"""Anomaly Detection based on outlier score
	"""

	def __init__(self, graph) -> None:
		"""Intialise the AnomalyDetection class

		Args:
			graph (nx.Graph): Networkx graph object to analyse.
		"""
		self.graph = graph


	def outlier_score(self, xi, yi, model):
		"""Caclulate the outlier score for a given data point

		Args:
			xi (float): X feature
			yi (float): Y feature
			model (LinearRegression): Fitted linear regression model

		Returns:
			float: Outlier score
		"""
		expected = np.exp(model.predict([[np.log(xi)]]))[0][0]
		score = (max(yi, expected) / (min(yi, expected))) * np.log(abs(yi-expected)+1)
		return score


	def calculate_outlier_scores(self, X, Y, normalise=True):
		"""Calculate outlier scores of all data points

		Args:
			X (list): X feature list
			Y (list): Y feature list

		Returns:
			list: List of outlier scores
			LinearRegression: Fitted linear regression model
		"""
		# Prepare data
		y_train = np.log(Y).reshape(len(Y), 1)
		x_train = np.log(X).reshape(len(X), 1)

		# Fit linear regression model
		model = LinearRegression()
		model.fit(x_train, y_train)

		alpha = model.coef_[0][0]
		beta = model.intercept_[0]

		print("Fitted power law:")
		print(f"\talpha: {alpha}")
		print(f"\tbeta: {beta}")

		outlier_scores = []
		for xi, yi in zip(X, Y):
			score = self.outlier_score(xi, yi, model)
			outlier_scores.append(score)
		
		# Normalise scores
		if normalise:
			outlier_scores = MinMaxScaler().fit_transform(np.array(outlier_scores).reshape(-1, 1)).flatten()

		return outlier_scores, model

	def calculate_total_score(self, X, Y, nodes):
		"""Calculate total score of all data points using outlier scores
		
		Args:
			X (list): X feature list
			Y (list): Y feature list
			nodes (list): List of NewtorkX nodes

		Returns:
			LinearRegression: Fitted linear regression model
			list: Filtered X feature list
			list: Filtered Y feature list
			list: Filtered node_ids
		"""
		lenfirst = len(X)
	
		# Filter nodes if their features contain 0 values
		X = np.array(X)
		Y = np.array(Y)
		mask = (Y > 0) & (X > 0)
		X = X[mask]
		Y = Y[mask]
		nodes = np.array(nodes)[mask]

		print(f"Filtered node list from: {lenfirst} to {len(X)}")

		# Calculate outlier scores
		outlier_scores, model = self.calculate_outlier_scores(X, Y)

		# Create score dict and set node attribute in graph
		score_dict = {k[0]: v for k, v in zip(nodes.tolist(), outlier_scores.tolist())}
		nx.set_node_attributes(self.graph, score_dict, "score")

		return model, X, Y, list(score_dict.keys())


	def detect_anomalies(self, group=-1):
		raise NotImplementedError("Please subclass!")


class BaseLOFAnomalyDetection(BaseAnomalyDetection):
	"""Anomaly Detection based on outlier score and LOF score
	"""

	def __init__(self, graph, n_neighbors=5000, processes=-1) -> None:
		"""Iniatialise the LOFAnomalyDetection class

		Args:
			graph (nx.Graph): Networkx graph object to analyse.
			n_neighbors (int): Number of neighbors to use for LOF
			processes (int): Number of processors to use for LOF
		"""
		super().__init__(graph)
		self.n_neighbors = n_neighbors
		self.processes = processes

	
	def calculate_LOF_score(self, X, Y, normalised=True):
		"""Calculate LOF score of all data points

		Args:
			X (list): X feature list
			Y (list): Y feature list

		Returns:
			list: List of LOF anomaly scores
		"""
		#prepare data for LOF
		y_train = Y
		x_train = X

		x_y_lof_data = np.array([x_train, y_train]).T # [[x_train1, y_train1], [x_train2, y_train2], ...]

		#LOF algorithm
		clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, n_jobs=self.processes)
		clf.fit_predict(x_y_lof_data)
		LOF_scores = -clf.negative_outlier_factor_

		# Normalise scores
		if normalised:
			LOF_scores = MinMaxScaler().fit_transform(np.array(LOF_scores).reshape(-1, 1)).flatten()

		return LOF_scores


	def calculate_total_score(self, X, Y, nodes):
		"""Calculate total score of all data points using outlier and LOF scores
		
		Args:
			X (list): X feature list
			Y (list): Y feature list
			nodes (list): Node list of type returned by nx.Graph.nodes(data=True) 

		Returns:
			LinearRegression: Fitted linear regression model
			list: Filtered X feature list
			list: Filtered Y feature list
			list: Filtered node_ids
		"""
		lenfirst = len(X)

		# Filter out nodes with 0 value
		X = np.array(X)
		Y = np.array(Y)
		mask = (Y > 0) & (X > 0)
		X = X[mask]
		Y = Y[mask]
		nodes = np.array(nodes)[mask]

		print(f"Filtered node list from: {lenfirst} to {len(X)}")

		# Calculate outlier scores and LOF scores
		outlier_scores, model = self.calculate_outlier_scores(X, Y)
		LOF_scores = self.calculate_LOF_score(X, Y)

		# Create score dict and set node attribute in graph
		score_dict = {}
		for node, outlier_score, LOF_score in zip(nodes, outlier_scores, LOF_scores):
			score_dict[node[0]] = {
				"outlier_score": outlier_score,
				"LOF_score": LOF_score,
				"score": outlier_score + LOF_score
			}
		nx.set_node_attributes(self.graph, score_dict)

		return model, X, Y, list(score_dict.keys())


class StarCliqueAnomalyDetection(BaseAnomalyDetection):
	def detect_anomalies(self, group):
		"""Detect star and clique anomalies: Nodes vs Edges

		Args:
			group (int): Group to analyse

		Returns:
			LinearRegression: Fitted linear regression model
			list: Filtered X feature list
			list: Filtered Y feature list
			list: Filtered node_ids
		"""
		nodes = select_group_nodes(self.graph, group)
		N, E = get_node_property_list(nodes, ["neighbors", "edges"])
		return self.calculate_total_score(N, E, nodes)	

class HeavyVicinityAnomalyDetection(BaseAnomalyDetection):
	"""Detect heavy-edge-centric anomalies: Total Weight vs Edges
	"""
	def detect_anomalies(self, group):
		"""Detect heavy-edge-centric anomalies: Total Weight vs Edges

		Args:
			group (int): Group to analyse

		Returns:
			LinearRegression: Fitted linear regression model
			list: Filtered X feature list
			list: Filtered Y feature list
			list: Filtered node_ids
		"""
		nodes = select_group_nodes(self.graph, group)
		E, W = get_node_property_list(nodes, ["edges", "weight"])
		return self.calculate_total_score(E, W, nodes)


class DominantEdgeAnomalyDetection(BaseAnomalyDetection):
	"""Detect dominant-edge-centric anomalies: Lambda vs Total Weight
	"""
	def detect_anomalies(self, group):
		"""Detect dominant-edge-centric anomalies: Lambda vs Total Weight
		
		Args:
			group (int): Group to analyse

		Returns:
			LinearRegression: Fitted linear regression model
			list: Filtered X feature list
			list: Filtered Y feature list
			list: Filtered node_ids
		"""
		nodes = select_group_nodes(self.graph, group)
		eigenvalues, W = get_node_property_list(nodes, ["eigenvalue", "weight"])
		return self.calculate_total_score(eigenvalues, W, nodes)


class StarCliqueLOFAnomalyDetection(BaseLOFAnomalyDetection, StarCliqueAnomalyDetection):
	pass


class HeavyVicinityLOFAnomalyDetection(BaseLOFAnomalyDetection, HeavyVicinityAnomalyDetection):
	pass


class DominantEdgeLOFAnomalyDetection(BaseLOFAnomalyDetection, DominantEdgeAnomalyDetection):
	pass
