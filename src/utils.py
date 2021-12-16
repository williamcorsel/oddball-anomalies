

def get_node_property_list(nodes, properties):
	"""Get list of a number of node properties fron a list of networkx nodes.

	Args:
		nodes (list): List of networkx format nodes.
		properties (list): List of string denoting the property keys to extract.

	Returns:
		list: List of lists of node properties. [[node1_prop1, node2_prop1, ...], [node1_prop2, node2_prop2, ...], ...]
	"""
	total_list = []

	for property in properties:
		property_list = [node[1][property] for node in nodes]
		total_list.append(property_list)

	return total_list

		

def select_group_nodes(graph, group):
	"""Select nodes from a networkx graph based on a group.

	Args:
		graph (networkx.Graph): Networkx graph object.
		group (str): Group name.
	
	Returns:
		list: List of networkx nodes that are of the specified group.
	"""
	nodes = [node for node in graph.nodes(data=True) if group < 0 or node[1]["group"] == group]
	return nodes
	