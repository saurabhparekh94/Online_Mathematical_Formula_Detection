"""
File Name: DiGraph.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""
import networkx as nx

'''
the code generates a dictionary of dictionary for maximumm weight of traversal and is still undirected.

'''
def edmonds(matrix):

    number_of_node = []
    nodes=matrix[0]
    length = len(nodes)
    for x in range(length):
        number_of_node.append(x)

    G = nx.DiGraph()
    G.add_nodes_from(number_of_node)

    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y] != 0:
                G.add_weighted_edges_from([(x, y, matrix[x][y])])

    Edmond = nx.tree.Edmonds(G)
    output = Edmond.find_optimum(attr='weight', kind='max')

    nodes = output.nodes()
    graph_as_adjacency = nx.to_dict_of_dicts(output, nodes)

    return graph_as_adjacency



