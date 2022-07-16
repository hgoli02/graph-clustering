from attr import define
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
GRAPH_SIZE = 100 
def show_graph(adjacency_matrix, labels=None, node_size=500):
    color_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'yellow'}
    colors = [color_map[x] for x in labels] if labels is not None else None
        
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=node_size, node_color=np.array(colors)[list(gr.nodes)] if labels is not None else None)
    plt.show()


def find_eigen_val_vec(mat, threshold):
    landa, v = np.linalg.eig(mat)
    landa = [0 if x < threshold else x for x in landa]
    landa = np.array(landa)
    indecies = np.argsort(landa)
    landa = landa[indecies[::1]]
    v = (v.T[indecies[::1]]).T
    return landa, v


adj = np.zeros((GRAPH_SIZE, GRAPH_SIZE))
file1 = open('data/data.txt', 'r')
lines = file1.readlines()
print(len(lines))
for l in lines[1:]:
    i, j = l.split()
    adj[int(i) - 1, int(j) - 1] = 1
    adj[int(j) - 1, int(i) - 1] = 1

sum_matrix = np.sum(adj, axis= 1)
lapl = np.diag(sum_matrix) - adj
val, vec =  find_eigen_val_vec(lapl, 1e-12)
vec1 = vec[:, 1]
vec2 = vec[:, 2]

cluster1 = [1 if x < 0 else 2 for x in vec1]
cluster2 = [1 if x > 0 and y > 0 else (2 if x > 0 and y < 0 else (3 if x < 0 and y > 0 else 4)) for x,y in zip(vec1,vec2)]

show_graph(adj, cluster1)
show_graph(adj, cluster2)