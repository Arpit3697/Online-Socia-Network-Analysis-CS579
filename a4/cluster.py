"""
cluster.py
"""

# Imports
import copy
import networkx as nx
import pickle


def girvan_newman(graph):



    graph_copy = graph.copy()
    connected_comp = []
    edge_len_betweness = nx.edge_betweenness_centrality(graph_copy)
    connected_comp = nx.connected_component_subgraphs(graph)
    cluster_comp = [a for a in connected_comp]
    new_value = sorted(edge_len_betweness.items(), key=lambda b:(-b[1],b[0][0],b[0][1]))
    random = 0
    while len(cluster_comp) ==1:
        graph_copy.remove_edge(*new_value[random][0])
        random += 1
        new_comp = nx.connected_component_subgraphs(graph_copy)
        cluster_comp = [a for a in new_comp]

    cluster_comp[0],cluster_comp[1]=cluster_comp[1],cluster_comp[0]
    return cluster_comp

    pass

def volume(nodes, graph):

    count = 0
    for e in graph.edges():
        if e[0] in nodes or e[1] in nodes:
            count += 1
    return count

def cut(S, T, graph):


    set_cut = 0
    for i in S:
        for j in T:
            if graph.has_edge(i,j):
                set_cut = set_cut + 1
    return set_cut
    pass

def norm_cut(S, T, graph):

    tem_graph = graph.copy()
    cut4set = cut(S,T,tem_graph)
    S_vol_set = float(volume(S,tem_graph))
    T_vol_set = float(volume(T,tem_graph))
    return float(cut4set)/S_vol_set + float(cut4set/T_vol_set)
    pass

def jaccard(graph, node, k):

    jaccard_coeff = []
    neighbor_node = set(graph.neighbors(node))
    node_new = graph.nodes()
    for n in node_new:
        if not graph.has_edge(node,n) and n != node:
            nn = set(graph.neighbors(n))
            jaccard = (len(neighbor_node.intersection(nn)))/(len(neighbor_node.union(nn)))
            jaccard_coeff.append(((node,n),jaccard))
    jaccard_sorted = sorted(jaccard_coeff, key = lambda val:(-val[1],val[0]))
    return jaccard_sorted[:k]
    pass





def main():

    graph = pickle.load(open('graph.pickle', 'rb'))
    clusters = girvan_newman(graph)
    #print(clusters)
    pickle.dump(clusters, open('clusters.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
