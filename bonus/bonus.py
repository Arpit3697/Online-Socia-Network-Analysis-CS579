import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    nghbors = set(graph.neighbors(node));
    scores = []
    for n in graph.nodes():
      if ((n not in nghbors) & (n != node)):
        nghbors2 = set(graph.neighbors(n))      
        nume=0
        denom_1=0
        denom_2=0
        for i in (nghbors & nghbors2):
          nume+=(1/graph.degree(i))      
        for i in graph.neighbors(node):
          denom_1+=(graph.degree(i))      
        for i in graph.neighbors(n):
          denom_2+=(graph.degree(i))                
        scores.append(((node,n),nume/((1/denom_1)+(1/denom_2))))
    return sorted(scores, key=lambda x: (-x[1],x[0][1]))
    pass