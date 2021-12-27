from tensordrug.data.graph import Graph

edge_list = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]
graph = Graph(edge_list, num_nodes=6)

print("graph created")
print(graph)

subgraph = graph.subgraph([2,3,4])
print(subgraph)