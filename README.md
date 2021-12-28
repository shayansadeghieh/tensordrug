# tensordrug
This is a tensorflow implementation of torchdrug (https://github.com/DeepGraphLearning/torchdrug). 

Removing barriers to nice packages (such as the package not being written in your deep learning framework of choice) is important. Less barriers means more solutions.

You use this package the same way as Torchdrug. The only difference being the underlying code behind the library has been converted from torch to tensorflow.

```
from tensordrug.data.graph import Graph

edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
graph = Graph(edge_list, num_nodes=6)
subgraph = graph.subgraph([2, 3, 4])

```
