# tensordrug :turtle:
This is a tensorflow implementation of torchdrug (https://github.com/DeepGraphLearning/torchdrug). 

Removing barriers to nice packages (such as the package not being written in your deep learning framework of choice) is important. Less barriers means more solutions.

You use this package the same way as Torchdrug. The only difference being the underlying code behind the library has been converted from torch to tensorflow.

```
from tensordrug.data.graph import Graph

edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
graph = Graph(edge_list, num_nodes=6)
subgraph = graph.subgraph([2, 3, 4])

```

```
mol = Molecule.from_smiles("C1=CC=CC=C1")

print(mol.node_feature)
print(mol.atom_type)
print(mol.to_scaffold())
```
The output should look something like:
```
tf.Tensor(
[[ 0.0000000e+00  0.0000000e+00  1.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  1.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  1.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  1.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   1.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00
   0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
   1.0000000e+00  1.0000000e+00  1.5000000e+00  0.0000000e+00
   0.0000000e+00]
 [ 0.0000000e+00...(super long tensor)
 tf.Tensor([6 6 6 6 6 6], shape=(6,), dtype=int32)
 c1ccccc1
```
