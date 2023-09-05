# GNN-RBN
Code for the paper PAPER NAME.

## Replicate the results:
In the python directory there are the main file that are the experiments presented in the paper. The dataset are in the repository. For the Mutagenicity dataset from TUDataset, it is automatically downloaded into the dataset folder when you run the experiment.
- For the experiments with the Alpha problem run **main_kfold_blue.py**. 
- For the synthetic dataset experiments run **main_kfold_triangle.py**. 
- For the Mutagenicity experiments run **main_kfold_mutag.py**.


All the python files with kfold will generate multiple experiments and will report the results in txt files. Each experiments will be inside a directory with the name of the seed used.

## How to work with Primula
With the generated RBN we can infer the same problem of the examples in the paper. In the folder **rdef_files** there are contained some base graphs to use in Primula. The .rdef file are just definition of a graph for the Primula system, each node is declared with its type and arguments it takes. If you want to change the prior probability for some relation in the RBN definition, change it directly to the .rbn file, usually are on the top.

## Conda environment
The PyTorch version used was the 2.0.1 and torch-geometric 2.3.1
If one wants to have the exact same conda environment just use the file **environment.yml** in the repository.
`conda env create -f environment.yml
conda env create -f environment.yml`