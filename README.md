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

Here the steps that are needed to start replicating the examples with Primula:
- Download and compile primula from the original [GitHub repository](https://github.com/manfred-jaeger-aalborg/primula3). The main class is RBNgui.Primula, in the function loadDefaults() you can write the path of the .rbn and .rdef files without using the gui
- The Model source is for the .rbn file and the Data source is for the .rdef file
- Click on Modules -> Bavaria to see the graphs. Click and activate both the top toggles to see the Input structure and the Probabilistic Relations.
- Click on Modules -> Inference Module for the inference part.
  
## Replicate the MAP experiments with the Synthetic dataset:

### Training the GNN

1. **Run the main_kfold_triangle.py File**:
   - Execute the `main_kfold_triangle.py` file with the default settings.
   - Allow the script to complete its execution. It will train the GNN model and generate the corresponding RBN.

### Collecting Results

- The results are stored in a file named `results.txt`, representing the average of multiple experiments.
- Inside each experiment's folder, you will find a .rbn file that should be imported into Primula.
- The [`rdef_files/synthetic`](https://github.com/raffaelepojer/RBN-GNN/tree/main/rdef_files/synthetic) directory contains the rdef files for the experiments, ranging from 2-node graphs to 9-node graphs.

### Loading .rdn and .rdef Files

- Load the .rdn and .rdef files into Primula.
- Proceed to the Interface Module.

### Setting Queries for Attributes or Binary Relations

1. **Configure Queries for Attributes and Binary Relations**:
   - In the Interface Module, configure various Attributes or Binary relations (edges).
   - To generate the most probable graph for a given class, utilize the MAP module.
   
2. **Attribute Queries**:
   - Click the **Query** button and select the attributes you want to query for each node.
   - In the synthetic example, each node has 7 possible attributes [A, B, C, D, E, F, G].
   - Click on one attribute and double-click on the `[node*]` element in the Element names list. This assigns the selected Attributes to all nodes.
   - Repeat this process for all 7 attributes (Note: the Query button can only be clicked on the first attempt).
   
3. **Edge Queries**:
   - Assign all possible edge combinations to the nodes by clicking the **Query** button, selecting the edge attribute in the Binary relations list, and double-clicking again on the `[node*]` element.

### Selecting the Class

- **Choose the Class for Computation**:
   - Set the class for which you want to compute the most probable graph.
   - In the compiled RBN, all positive classes are assigned to the Arbitrary relation `soft_max_0/0`.
   - Click the **True** button and then click on the `soft_max_0/0` relation to assign it as the true value.

### Computing MAP

1. **Perform MAP Computation**:
   - Start the MAP computation by clicking the MAP button and then clicking the **Settings MAP** button.
   - A new window will open, allowing you to set the number of restarts you want (-1 to stop manually or another value, e.g., 10).
   
2. **Executing MAP Computation**:
   - Click the Start button in the Inference Module and wait for the restarts to complete or manually stop the computation.
   - When the computation finishes, press the **Set MAP Vals** button to apply the computed values to the graph.

### Viewing Results

- **Viewing Computed Values**:
   - View the computed values from Bavaria.
   - Save these results in an .rdef file by clicking on the Primula main interface's **Save Data** option under the **Run** menu.

### Additional Analysis

- **Perform Additional Analysis**:
   - In the repository, there is a folder (FOLDER NAME) containing a Jupyter notebook to parse the saved .rdef file and feed it into the trained model to check the probability.
   - Alternatively, you can perform this analysis using the **MCMC** module of Primula.


## Conda environment
The PyTorch version used was the 2.0.1 and torch-geometric 2.3.1

If one wants to have the exact same conda environment just use the file **environment.yml** in the repository.


`conda env create -f environment.yml`
`conda activate my-experiment-env`