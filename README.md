# GNN-RBN
Code for the paper "Generalized Reasoning with Graph Neural Networks by Relational Bayesian Network Encodings".

# Replicate the Results

To replicate the paper's results, follow these steps:

1. **Experiments Setup**:
   - Navigate to the Python directory.
   - Ensure required datasets are available, with Mutagenicity automatically downloaded.

2. **Running Experiments**:
   - Alpha Problem:
     - Run **main_kfold_blue.py**.
   - Synthetic Datasets:
     - Run **main_kfold_triangle.py**.
   - Mutagenicity:
     - Run **main_kfold_mutag.py**.

These scripts contain the code to replicate the paper's experiments. Ensure dataset availability and dependencies before execution.

## How to Work with Primula

More examples can be found in `examples-primula` in this repo.

Once you have generated the RBN, you can perform inference using Primula. In the **rdef_files** folder, you'll find base graphs that can be used with Primula. These .rdef files define the structure of a graph for the Primula system, including node types and their arguments. If you need to modify prior probabilities for relations in the RBN definition, you can typically find them at the top of the .rbn file.

To replicate examples using Primula, follow these steps:

1. **Download and Compile Primula**:
   - Download Primula from the original [GitHub repository](https://github.com/manfred-jaeger-aalborg/primula3).
   - The primary class is `RBNgui.Primula`. In the `loadDefaults()` function, you can specify the paths to the .rbn and .rdef files without using the GUI.

2. **Set Model and Data Sources**:
   - Use the Model source for the .rbn file.
   - Use the Data source for the .rdef file.

3. **View Graphs in Bavaria**:
   - Click on `Modules -> Bavaria` to visualize the graphs.
   - Activate both top toggles to display the Input structure and the Probabilistic Relations.

4. **Perform Inference**:
   - Access the Inference Module by clicking on `Modules -> Inference Module` for the inference part.

Following these steps will allow you to work with Primula and perform inference using the generated RBN, replicating examples from the paper.

  
## Replicate the MAP experiments with the Synthetic dataset:

### Training the GNN

1. **Run the main_kfold_triangle.py File**:
   - Execute the `main_kfold_triangle.py` file with the default settings.
   - Allow the script to complete its execution. It will train the GNN model and generate the corresponding RBN.

### Collecting Results

- The results are stored in a file named `results.txt`, representing the average of multiple experiments.
- Inside each experiment's folder, you will find a .rbn file that should be imported into Primula.
- The `rdef_files/synthetic` directory contains the rdef files for the experiments, ranging from 2-node graphs to 9-node graphs.

### Loading .rdn and .rdef Files

- Load the .rdn and .rdef files into Primula.
- Proceed to the Interface Module from the Modules menu.

### Setting Queries for Attributes or Binary Relations

Open the **Inference Module** to configure various Attributes or Binary relations (edges) and generate the most probable graph for a given class, using the MAP module.
   
1. **Attribute Queries**:
   - Click the **Query** button to enter in the "query mode", and select the attributes you want to query for each node. In the synthetic example, each node has 7 possible attributes [A, B, C, D, E, F, G]. In order to do this, click on one attribute (e.g. C) and click on the `[node*]` element in the **Element names** list. This assigns the selected Attributes to all nodes.
   - Repeat this process for all 7 attributes (Note: the Query button can only be clicked on the first time).
   
2. **Edge Queries**:
   - Assign all possible edge combinations to the nodes by selecting the edge attribute in the Binary relations list, and double-clicking again on the `[node*]` element (this requires to be in the "query mode", click on the Query button to enter in this mode).

### Selecting the Class

- **Choose the Class for Computation**:
   - Set the class for which you want to compute the most probable graph. In the compiled RBN, all positive classes are assigned to the **Arbitrary relation** `soft_max_0/0`, negative to `soft_max_1/0`.
   - Click the **True** button and then click on the `soft_max_0/0` relation to assign it as the true value.

### Computing MAP

1. **Perform MAP Computation**:
   - Start the MAP computation by clicking the **MAP** button and then clicking the **Settings MAP** button.
   - A new window will open, allowing you to set the number of restarts you want (-1 to stop manually or another value, e.g., 10).
   
2. **Executing MAP Computation**:
   - Click the Start button in the Inference Module and wait for the restarts to complete or manually stop the computation.
   - When the computation finishes, press the **Set MAP Vals** button to apply the computed values to the graph.

### Viewing Results

- **Viewing Computed Values**:
   - View the computed values from Bavaria.
   - Save these results in a .rdef file by clicking on the Primula main interface's **Save Data** option under the **Run** menu.

### Additional Analysis

- **Perform Additional Analysis**:
   - In the repository, there is a file called **test_prob.py** to parse the saved .rdef file and feed it into the trained model to check the probability.
   - Alternatively, you can perform this analysis using the **MCMC** module of Primula.


## Conda environment
The PyTorch version used was the 2.0.1 and torch-geometric 2.3.1

If one wants to have the exact same conda environment just use the file **environment.yml** in the repository.


`conda env create -f environment.yml`
`conda activate my-experiment-env`