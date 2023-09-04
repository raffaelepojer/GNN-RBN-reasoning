import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import os
import json
import matplotlib.pyplot as plt
import hashlib

def load_rbns(rbn_paths):
    rbns = []
    for file_path in rbn_paths:
        if file_path.lower().endswith(('.rdef')):
            rbns.append(file_path)
    return rbns

def save_color_mapping_to_txt(color_mapping, file_path):
    with open(file_path, 'w') as f:
        json.dump(color_mapping, f)

def load_color_mapping_from_txt(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
def generate_random_hex_color(seed):
    # Convert the seed to bytes and hash it
    hashed_seed = hashlib.sha256(str(seed).encode()).digest()
    # Extract the first three bytes and convert to hexadecimal format
    hex_color = "#{:02x}{:02x}{:02x}".format(hashed_seed[0], hashed_seed[1], hashed_seed[2])
    return hex_color

def assign_shapes_to_nodes(graph):
    for node in graph.nodes():
        node_name = graph.nodes[node].get('label', '')  # Get the name of the node
        shape = 'd' if node_name == 'alpha1' else 'o'  # Use 'd' for diamond if node_name is 'alpha1', 'o' otherwise
        graph.nodes[node]['shape'] = shape

def get_unique_labels(rbns, color_mapping_file=None):
    # Step 1: Get unique labels from the nodes across all graphs
    all_labels = set()
    for rbn in rbns:
        # Parse the XML file to get the labels from each graph
        tree = ET.parse(rbn)
        root = tree.getroot()
        for d in root.findall(f'./Data/DataForInputDomain/ProbabilisticRelsCase/d[@val="true"]'):
            args = d.attrib['args'][1:-1].split(')(')
            for node in args:
                if (not "," in node) and (node != ''):
                    all_labels.add(d.attrib['rel'])

    # Step 2: Load the existing color_mapping from the file if provided
    if color_mapping_file:
        color_mapping = load_color_mapping_from_txt(color_mapping_file)
    else:
        color_mapping = {}

    # Step 3: Create a mapping of label to color (ensure consistent color assignment across all graphs)
    unique_labels = sorted(list(all_labels))

    colors = [color for color in color_mapping.values() if color[0]=="#"]  # Get existing numeric colors
    existing_labels = [label for label, color in color_mapping.items() if color in colors]

    color_mapping = {label: color_mapping.get(label, '') for label in existing_labels}  # Keep existing colors

    # Assign new colors to new labels
    new_labels = [label for label in unique_labels if label not in existing_labels]
    for i, label in enumerate(new_labels, len(existing_labels)):
        color_mapping[label] = generate_random_hex_color(i)

    # Step 4: Save the final color_mapping to the file
    if color_mapping_file:
        save_color_mapping_to_txt(color_mapping, color_mapping_file)

    return color_mapping

def assign_colors_to_nodes(graph, color_mapping):
    # Iterate through each node in the graph
    for node in graph.nodes():
        # Get the node type (e.g., "Ca", "F", "I", etc.)
        node_type = graph.nodes[node].get('label', '')  # Get the label if it exists, otherwise use an empty string
        # Use the color_mapping to get the color for this node type
        color = color_mapping.get(node_type, '#ffffff')  # Default to gray if node type not in the color mapping
        # Set the 'color' attribute for the node
        graph.nodes[node]['color'] = color

        # Set the label to "?" for nodes without a label
        if not graph.nodes[node].get('label'):
            graph.nodes[node]['label'] = '?'

def create_graph(rbn_path, out_folder, color_mapping, saved_node_positions):
    # Parse the XML file
    tree = ET.parse(rbn_path)
    root = tree.getroot()

    # Extract node information
    nodes = {}
    for obj in root.findall('./Data/DataForInputDomain/Domain/obj'):
        ind = obj.attrib['ind']
        name = obj.attrib['name']
        nodes[name] = ind
    
    edges = []
    if len(root.findall('./Data/DataForInputDomain/PredefinedRels/d[@rel="edge"][@val="true"]')) > 0:
        for d in root.findall('./Data/DataForInputDomain/PredefinedRels/d[@rel="edge"][@val="true"]'):
            args = d.attrib['args'][1:-1].split(')(')
            for arg in args:
                node1, node2 = arg.split(',')
                edges.append((node1.strip(), node2.strip()))  # Remove leading/trailing spaces

    elif len(root.findall('./Data/DataForInputDomain/ProbabilisticRelsCase/d[@rel="edge"][@val="true"]')) > 0:
        for d in root.findall('./Data/DataForInputDomain/ProbabilisticRelsCase/d[@rel="edge"][@val="true"]'):
            args = d.attrib['args'][1:-1].split(')(')
            for arg in args:
                node1, node2 = arg.split(',')
                edges.append((node1.strip(), node2.strip())) 
    
    G = nx.DiGraph()

    # Add nodes to the graph with labels from PredefinedRels
    # for d in root.findall(f'./Data/DataForInputDomain/PredefinedRels/d[@val="true"]'):
    #     label = ""
    #     args = d.attrib['args'][1:-1].split(')(')
    #     for node in args:
    #         if (not "," in node) and (node != ''):
    #             label = node  # Use the node name as the label
    #             G.add_node(node, label=label)
    
    # Add nodes to the graph with labels from ProbabilisticRelsCase
    for d in root.findall(f'./Data/DataForInputDomain/ProbabilisticRelsCase/d[@val="true"]'):
        label = ""
        args = d.attrib['args'][1:-1].split(')(')
        tex = ""
        for node in args:
            # nodes needs to be (n) and not empty ()
            if (not "," in node) and (node != ''):
                tex += node + " "
                if not "const" in d.attrib['rel']:
                    label = d.attrib['rel']
                    G.add_node(node, label=label.strip())

    # Add edges to the graph
    G.add_edges_from(edges)

    if saved_node_positions is None:
        # Perform initial graph layout and save node positions
        # pos = nx.spring_layout(G, seed=190) #30 190
        pos = nx.shell_layout(G)
        saved_node_positions = pos.copy()
    else:
        # Use saved node positions for consistent layout
        pos = saved_node_positions

    # Extract the base name of the input XML file
    xml_file_base_name = os.path.splitext(os.path.basename(rbn_path))[0]

    # Generate the output file name based on the base name of the input XML file
    output_file_name = os.path.join(out_folder, f"{xml_file_base_name}-graph.png")

    assign_shapes_to_nodes(G) 

    # Rest of the function remains the same
    assign_colors_to_nodes(G, color_mapping)
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    if len(G.nodes()) > 0:
        # Draw the graph using matplotlib
        plt.figure()
        node_shapes = [G.nodes[node]['shape'] for node in G.nodes()]  # Get shapes for each node
        print(node_shapes)
        node_labels = nx.get_node_attributes(G, 'label')
        non_empty_node_labels = {node: label for node, label in node_labels.items() if label}
        fig, ax = plt.subplots(figsize=(8,6))
        nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=15, width=1.8)
        # nx.draw_networkx_labels(G, pos, font_size=14)

        for shape in set(node_shapes):
            # The nodes with the desired shapes
            node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=node_list,
                                   node_color=[G.nodes[node]['color'] for node in node_list],
                                   node_shape=shape,
                                   node_size=1300,
                                   edgecolors='black',  # Set the border color for the nodes
                                   linewidths=1)

        # nx.draw(G, pos, with_labels=True, labels=non_empty_node_labels, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrowstyle='-', arrowsize=15, edge_color='gray', width=1.3, alpha=1, node_shape=node_shapes)  # Pass node_shapes to the nx.draw function
        plt.margins(0.1)
        plt.axis('off')
        plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
        # plt.show()

        return saved_node_positions

if __name__ == "__main__":
    rdef_folder =  "/Users/raffaelepojer/Dev/RBN-GNN/datasets/alpha/p1_a_small/log_examples/edges"
    output_folder =  "/Users/raffaelepojer/Dev/RBN-GNN/datasets/alpha/p1_a_small/log_examples/edges"

    rdef_paths = [os.path.join(rdef_folder, file) for file in os.listdir(rdef_folder) if os.path.splitext(file)[1] == '.rdef']
   
    rdef_paths.sort(reverse=True)
    print(rdef_paths)

    rdefs = load_rbns(rdef_paths)
    color_mapping = get_unique_labels(rdefs, "/Users/raffaelepojer/Dev/RBN-GNN/datasets/alpha/p1_a_small/graphs_images/color.json")
    
    print(color_mapping)

    saved_node_positions = None  # Initialize to None

    for rdef in rdefs:
        saved_node_positions = create_graph(rdef, output_folder, color_mapping, saved_node_positions)
