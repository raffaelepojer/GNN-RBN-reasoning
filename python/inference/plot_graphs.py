import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import os
import json
import matplotlib.pyplot as plt
import hashlib

def nudge(pos, x_shift, y_shift):
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

def generate_random_hex_color(seed):
    # Convert the seed to bytes and hash it
    hashed_seed = hashlib.sha256(str(seed).encode()).digest()
    # Extract the first three bytes and convert to hexadecimal format
    hex_color = "#{:02x}{:02x}{:02x}".format(hashed_seed[0], hashed_seed[1], hashed_seed[2])
    return hex_color

def assign_colors_to_nodes(graph, color_mapping):
    # Iterate through each node in the graph
    for node in graph.nodes():
        # Get the node type (e.g., "Ca", "F", "I", etc.)
        node_type = graph.nodes[node]['label']
        # print(color_mapping)s
        # Use the color_mapping to get the color for this node type
        color = color_mapping.get(node_type, 'black')  # Default to black if node type not in the color mapping
        # Set the 'color' attribute for the node
        graph.nodes[node]['color'] = color

def create_graph(rbn_path, out_folder, color_mapping):
    # Parse the XML file
    tree = ET.parse(rbn_path)
    root = tree.getroot()

    # Extract node information
    nodes = {}
    for obj in root.findall('./Data/DataForInputDomain/Domain/obj'):
        ind = obj.attrib['ind']
        name = obj.attrib['name']
        nodes[name] = ind

    # Extract edge information with a value of 'true'
    edges = []
    for d in root.findall('./Data/DataForInputDomain/ProbabilisticRelsCase/d[@rel="edge"][@val="true"]'):
        args = d.attrib['args'][1:-1].split(')(')
        for arg in args:
            node1, node2 = arg.split(',')
            edges.append((node1.strip(), node2.strip()))  # Remove leading/trailing spaces

    # Create the graph using NetworkX with directed edges
    G = nx.DiGraph()

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


    # pos = nx.spring_layout(G, seed=1, k=0.40, iterations=30, center=(0.5,0.5))
    # pos = nudge(pos, 0, 0.2)
    
    pos = nx.shell_layout(G)

    # node_colors = [color_mapping[G.nodes[node]['label']] for node in G.nodes]

    assign_colors_to_nodes(G, color_mapping)
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    if len(G.nodes()) > 0:
        # Draw the graph using matplotlib
        # plt.figure(figsize=(9.4, 7.8))
        plt.figure()
        # node_labels = nx.get_node_attributes(G, 'label')
        # nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=700, font_size=15, font_weight='bold', arrowstyle='-', arrowsize=15, edge_color='black', width=1.3, alpha=1)
        
        # Update the label positions outside the nodes
        label_pos = {node: (x, y + 0.1) for node, (x, y) in pos.items()}  # Adjust the y-coordinate as needed

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=False, labels=node_labels, node_color=node_colors, node_size=1500, font_size=8, font_weight='bold', arrowstyle='-', arrowsize=15, edge_color='black', width=3.3, alpha=1)

        # Draw the labels outside the nodes
        # nx.draw_networkx_labels(G, label_pos, font_size=15, labels=node_labels, font_weight='bold', font_color='black')


        plt.axis('off')
        plt.savefig(out_folder + "/" + "n" + str(len(G.nodes())) + "-graph.png", dpi=300)
        # plt.show()

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

if __name__ == "__main__":
    rdef_folder =  "/Users/raffaelepojer/Dev/RBN-GNN/models/triangle_10_8_6_20230725-152135/exp_41/graphs/10_restarts/1_graphs/edge_050/"
    output_folder =  "/Users/raffaelepojer/Dev/RBN-GNN/models/triangle_10_8_6_20230725-152135/exp_41/graphs/10_restarts/1_graphs/edge_050/"

    rdef_paths = [os.path.join(rdef_folder, file) for file in os.listdir(rdef_folder) if os.path.splitext(file)[1] == '.rdef']

    rdef_paths.sort()

    rdefs = load_rbns(rdef_paths)
    color_mapping = get_unique_labels(rdefs, "/Users/raffaelepojer/Dev/RBN-GNN/models/triangle_10_8_6_20230725-152135/exp_41/color.json")
    
    print(color_mapping)

    for rdef in rdefs:
        create_graph(rdef, output_folder, color_mapping)