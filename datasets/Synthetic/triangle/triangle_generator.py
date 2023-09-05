import networkx as nx
import matplotlib.pyplot as plt
import random
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import os
import os.path as osp
import scipy.sparse as sp
import pickle

def generate_class_triangle_base_0(min_node, max_node, m, p):
    n_nodes = np.random.randint(min_node, max_node+1)
    if random.uniform(0, 1) <= 0.5:
        graph = nx.barabasi_albert_graph(n_nodes, m)
    else:
        graph = nx.erdos_renyi_graph(n_nodes, p)
    # feature vector
    x = np.zeros((n_nodes, 7))
    for i in range(n_nodes):
        k = np.random.randint(0, high=7)
        x[i, k] = 1

    # choose a node to attach the first structure
    '''
    B == A == B
         |
     node_hook
    '''
    node_hook = np.random.randint(0, n_nodes)
    x = np.append(x, [[1,0,0,0,0,0,0]], axis=0) # A
    graph.add_edge(node_hook, n_nodes)
    A_idx = n_nodes
    n_nodes += 1

    # attach the two other nodes to the first added
    x = np.append(x, [[0,1,0,0,0,0,0]], axis=0) # B
    x = np.append(x, [[0,1,0,0,0,0,0]], axis=0) # B
    graph.add_edge(A_idx, n_nodes)
    graph.add_edge(A_idx, n_nodes+1) 
    n_nodes += 2

    # create the second structure
    '''
    D == C == D
         |
     node_hook
    '''
    node_hook = np.random.randint(0, n_nodes)
    x = np.append(x, [[0,0,1,0,0,0,0]], axis=0) # C
    graph.add_edge(node_hook, n_nodes)
    C_idx = n_nodes
    n_nodes += 1

    # attach the two other nodes to the first added
    x = np.append(x, [[0,0,0,1,0,0,0]], axis=0) # D
    x = np.append(x, [[0,0,0,1,0,0,0]], axis=0) # D
    graph.add_edge(C_idx, n_nodes)
    graph.add_edge(C_idx, n_nodes+1) 
    n_nodes += 2

    graph_label = torch.tensor([1])

    node_colors = [color for color in np.argmax(x, axis=1)]
    color_map = {0: "blue", 1: "green", 2: "red", 3: "yellow", 4: "purple", 5: "orange", 6: "gray"}
    node_colors_name = [color_map[color] for color in node_colors]

    # create torch geometric Data
    ne = len(graph.edges)
    torch_edge_index = torch.zeros((2, 2 * ne), dtype=torch.long)
    for i, e in enumerate(graph.edges):
        torch_edge_index[:, i] = torch.tensor(e)
        torch_edge_index[:, ne+i] = torch.tensor(tuple(reversed(e)))

    torch_x = torch.from_numpy(x).type(torch.FloatTensor)
    pyg_data = Data(x=torch_x, edge_index=torch_edge_index, graph_label=graph_label)

    return (graph, node_colors_name), pyg_data

def generate_class_triangle_base_1(min_node, max_node, m, p):
    n_nodes = np.random.randint(min_node, max_node+1)
    if random.uniform(0, 1) <= 0.5:
        graph = nx.barabasi_albert_graph(n_nodes, m)
    else:
        graph = nx.erdos_renyi_graph(n_nodes, p)
    while(True):
        x = np.zeros((len(graph.nodes()), 7))
        for node in graph.nodes():
            k = np.random.randint(low=0, high=7)
            x[node, k] = 1
        BB = False
        DD = False
        for node in graph.nodes():
            k = np.argmax(x[node], axis=0)
            if k == 0: # if A
                count_B = 0
                for neigh in graph.neighbors(node):
                    count_B += 1 if x[neigh, 1] == 1 else 0
                BB = True if count_B >= 2 else BB
            if k == 2: # if C
                count_D = 0
                for neigh in graph.neighbors(node):
                    count_D += 1 if x[neigh, 3] == 1 else 0
                DD = True if count_D >= 2 else DD
                
        if not (BB and DD): # redo if one present
            break

    # add a structure inside the graph either one OR the other one
    # if one structure is already present do not insert any structure
    if not(BB or DD) and random.uniform(0, 1) < 0.4:
        if random.uniform(0, 1) <= 0.5:
            # choose a node to attach the first structure
            '''
            B == A == B
                 |
             node_hook
            '''
            node_hook = np.random.randint(0, n_nodes)
            x = np.append(x, [[1,0,0,0,0,0,0]], axis=0) # A
            graph.add_edge(node_hook, n_nodes)
            A_idx = n_nodes
            n_nodes += 1

            # attach the two other nodes to the first added
            x = np.append(x, [[0,1,0,0,0,0,0]], axis=0) # B
            x = np.append(x, [[0,1,0,0,0,0,0]], axis=0) # B
            graph.add_edge(A_idx, n_nodes)
            graph.add_edge(A_idx, n_nodes+1) 
            n_nodes += 2
        else:
            # create the second structure
            '''
            D == C == D
                 |
             node_hook
            '''
            node_hook = np.random.randint(0, n_nodes)
            x = np.append(x, [[0,0,1,0,0,0,0]], axis=0) # C
            graph.add_edge(node_hook, n_nodes)
            C_idx = n_nodes
            n_nodes += 1

            # attach the two other nodes to the first added
            x = np.append(x, [[0,0,0,1,0,0,0]], axis=0) # D
            x = np.append(x, [[0,0,0,1,0,0,0]], axis=0) # D
            graph.add_edge(C_idx, n_nodes)
            graph.add_edge(C_idx, n_nodes+1) 
            n_nodes += 2

    # create torch geometric Data
    ne = len(graph.edges)
    torch_edge_index = torch.zeros((2, 2 * ne), dtype=torch.long)
    for node, e in enumerate(graph.edges):
        torch_edge_index[:, node] = torch.tensor(e)
        torch_edge_index[:, ne+node] = torch.tensor(tuple(reversed(e)))

    node_colors = [color for color in np.argmax(x, axis=1)]
    color_map = {0: "blue", 1: "green", 2: "red", 3: "yellow", 4: "purple", 5: "orange", 6: "gray"}
    node_colors_name = [color_map[color] for color in node_colors]
    graph_label = torch.tensor([0])

    torch_x = torch.from_numpy(x).type(torch.FloatTensor)
    pyg_data = Data(x=torch_x, edge_index=torch_edge_index, graph_label=graph_label)

    return (graph, node_colors_name), pyg_data


def generate_generic_graph(min_node, max_node, m, p):
    n_nodes = np.random.randint(min_node, max_node+1)
    if random.uniform(0, 1) <= 0.5:
        graph = nx.barabasi_albert_graph(n_nodes, m)
    else:
        graph = nx.erdos_renyi_graph(n_nodes, p)
    
    x = np.zeros((len(graph.nodes()), 7))
    for node in graph.nodes():
        k = np.random.randint(low=0, high=7)
        x[node, k] = 1

    BB = False
    DD = False
    for node in graph.nodes():
        k = np.argmax(x[node], axis=0)
        if k == 0: # if A
            count_B = 0
            for neigh in graph.neighbors(node):
                count_B += 1 if x[neigh, 1] == 1 else 0
            BB = True if count_B >= 2 else BB
        if k == 2: # if C
            count_D = 0
            for neigh in graph.neighbors(node):
                count_D += 1 if x[neigh, 3] == 1 else 0
            DD = True if count_D >= 2 else DD
    
    label = 1 # 1 negative, 0 positive
    if (BB and DD): # redo if one present
        label = 0

    # create torch geometric Data
    ne = len(graph.edges)
    torch_edge_index = torch.zeros((2, 2 * ne), dtype=torch.long)
    for node, e in enumerate(graph.edges):
        torch_edge_index[:, node] = torch.tensor(e)
        torch_edge_index[:, ne+node] = torch.tensor(tuple(reversed(e)))

    node_colors = [color for color in np.argmax(x, axis=1)]
    color_map = {0: "blue", 1: "green", 2: "red", 3: "yellow", 4: "purple", 5: "orange", 6: "gray"}
    node_colors_name = [color_map[color] for color in node_colors]
    graph_label = torch.tensor([label])

    torch_x = torch.from_numpy(x).type(torch.FloatTensor)
    pyg_data = Data(x=torch_x, edge_index=torch_edge_index, y=graph_label)

    return (graph, node_colors_name), pyg_data

def nx_to_sparse_block_diagonal(G_list):
    block_list = []
    for G in G_list:
        block_list.append(nx.to_scipy_sparse_array(G, weight=None, format='csr'))

    return block_list

def write_dataset(num_positive, num_negative, batch_size=64, train=0.7, valid=0.1, min_node=5, max_node=10, verbose=True):
    tr_dr = osp.dirname(osp.realpath(__file__))
    if not os.path.exists(tr_dr):
        os.mkdir(tr_dr)

    # generate positive graphs
    graph_list = []
    count = 0
    tmp_pos = num_positive
    tmp_neg = num_negative
    while((num_positive + num_negative) > 0):
        graph_p, data = generate_generic_graph(min_node=min_node, max_node=max_node, m=3, p=0.3)
        if data.y.item() == 0 and num_positive > 0:
            count += 1
            graph_list.append(data)
            # graph_draw(graph_p, data)
            if count % 100 == 0 and verbose == True:
                print("remaining pos:", num_positive)
            num_positive -= 1
        if data.y.item() == 1 and num_negative > 0:
            count += 1
            graph_list.append(data)
            # graph_draw(graph_p, data)
            if count % 100 == 0 and verbose == True:
                print("remaining neg:", num_negative)
            num_negative -= 1
    
    random.shuffle(graph_list)
    with open(tr_dr + '/graph_list_' + str(min_node) + '_' + str(max_node) + '_' + \
              str(tmp_pos) + '_' + str(tmp_neg) + '.pt', 'wb') as f:
        pickle.dump(graph_list, f)


def graph_draw(graph, data):
    labels = {}
    node_labels = [label.item() for label in np.argmax(data.x, axis=1)]
    labels_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}
    node_map_name = [labels_map[label] for label in node_labels]
    for node, label in zip(list(graph[0].nodes), node_map_name):
        labels[node] = label

    nx.draw_networkx(graph[0], pos=nx.spring_layout(graph[0]), labels=labels, node_color=graph[1])
    plt.show()

if __name__ == "__main__":

    write_dataset(7000, 7000, batch_size=64, train=0.7, valid=0.1, min_node=5, max_node=10, verbose=True)
