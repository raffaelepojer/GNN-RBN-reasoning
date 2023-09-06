############################################################################################################
# This file cab be used as an example of how to compute the probability of a generated graph from Primula. #
############################################################################################################

import torch
import torch.nn as nn
import torch.functional as F
import re
import itertools
from gnn.ACR_graph import *

def extract_edges(file_path):
    file = open(file_path, "r")
    lines = []
    for x in file:
        lines.append(x.strip())

    lambda1 = lambda s: int(str(s[-1]))
    features = []
    first = 0
    for i, l in enumerate(lines):
        if l == "<ProbabilisticRelsCase>":
            first = i
        if i > first and "edge" in l and "true" in l:
            for match in re.findall(r'(?<=\().*?(?=\))',l):
                a,b = map(lambda1, match.split(','))
                features.append([a, b])
                features.append([b, a])
            break
    features.sort()

    if len(features) > 0:
        return torch.tensor(list(k for k,_ in itertools.groupby(features))).t()
    else:
        a = torch.tensor([[]], dtype=torch.long)
        b = torch.tensor([[]], dtype=torch.long)
        edge_index = torch.cat((a, b), 0)
        return edge_index

def count_nodes(file_path):
    file = open(file_path, "r")
    lines = []
    for x in file:
        lines.append(x.strip())
    lambda1 = lambda s: int(s[-1])
    nodes = []
    first = 0
    for i, l in enumerate(lines):
        if l == "<PredefinedRels>":
            first = i
        if i > first and "node" in l and "true" in l:
            regex = r"(?<=\()[a-zA-Z0-9_]+(?=\))"
            match = re.findall(regex, "".join(l))
            nodes = list(map(lambda1, match))
    return len(nodes)

def extract_features(file_path):
    features_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    file = open(file_path, "r")
    lines = []
    for x in file:
        lines.append(x.strip())

    lambda1 = lambda s: int(s[-1])
    all_features = []
    num_nodes = count_nodes(file_path)
    for i in range(num_nodes):
        all_features.append([0.0]*7)

    first = 0
    for i, l in enumerate(lines):
        if l == "<ProbabilisticRelsCase>":
            first = i
        
        match = re.search(r'rel="([^"]*)"', l)
        for j, feature in enumerate(features_name):
            if i > first and match and match.group(1) in features_name and match.group(1) in feature and "true" in l:
                regex = r"(?<=\()[a-zA-Z0-9_]+(?=\))"
                match = re.findall(regex, "".join(l))
                nodes = list(map(lambda1, match))
                for idx in nodes:
                    all_features[idx][j] = 1.0
                break
            
    return(all_features)

def compute_prob():
    # use the same dimensions as in the trained model used
    model = MYACRGnnGraph(
        input_dim=7,
        hidden_dim=[10, 8, 6],
        num_layers=3,
        mlp_layers=0,
        final_read="add",
        num_classes=2
    )

    model.load_state_dict(torch.load(f"PATH-GNN/rbn_acr_graph_triangle_10_8_6_add.pt"))
    model.eval()

    rdef_path = f"PATH-RDEF/n6.rdef"

    edge_index = torch.tensor(extract_edges(rdef_path))

    x = torch.tensor(extract_features(rdef_path))
    batch = torch.zeros(x.size()[0]).type(torch.LongTensor)

    out = model(x, edge_index, batch)
    m = nn.Softmax(dim=1)
    print("Out probability: ", m(out))
    print("The probability for the positive class is: ", m(out)[0][0].item())


if __name__ == "__main__":
    compute_prob()