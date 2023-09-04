# create the .rdef for the Primula graph files
import os.path as osp
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

def write_relations(feature_name, constraint=False, close_world=False):
    result = "\t<Relations>\n"
    result += "\t\t<Rel name=\"node\" arity=\"1\" argtypes=\"Domain\" valtype=\"boolean\" default=\"false\" type=\"predefined\" color=\"(100,100,100)\"/>\n"
    if close_world:
        result += "\t\t<Rel name=\"edge\" arity=\"2\" argtypes=\"node,node\" valtype=\"boolean\" default=\"false\" type=\"probabilistic\" color=\"(0,70,0)\"/>\n"
    else:
        result += "\t\t<Rel name=\"edge\" arity=\"2\" argtypes=\"node,node\" valtype=\"boolean\" default=\"?\" type=\"probabilistic\" color=\"(0,70,0)\"/>\n"
    
    for n_feature in feature_name:
        if close_world:
            result += \
                "\t\t<Rel name=\"" + n_feature + "\" arity=\"1\" argtypes=\"node\" valtype=\"boolean\" default=\"false\" type=\"probabilistic\" color=\"(" + str(random.randint(0, 254)) + \
                    "," + str(random.randint(0, 254)) + "," + str(random.randint(0, 254)) + ")\"/>\n"
        else:
            result += \
                "\t\t<Rel name=\"" + n_feature + "\" arity=\"1\" argtypes=\"node\" valtype=\"boolean\" default=\"?\" type=\"probabilistic\" color=\"(" + str(random.randint(0, 254)) + \
                    "," + str(random.randint(0, 254)) + "," + str(random.randint(0, 254)) + ")\"/>\n"
    
    if constraint:
        for i, n_feature1 in enumerate(feature_name):
            for n_feature2 in range(i, len(feature_name)):
                n_feature2 = feature_name[n_feature2]
                if n_feature1 != n_feature2:
                    result += "\t\t<Rel name=\"const_" + n_feature1 + "_" + n_feature2 + "\" arity=\"1\" argtypes=\"node\" valtype=\"boolean\" default=\"?\" type=\"probabilistic\"/>\n"

    result += "\t\t<Rel name=\"all_const\" arity=\"1\" argtypes=\"node\" valtype=\"boolean\" default=\"?\" type=\"probabilistic\"/>\n"
    result += "\t\t<Rel name=\"soft_max_0\" arity=\"0\" argtypes=\"\" valtype=\"boolean\" default=\"?\" type=\"probabilistic\"/>\n"
    result += "\t\t<Rel name=\"soft_max_1\" arity=\"0\" argtypes=\"\" valtype=\"boolean\" default=\"?\" type=\"probabilistic\"/>\n"
    result += "\t</Relations>\n"

    return result

def write_input_data(n_nodes):
    result = "\t\t\t<Domain>\n"

    angle = 0
    angle_step = math.pi*2 / n_nodes
    coordinate = []
    radius = 200
    translate = 700
    for i in range(n_nodes):
        coor = (math.cos(angle) * radius + translate, math.sin(angle) * radius + translate)
        result += "\t\t\t\t<obj ind='" + str(i) + "' name='n" + str(i) + "' coords='" + str(coor[0]) + "," + str(coor[1]) + "' />\n"
        angle += angle_step

    result += "\t\t\t</Domain>\n"
    return result

def write_pred_rels(n_nodes):
    result = "\t\t\t<PredefinedRels>\n"
    nodes = ""
    for i in range(n_nodes):
        nodes += "(n" + str(i) + ")"
    result += "\t\t\t\t<d rel=\"node\" args=\"" + nodes + "\" val=\"true\"/>\n"
    result += "\t\t\t</PredefinedRels>\n"
    return result

def write_prob_rels(feature_name, edge_index, node_feature, constraint=False, true_class=None):
    result = "\t\t\t<ProbabilisticRelsCase>\n"
    nodes = ""
    for i in range(len(node_feature)):
        nodes += "(n" + str(i) + ")"
    
    if constraint:
        for i, n_feature1 in enumerate(feature_name):
            for n_feature2 in range(i, len(feature_name)):
                n_feature2 = feature_name[n_feature2]
                if n_feature1 != n_feature2:
                    result += \
                        "\t\t\t\t<d rel=\"const_" + n_feature1 + "_" + n_feature2 + "\" args=\"" + nodes + "\" val=\"false\"/>\n"
        result += "\t\t\t\t<d rel=\"all_const\" args=\"" + nodes + "\" val=\"true\"/>\n"

    if len(edge_index) > 0:
        result += "\t\t\t\t<d rel=\"edge\" args=\""
        edge_l = ""
        for edge in edge_index:
            edge_l += "(n" + str(edge[0]) + ",n" + str(edge[1]) + ")"

        result += edge_l + "\" val=\"true\"/>\n"

    for i_type, node_n in enumerate(feature_name):
        found = False
        node_str = "\t\t\t\t<d rel=\"" + node_n + "\" args=\""
        for n_idx, node in enumerate(node_feature):
            for idx, feat in enumerate(node):
                if feat==1 and idx == i_type:
                    node_str += "(n" + str(n_idx) + ")"
                    found = True
        if found:
            node_str += "\" val=\"true\"/>\n"
            result += node_str

    if true_class is not None:
        result += "\t\t\t\t<d rel=\"" + true_class + "\" args=\"\" val=\"true\"\>\n" 


    result += "\t\t\t</ProbabilisticRelsCase>\n"
    return result

def write_data(feature_name, edge_index, node_features, constraint, true_class=None):
    result = "\t<Data>\n"
    result += "\t\t<DataForInputDomain>\n"

    result += write_input_data(len(node_features))
    result += write_pred_rels(len(node_features))
    result += write_prob_rels(feature_name, edge_index, node_features, constraint=constraint, true_class=true_class)

    result += "\t\t</DataForInputDomain>\n"
    result += "\t</Data>\n"
    return result

def write_rdef(filepath, feature_name, edge_index, node_feature, constraint=False, true_class=None):
    f = open(filepath, 'w')
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n<root>\n")
    close_world = len(edge_index) > 0
    f.write(write_relations(feature_name=feature_name, constraint=constraint, close_world=close_world))

    f.write(write_data(feature_name, edge_index, node_feature, constraint=constraint))

    f.write("</root>")
    print("RDEF written in {0}".format(filepath))
    f.close()


if __name__ == "__main__":
    num_nodes = 12
    save_dir = f"/Users/raffaelepojer/Dev/RBN-GNN/models/Mutagenicity_14_8_6_20230726-190823/exp_88/graphs/"
    rdef_name = "base_class_0_n" + str(num_nodes) + "_0.rdef"

    feature_name = ["C","O","Cl","H","N","F","Br","S","P","I","Na","K","Li","Ca"]
    # feature_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    edge_index = []
    edge_index_t = []
    for f1 in range(num_nodes):
        for f2 in range(f1, num_nodes):
            if f1 != f2:
                if random.uniform(0, 1) < 0.3:
                    edge_index.append([f1, f2])
                    edge_index_t.append([f1, f2])
                    edge_index_t.append([f2, f1])
    print(edge_index)

    node_feature = []
    for i in range(num_nodes):
        tmp = [0] * len(feature_name)
        idx_r = random.randint(0, 6)
        tmp[idx_r] = 1.0
        node_feature.append(tmp)
    print(node_feature)

    edge_index = []
    node_feature = [[]] * num_nodes 

    write_rdef(save_dir+rdef_name, feature_name, edge_index, node_feature, constraint=True, true_class=None)