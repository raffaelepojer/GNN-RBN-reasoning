import os.path as osp
import numpy as np
import torch
from gnn_to_rbn import *
import sys
sys.path.append('/Users/raffaelepojer/Dev/RBN-GNN/python/')
from gnn.ACR_graph import *

if __name__ == "__main__":

    model = MYACRGnnGraph(
            input_dim=5,
            hidden_dim=[4, 2],
            num_layers=2,
            mlp_layers=0,
            final_read="add"
        )
    
    model_name = "test"
    path = "/Users/raffaelepojer/Desktop/test-rbn/weights/"
    rbn_path = "/Users/raffaelepojer/Desktop/test-rbn/weights/rbn"
    print(model.layers[0].A.weight.shape[1])

    # torch.save(model.state_dict(), path+model_name+".pt")
    # model.export_parameters(path+model_name)
    # print model parameters
    # for name, param in model.named_parameters():
    #     print(name, param)

    for name, layer in model.named_parameters():
        print(name)
        print(layer.data)
        print('*'*10)

    # get a list of single letters of the alphabet long as the number given in input
    # e.g. input 5 -> output ['a', 'b', 'c', 'd', 'e']
    prob_name = "TEST"
    feature_names = [chr(i) for i in range(97, 97+5)]
    feature_probs = [0.5]*5

    write_rbn_ACR_graph(rbn_path, model, prob_name, feature_names, feature_probs)
    
