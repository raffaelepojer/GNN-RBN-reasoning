import random
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from utils.TU_dataset_reader import tud_to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import math
from gnn import *
from sklearn.model_selection import KFold
from datetime import datetime
from utils.early_stopper import EarlyStopper
from rbn.gnn_to_rbn import *
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter


def train(model, data, criterion, optimizer, device="cpu"):
    model.train()
    t = 0  # count of true predictions
    f = 0  # count of false predictions
    epoch_loss = 0
    for batch in data:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y
        l = criterion(out, target)
        _, pred = out.max(1)

        tt = len(np.where(target == pred)[0])
        t += tt
        f += len(target)-tt

        # l = criterion(out.reshape([-1]), target.float())

        epoch_loss += l.detach().item()
        l.backward()
        optimizer.step()

    accuracy = t/(t+f)
    avg_loss = epoch_loss/len(data)
    return avg_loss, accuracy


def test(model, data, criterion, device="cpu"):
    model.eval()
    t = 0
    f = 0
    epoch_loss = 0
    for batch in data:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y
        l = criterion(out, target)

        # l = criterion(out.reshape([-1]), target.float())
        epoch_loss += l.detach().item()
        _, pred = out.max(1)

        tt = len(torch.where(target == pred)[0])
        t += tt
        f += len(target)-tt

    accuracy = t/(t+f)
    avg_loss = epoch_loss / len(data)
    return avg_loss, accuracy


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(
        args,
        train_data=None,
        test_data=None,
        save_gnn_model=None,
        save_rbn_model=None,
        plot=None):
    pass

def train_test(seed,
               hidden_dim,
               max_epochs,
               final_readout,
               lr,
               BATCH_SIZE,
               k,
               save_logs,
               ds_name,
               mlp_layers,):
    
    seed_everything(seed)
    device = torch.device("cpu")

    # ACR-GNN definition
    hidden_dim = hidden_dim
    gnn_layers = len(hidden_dim)
    mlp_layers = mlp_layers
    # feature_dim = 7

    max_epochs = max_epochs
    final_readout = final_readout
    lr = lr
    min_acc = 120.0

    # Dataset definition
    BATCH_SIZE = BATCH_SIZE
    k = k
    
    save_logs = save_logs
    
    ds_name = ds_name

    gnn_layers_string = print_list_with_underscores(hidden_dim)
    experiment_path = f"/Users/raffaelepojer/Dev/RBN-GNN/models/" + ds_name + \
        "_" + gnn_layers_string + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    file_stem = "RBN_acr_graph_" + ds_name

    tensorboard_path = experiment_path + "tensorboard/"
    if save_logs:
        create_folder_if_not_exists(experiment_path)
        create_folder_if_not_exists(tensorboard_path)
        writer = SummaryWriter(log_dir=tensorboard_path)

    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '../', 'datasets')
    dataset = TUDataset(path, name=ds_name).shuffle()
    print(dataset)

    # keep the 10% of the dataset for validation
    val_dataset = dataset[:int(len(dataset)*0.1)]
    # keep the 90% of the dataset for training and testing
    dataset = dataset[int(len(dataset)*0.1):]
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # get the feature dimension
    feature_dim = dataset.num_features
    print("feature dim: {}".format(feature_dim))

    splits = KFold(n_splits=k, shuffle=True)  # , random_state=42)

    model = MYACRGnnGraph(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=gnn_layers,
        mlp_layers=mlp_layers,
        final_read=final_readout,
        num_classes=2
    )

    model.to(device)

    print("testing acr-gnn with {} layers".format(gnn_layers))

    best_loss = math.inf

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    # for name, param in model.named_parameters():
    #     if "mlp" in name:
    #         param.requires_grad = True
    #         param.lr = lr / 2
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        early_stopper = EarlyStopper(patience=100, min_delta=0.0)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=20, T_mult=2, eta_min=0.00005, last_epoch=-1)

        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

        for epoch in range(max_epochs):
            print("lr:", optimizer.param_groups[0]['lr'])

            train_loss, train_acc = train(
                model, train_loader, loss, optimizer, device)
            test_loss, test_acc = test(model, test_loader, loss, device)
            # scheduler.step()

            if early_stopper.early_stop(test_loss):
                print("Exit from training before for early stopping")
                if test_loss < best_loss:
                    at_fold = fold+1
                    best_loss = test_loss
                    best_epoch = epoch
                    best_params = model.state_dict()
                break

            if test_loss < best_loss:
                at_fold = fold+1
                best_loss = test_loss
                best_epoch = epoch
                best_params = model.state_dict()

            if save_logs:
                writer.add_scalars('runs_split_{}_{}'.format(final_readout, fold), {
                    'Loss/train': train_loss, 'Loss/test': test_loss}, epoch+1)
                writer.add_scalars('runs_split_{}_{}'.format(final_readout, fold), {
                    'Acc/train': train_acc, 'Acc/test': test_acc}, epoch+1)

            print("Epoch: {}".format(epoch+1))
            print("\tTrain : \tTrain loss: {:.4f}\tTrain accuracy: {:.2f}".format(
                train_loss, train_acc))
            print("\tTest: \tTest loss:  {:.4f}\tTest accuracy:  {:.2f}".format(
                test_loss, test_acc))

    print("Best model found at epoch {} with loss {} - fold {}".format(best_epoch+1, best_loss, at_fold))

    print("*"*40)
    model.load_state_dict(best_params)
    model.eval()
    _, train_acc = test(model, train_loader, loss, device)
    _, val_acc = test(model, val_loader, loss, device)
    _, test_acc = test(model, test_loader, loss, device)
    print("final train accuracy:\t{}".format(train_acc))
    print("final test accuracy:\t{}".format(test_acc))
    print("final val accuracy:\t{}".format(val_acc))
    print("*"*40)

    print("*"*40)

    return train_acc, test_acc, val_acc


if __name__ == "__main__":
    seed_everything(1)
    device = torch.device("cpu")

    # ACR-GNN definition
    hidden_dim = [12, 8, 4]
    gnn_layers = len(hidden_dim)
    # feature_dim = 7

    max_epochs = 400
    final_readout = "add"
    lr = 0.001
    min_acc = 120.0

    # Dataset definition
    BATCH_SIZE = 64
    k = 3
    
    save_logs = False
    
    ds_name = "Mutagenicity"
    # ds_name = "MUTAG"

    gnn_layers_string = print_list_with_underscores(hidden_dim)
    experiment_path = f"/Users/raffaelepojer/Dev/RBN-GNN/models/" + ds_name + \
        "_" + gnn_layers_string + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    file_stem = "RBN_acr_graph_" + ds_name

    tensorboard_path = experiment_path + "tensorboard/"
    if save_logs:
        create_folder_if_not_exists(experiment_path)
        create_folder_if_not_exists(tensorboard_path)
        writer = SummaryWriter(log_dir=tensorboard_path)

    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '../', 'datasets')
    dataset = TUDataset(path, name=ds_name).shuffle()
    print(dataset)

    # keep the 10% of the dataset for validation
    val_dataset = dataset[:int(len(dataset)*0.1)]
    # keep the 90% of the dataset for training and testing
    dataset = dataset[int(len(dataset)*0.1):]
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # get the feature dimension
    feature_dim = dataset.num_features
    print("feature dim: {}".format(feature_dim))

    splits = KFold(n_splits=k, shuffle=True)  # , random_state=42)

    model = MYACRGnnGraph(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=gnn_layers,
        mlp_layers=1,
        final_read=final_readout,
        num_classes=2
    )

    model.to(device)

    print("testing acr-gnn with {} layers".format(gnn_layers))
    history = {'train_loss': [], 'test_loss': [],
               'train_acc': [], 'test_acc': []}

    best_loss = math.inf

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    # for name, param in model.named_parameters():
    #     if "mlp" in name:
    #         param.requires_grad = True
    #         param.lr = lr / 2
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        early_stopper = EarlyStopper(patience=100, min_delta=0.0)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=20, T_mult=2, eta_min=0.00005, last_epoch=-1)

        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

        for epoch in range(max_epochs):
            print("lr:", optimizer.param_groups[0]['lr'])

            train_loss, train_acc = train(
                model, train_loader, loss, optimizer, device)
            test_loss, test_acc = test(model, test_loader, loss, device)
            # scheduler.step()

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            if early_stopper.early_stop(test_loss):
                print("Exit from training before for early stopping")
                if test_loss < best_loss:
                    at_fold = fold+1
                    best_loss = test_loss
                    best_epoch = epoch
                    best_params = model.state_dict()
                break

            if test_loss < best_loss:
                at_fold = fold+1
                best_loss = test_loss
                best_epoch = epoch
                best_params = model.state_dict()

            if save_logs:
                writer.add_scalars('runs_split_{}_{}'.format(final_readout, fold), {
                    'Loss/train': train_loss, 'Loss/test': test_loss}, epoch+1)
                writer.add_scalars('runs_split_{}_{}'.format(final_readout, fold), {
                    'Acc/train': train_acc, 'Acc/test': test_acc}, epoch+1)

            print("Epoch: {}".format(epoch+1))
            print("\tTrain : \tTrain loss: {:.4f}\tTrain accuracy: {:.2f}".format(
                train_loss, train_acc))
            print("\tTest: \tTest loss:  {:.4f}\tTest accuracy:  {:.2f}".format(
                test_loss, test_acc))

    print("Best model found at epoch {} with loss {} - fold {}".format(best_epoch+1, best_loss, at_fold))

    print("*"*40)
    model.load_state_dict(best_params)
    model.eval()
    train_loss, train_acc = test(model, train_loader, loss, device)
    val_loss, val_acc = test(model, val_loader, loss, device)
    test_loss, test_acc = test(model, test_loader, loss, device)
    print("final train accuracy:\t{}".format(train_acc))
    print("final test accuracy:\t{}".format(test_acc))
    print("final val accuracy:\t{}".format(val_acc))
    print("*"*40)

    avg_train_loss = np.mean(history['train_loss'])
    avg_test_loss = np.mean(history['test_loss'])
    avg_train_acc = np.mean(history['train_acc'])
    avg_test_acc = np.mean(history['test_acc'])

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}".format(
        avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc))

    print("*"*40)

    # export model
    if test_acc > min_acc:
        # feature_names = [chr(i) for i in range(97, 97+feature_dim)]
        # feature_names = ["Carbon", "Nitrogen", "Oxygen", "Fluorine", "Iodine", "Chlorine", "Bromine"]
        feature_names = ["C", "O", "Cl", "H", "N", "F",
                         "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
        feature_probs = [0.5]*feature_dim

        base_name = f"{file_stem}" + "_" + \
            f"{print_list_with_underscores(hidden_dim)}"
        rbn_name = experiment_path + "/" + base_name + ".rbn"
        gnn_name = experiment_path + "/" + base_name + ".pt"

        write_rbn_ACR_graph(rbn_name, model, ds_name, feature_names, feature_probs,
                            constraints=True, soft_prob=0.99, read_type=final_readout)

        torch.save(model.state_dict(), gnn_name)
        print("Files written to: ", experiment_path)

    if save_logs:
        writer.flush()
        writer.close()
