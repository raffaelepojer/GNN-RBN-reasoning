import random
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, SubsetRandomSampler
from torch_geometric.loader import DataLoader
import math
from gnn import *
from sklearn.model_selection import KFold
from datetime import datetime
from utils.early_stopper import EarlyStopper
from rbn.gnn_to_rbn import *
from utils.utils import *
import utils.read_alpha
from argparse import ArgumentParser 
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
        target = batch.node_labels
        out = torch.squeeze(out)
        l = criterion(out, target.float())

        pred = (torch.sigmoid(out) > 0.5).long()

        tt = len(np.where(target == pred)[0])
        t += tt
        f += len(target)-tt

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
        target = batch.node_labels
        out = torch.squeeze(out)
        l = criterion(out, target.float())

        epoch_loss += l.detach().item()
        pred = (torch.sigmoid(out) > 0.5).long()

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

def train_test(seed,
               hidden_dim,
               max_epochs,
               final_readout,
               lr,
               BATCH_SIZE,
               k,
               save_logs,
               ds_name,
               mlp_layers,
               fwd_dp,
               lin_dp,
               mlp_dp,
               min_acc,
               schedule_lr_fold,
               date):

    seed_everything(seed)
    device = torch.device("cpu")

    gnn_layers = len(hidden_dim)
    # feature_dim = 7

    gnn_layers_string = print_list_with_underscores(hidden_dim)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_base_path = os.path.join(script_dir, "..", "models", ds_name + "_" + gnn_layers_string + "_" + date)
    experiment_path = os.path.join(experiment_base_path, "exp_" + str(seed))
    file_stem = "rbn_acr_graph_" + ds_name

    tensorboard_path = os.path.join(experiment_path, "tensorboard")
    if save_logs:
        create_folder_if_not_exists(experiment_base_path)
        create_folder_if_not_exists(experiment_path)
        create_folder_if_not_exists(tensorboard_path)
        writer = SummaryWriter(log_dir=tensorboard_path)

    alpha = 1
    basedir = os.path.join(script_dir, "..", "datasets", "alpha")
    data_dir = os.path.join(basedir, f"p{alpha}_a_small")

    # data_stem_train = "train-random-erdos-5000-40-50.txt" 
    # data_stem_test_1 = "test-random-erdos-500-40-50.txt"
    # data_stem_test_2 = "test-random-erdos-500-51-60.txt"

    data_stem_train = "train-barabasi-m2_2-5000-5-8.txt" 
    data_stem_test_1 = "test-barabasi-m2_2-500-5-8.txt"
    data_stem_test_2 = "test-barabasi-m2_2-500-10-15.txt"
    
    
    data_train, _ = utils.read_alpha.load_data(
                        dataset=os.path.join(data_dir, data_stem_train),
                        degree_as_node_label=False)
    data_test_1, _ = utils.read_alpha.load_data(
                        dataset=os.path.join(data_dir, data_stem_test_1),
                        degree_as_node_label=False)
    data_test_2, _ = utils.read_alpha.load_data(
                        dataset=os.path.join(data_dir, data_stem_test_2),
                        degree_as_node_label=False)

    dataset = data_train + data_test_1

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_2 = DataLoader(data_test_2, batch_size=BATCH_SIZE, shuffle=False)

    # get the feature dimension
    feature_dim = next(iter(train_loader)).x.shape[1]
    print("feature dim: {}".format(feature_dim))

    splits = KFold(n_splits=k, shuffle=True)  # , random_state=42)

    model = MYACRGnnNode(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=gnn_layers,
        mlp_layers=0, # mlp not currently supported
        final_read=final_readout,
        num_classes=1
    )

    model.to(device)

    print("testing acr-gnn with {} layers".format(gnn_layers))

    best_loss = math.inf

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.0001)

    loss = nn.BCEWithLogitsLoss(reduction='mean')

    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(train_loader)))):
        early_stopper = EarlyStopper(patience=50, min_delta=0.05)

        if schedule_lr_fold > 0 and fold > 0:
            lr *= schedule_lr_fold

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.0001)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=20, T_mult=2, eta_min=0.00005, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True, min_lr=0.00005)

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
            # scheduler.step(test_loss)

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

    text = "Results\n"
    _, train_acc = test(model, train_loader, loss, device)
    text += "final train accuracy:\t{}\n".format(train_acc)
    _, test_acc = test(model, test_loader, loss, device)
    text += "final test accuracy:\t{}\n".format(test_acc)
    _, val_acc = test(model, test_loader_2, loss, device)
    text += "final val accuracy:\t{}\n".format(val_acc)
    print(text)
    print("*"*40)
    write_string_to_file(os.path.join(experiment_path, "results.txt"), text)

    # export model
    if test_acc > min_acc:
        # feature_names = [chr(i) for i in range(97, 97+feature_dim)]
        feature_names = ["blue", "green", "red", "yellow", "purple"]
        feature_probs = [0.5] * 5

        base_name = f"{file_stem}" + "_" + \
            f"{print_list_with_underscores(hidden_dim)}"
        rbn_name = os.path.join(experiment_path, base_name + ".rbn")
        gnn_weights = os.path.join(experiment_path, base_name + ".pt")

        write_rbn_ACR_node(rbn_name, model, ds_name, feature_names, feature_probs,
                            constraints=args.constraints, soft_prob=0.99, read_type=final_readout)

        torch.save(model.state_dict(), gnn_weights)
        print("Files written to: ", experiment_path)

    if save_logs:
        writer.flush()
        writer.close()

    return model, train_acc, test_acc, val_acc, experiment_base_path


def main(args):
    min_acc = 0.0
    lr = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.batch_size
    k=args.k
    final_readout = args.final_readout
    ds_name = args.ds_name
    save_logs = args.save_logs
    hidden_dim = args.hidden_dim
    mlp_layers = 0
    fwd_dp = 0.0
    lin_dp = 0.0
    mlp_dp = 0.0

    schedule_lr_fold = 0
    num_of_experiments = args.num_exp
    seeds = random_list(num_of_experiments, 1, 100)
    train_accs = []
    test_accs = []
    val_accs = []
    exp_path = ""
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_acc = 0.0
    best_seed = -1
    for seed in seeds:
        model, train_acc, test_acc, val_acc, path = train_test(seed=seed,
                                                               hidden_dim=hidden_dim,
                                                               max_epochs=max_epochs,
                                                               final_readout=final_readout,
                                                               lr=lr,
                                                               BATCH_SIZE=BATCH_SIZE,
                                                               k=k,
                                                               save_logs=save_logs,
                                                               ds_name=ds_name,
                                                               mlp_layers=mlp_layers,
                                                               fwd_dp=fwd_dp,
                                                               lin_dp=lin_dp,
                                                               mlp_dp=mlp_dp,
                                                               min_acc=min_acc,
                                                               schedule_lr_fold=schedule_lr_fold,
                                                               date=date)
        if val_acc > best_acc:
            best_seed = seed
            best_acc = val_acc

        exp_path = path
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print()
    text = "Results for {} experiments\n".format(num_of_experiments)
    avg, std = average_and_standard_deviation(train_accs)
    text = "Average train accuracy:\t{:.4f}\tSTD:\t{:.4f}".format(avg, std)
    text += "\n"
    avg, std = average_and_standard_deviation(test_accs)
    text += "Average test accuracy:\t{:.4f}\tSTD:\t{:.4f}".format(avg, std)
    text += "\n"
    avg, std = average_and_standard_deviation(val_accs)
    text += "Average valid accuracy:\t{:.4f}\tSTD:\t{:.4f}".format(avg, std)
    text += "\n\n"
    text += "Best model found at seed {} with acc {}".format(
        best_seed, best_acc)
    text += "\n"
    text += "Parameters used:\n"
    text += "ds_name: {}\n".format(ds_name)
    text += "hidden_dim: {}\n".format(hidden_dim)
    text += "mlp_layers: {}\n".format(mlp_layers)
    text += "fwd_dp: {}\n".format(fwd_dp)
    text += "lin_dp: {}\n".format(lin_dp)
    text += "mlp_dp: {}\n".format(mlp_dp)
    text += "final_readout: {}\n".format(final_readout)
    text += "lr: {}\n".format(lr)
    text += "max_epochs: {}\n".format(max_epochs)
    text += "BATCH_SIZE: {}\n".format(BATCH_SIZE)
    text += "k: {}\n".format(k)
    text += "schedule_lr_fold: {}\n".format(schedule_lr_fold)
    print(text)
    write_string_to_file(exp_path + "results.txt", text)

if __name__ == "__main__":
    parser = ArgumentParser(description="ACR-GNN for blue experiment with k-fold cross validation on multiple seeds")
    parser.add_argument("--num_exp", type=int, default=3, help="Number of experiments")
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[20], help="Hidden layer dimensions")
    parser.add_argument("--k", type=int, default=3, help="Number of folds")
    parser.add_argument("--final_readout", type=str, default="add", help="Final readout function")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Max number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--constraints", type=bool, default=True, help="Constraints for RBN")
    parser.add_argument("--ds_name", type=str, default="alpha1", help="Dataset name")
    parser.add_argument("--save_logs", type=bool, default=True, help="Save logs")
    parser.add_argument("--save_rbn", type=bool, default=True, help="Save RBN")
    args = parser.parse_args()
    main(args)