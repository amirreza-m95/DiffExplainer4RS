import pandas as pd
import numpy as np
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

from pathlib import Path
import pickle
import torch
from torch import nn, optim, Tensor
import torch.nn as nn
import torch.nn.functional as F
import optuna
import logging
import matplotlib.pyplot as plt

from recommenders_architecture import *
from help_functions import *
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from scipy import sparse

import os
import argparse

data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "VAE" ## Can be MLP, VAE, MLP_model, GMF_model, NCF, LightGCN

DP_DIR = Path("datasets", "lxr-CE", data_name)
export_dir = Path(os.getcwd())
files_path = Path(export_dir, DP_DIR)
checkpoints_path = Path(export_dir, "checkpoints", "recommenders")
Neucheckpoints_path = Path(export_dir, "checkpoints", "recommenders")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# print(f'device is set to {device}')

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single" #changed
}

num_users_dict = {
    "ML1M":6037,
    "Yahoo":13797, 
    "Pinterest":19155
}

num_items_dict = {
    "ML1M":3381,
    "Yahoo":4604, 
    "Pinterest":9362
}

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

ITERATIONS = 5000
EPOCHS = 20

BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 200
ITERS_PER_LR_DECAY = 200
LAMBDA = 1e-6



output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)


for row in range(static_test_data.shape[0]):
    static_test_data.iloc[row, static_test_data.iloc[row,-2]]=0
test_array = static_test_data.iloc[:,:-2].to_numpy()

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

from recommenders_architecture import *

kw_dict = {'device':device,
          'num_items': num_items,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name}


train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

def MLP_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0, 4) # hyperparameter that weights the different loss terms
    epochs = 10
    model = MLP(hidden_dim, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    
    print(f'======================== new run - {recommender_name} ========================')
    logger.info(f'======================== new run - {recommender_name} ========================')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss=[]
        train_neg_loss=[]
        if epoch!=0 and epoch%10 == 0: # decrease the learning rate every 10 epochs
            lr = 0.1*lr
            optimizer.lr = lr
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device)

            batch_pos_idx = train_matrix[batch_idx,-2]
            batch_neg_idx = train_matrix[batch_idx,-1]
            
            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)
            
            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
            
            # MSE loss
            pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
            neg_loss = torch.mean((neg_output)**2)
            
            batch_loss = pos_loss + beta*neg_loss
            batch_loss.backward()
            optimizer.step()
            
            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())
            
        print(f'train pos_loss = {np.mean(train_pos_loss)}, neg_loss = {np.mean(train_neg_loss)}')    
        train_losses.append(np.mean(loss))
        if epoch % ((epochs*5)/100) == 0:
            print(f'this model is saved as MLP_{data_name}_{epoch}_{hidden_dim}.pt')
            torch.save(model.state_dict(), Path(Neucheckpoints_path, f'MLP_{data_name}_{epoch}_{hidden_dim}.pt'))
        # torch.save(model.state_dict(), Path(checkpoints_path, f'MLP_{data_name}_{round(lr,4)}_{batch_size}_{trial.number}_{epoch}.pt'))


        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        
        test_pos = test_matrix[:,-2]
        test_neg = test_matrix[:,-1]
        
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        
        pos_items = torch.Tensor(items_array[test_pos]).to(device)
        neg_items = torch.Tensor(items_array[test_neg]).to(device)
        
        pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
        neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
        
        pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
        neg_loss = torch.mean((neg_output)**2)
        print(f'test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10) # metric for monitoring
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        test_losses.append(-hit_rate_at_10)
        if epoch>5: # early stop if the HR@10 decreases for 4 epochs in a row
            if test_losses[-2]<=test_losses[-1] and test_losses[-3]<=test_losses[-2] and test_losses[-4]<=test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
            
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

VAE_config= {
"enc_dims": [256,64],
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

def VAE_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [128,256])
    epochs = 20
    model = VAE(VAE_config ,**kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    print('======================== new run ========================')
    logger.info('======================== new run ========================')
    
    for epoch in range(epochs):
        if epoch!=0 and epoch%10 == 0:
            lr = 0.1*lr
            optimizer.lr = lr
        loss = model.train_one_epoch(train_array, optimizer, batch_size)
        train_losses.append(loss)
        if epoch % ((epochs*5)/100) == 0:
            print(f'this model is saved as VAE_{data_name}_{trial.number}_{epoch}_{batch_size}.pt')
            torch.save(model.state_dict(), Path(Neucheckpoints_path, f'VAE_{data_name}_{trial.number}_{epoch}_{batch_size}.pt'))
        # torch.save(model.state_dict(), Path(checkpoints_path, f'VAE_{data_name}_{trial.number}_{epoch}_{round(lr,4)}_{batch_size}.pt'))


        model.eval()
        test_matrix = static_test_data.to_numpy()
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        test_pos = test_array[:,-2]
        test_neg = test_array[:,-1]
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        output = model(test_tensor).to(device)
        pos_loss = -output[row_indices,test_pos].mean()
        neg_loss = output[row_indices,test_neg].mean()
        print(f'pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        print('Hit 10,  Hit 50, Hit 100, MPR, MPR')
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        test_losses.append(pos_loss.item())
        if epoch>5:
            if test_losses[-2]<test_losses[-1] and test_losses[-3]<test_losses[-2] and test_losses[-4]<test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
    
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(f"{recommender_name}_{data_name}_Optuna.log", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

study = optuna.create_study(direction='maximize')

logger.info("Start optimization.")

if recommender_name == 'MLP':
    study.optimize(MLP_objective, n_trials=10) 
elif recommender_name == 'VAE':
    study.optimize(VAE_objective, n_trials=10) 

with open(f"{recommender_name}_{data_name}_Optuna.log") as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"
    
    
# Print best hyperparameters and corresponding metric value
print("Best hyperparameters: {}".format(study.best_params))
print("Best metric value: {}".format(study.best_value))


from help_functions import *

recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),

    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
}

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]


def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)
    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path))
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    return recommender
    
model = load_recommender()



topk_test = {}
for i in range(len(test_array)):
    vec = test_array[i]
    tens = torch.Tensor(vec).to(device)
    topk_test[i] = int(get_user_recommended_item(tens, model, **kw_dict).cpu().detach().numpy())


hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)

print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)