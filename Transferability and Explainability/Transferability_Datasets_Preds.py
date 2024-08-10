# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:18:27 2024

@author: magha
"""


import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import json
import ast
from sklearn.model_selection import train_test_split
import joblib
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from mordred import Calculator, descriptors
import deepchem as dc
import time

from Graph_Featurizer import MoleculeDataset
from sklearn.model_selection import train_test_split

save_directory = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Manuscript_codes\Saved_Models"


""" Tabular Feat"""
path_save_data = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Revised_Processed_Saved_Data_solvent'
path_saved_model = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\ML based on Tabular Features'
path_Numpy = path_save_data


#Loading alpha beta Volume, Area, Sphericity features extracted from ESP maps
AB_Features_ESOL_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_ESOL_NoI.npy")
Labels_ESOL_NoI= pd.read_csv(f"{path_Numpy}/Labels_ESOL_NoI.csv")

AB_Features_PHYS_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_PHYS_NoI.npy")
Labels_PHYS_NoI= pd.read_csv(f"{path_Numpy}/Labels_PHYS_NoI.csv")

AB_Features_AQUA_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_AQUA_NoI.npy")
Labels_AQUA_NoI= pd.read_csv(f"{path_Numpy}/Labels_AQUA_NoI.csv")

AB_Features_OCHEM_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_OCHEM_NoI.npy")
Labels_OCHEM_NoI= pd.read_csv(f"{path_Numpy}/Labels_OCHEM_NoI.csv")

AB_Features_EOAP_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_E_A_P_O_NoI.npy")
Labels_EOAP_NoI= pd.read_csv(f"{path_Numpy}/Labels_E_A_P_O_NoI.csv")




# Mordred

featurizer = dc.feat.MordredDescriptors(ignore_3D=False)
smiles_EAPO = Labels_EOAP_NoI['smiles'].tolist()
Mordred_EOAP_NoI  = featurizer.featurize(smiles_EAPO)

smiles_ESOL = Labels_ESOL_NoI['smiles'].tolist()
Mordred_ESOL_NoI = featurizer.featurize(smiles_ESOL)

smiles_PHYS = Labels_PHYS_NoI['smiles'].tolist()
Mordred_PHYS_NoI = featurizer.featurize(smiles_PHYS)

smiles_AQUA = Labels_AQUA_NoI['smiles'].tolist()
Mordred_AQUA_NoI = featurizer.featurize(smiles_AQUA)

smiles_OCHEM = Labels_OCHEM_NoI['smiles'].tolist()
Mordred_OCHEM_NoI = featurizer.featurize(smiles_OCHEM)


def AB_Mord_Transfer_Eval (Main_Mordred_Feat, Main_AB_Feat, Main_Labels_data_csv, Test_Mordred_Feat, Test_AB_Feat, Test_Labels_data_csv, model_file_name_csv):
    
    calc = Calculator(descriptors, ignore_3D=False)
    Mordred_feature_names = [str(descriptor) for descriptor in calc.descriptors]
    AB_feature_names = [f'Alpha_{i}' for i in range(1, 55)] + [f'Beta_{i}' for i in range(1, 23)] + ['V', 'Area', 'sph']
    
    AB_top_n_feature_indices = np.load(os.path.join(path_saved_model, 'AB_top_n_feature_indices.npy'))
    M_top_n_feature_indices = np.load(os.path.join(path_saved_model, 'M_top_n_feature_indices.npy'))
    
    
    "Train Data"
    X_M = Main_Mordred_Feat
    X_M_Test = Test_Mordred_Feat
    X_AB = Main_AB_Feat
    X_AB_Test = Test_AB_Feat
    y_real_list = Main_Labels_data_csv['logS'].tolist()
    y_real_list_Test =  Test_Labels_data_csv['logS'].tolist()
    smiles_test = Test_Labels_data_csv['smiles'].tolist()
    y = np.array(y_real_list)
    y_Test = np.array(y_real_list_Test)
    
    X_df_AB = pd.DataFrame(X_AB, columns=AB_feature_names)
    X_df_AB_Test = pd.DataFrame(X_AB_Test, columns=AB_feature_names)

    X_df_M = pd.DataFrame(X_M, columns=Mordred_feature_names)
    X_df_M_Test = pd.DataFrame(X_M_Test, columns=Mordred_feature_names)

    y_df = pd.DataFrame(y, columns=['target'])
    y_df_Test = pd.DataFrame(y_Test, columns=['target'])

    X_train_AB, _, y_train_AB, _ = train_test_split(
        X_df_AB,
        y_df,
        test_size=0.2,
        random_state=104
    )
    
    
    X_train_M, _, y_train_M, _ = train_test_split(
        X_df_M,
        y_df,
        test_size=0.2,
        random_state=104
    )    
    
    
    X_train_selected_AB = X_train_AB.iloc[:, AB_top_n_feature_indices]
    X_test_selected_AB = X_df_AB_Test.iloc[:, AB_top_n_feature_indices]
    
    X_train_selected_M = X_train_M.iloc[:, M_top_n_feature_indices]
    X_test_selected_M = X_df_M_Test.iloc[:, M_top_n_feature_indices]
    
    X_train_selected_nonscaled = np.concatenate((X_train_selected_M, X_train_selected_AB), axis =1)
    X_test_selected_nonscaled = np.concatenate((X_test_selected_M, X_test_selected_AB), axis = 1)
    
    scaler = MinMaxScaler()
    X_train_selected = scaler.fit_transform(X_train_selected_nonscaled)
    X_test_selected = scaler.transform(X_test_selected_nonscaled)
    
    
    y_train = y_train_M
    y_test = y_Test
    
    Data_Name = model_file_name_csv.split('_')[1]
    model_path_save = f"{save_directory}/XGB_best_model_{Data_Name}.joblib"
    loaded_model = joblib.load(model_path_save)
    model = loaded_model
    
    start_time = time.time()  # Record start time before training loop

    test_predictions_selected = model.predict(X_test_selected)
    
    total_Preds_time = time.time() - start_time

    # Evaluate the model
    mse = mean_squared_error(y_test, test_predictions_selected)
    mae = mean_absolute_error(y_test, test_predictions_selected)
    r2 = r2_score(y_test, test_predictions_selected)
    rmse_knn = np.sqrt(mean_squared_error(y_test, test_predictions_selected))
    
    output_range_test = y_test.max() - y_test.min()
    
    rRMSE = np.sqrt(mse) / output_range_test
    
    print(f'Average prediction time: {total_Preds_time/len(y_test)} seconds')
    print("Relative RMSE (rRMSE):", rRMSE)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)
    print("RMSE:", rmse_knn)
    y_test = np.array(y_test).ravel() 
    return smiles_test, y_test, test_predictions_selected



"""Prediction Results Tabular"""
E_smiles, E_Labels_Tabular, E_Preds_Tabular = AB_Mord_Transfer_Eval (Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI, Mordred_ESOL_NoI, AB_Features_ESOL_NoI, Labels_ESOL_NoI, 'Labels_PHYS_NoI')
A_smiles, A_Labels_Tabular, A_Preds_Tabular = AB_Mord_Transfer_Eval (Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI, Mordred_AQUA_NoI, AB_Features_AQUA_NoI, Labels_AQUA_NoI, 'Labels_PHYS_NoI')
P_smiles, P_Labels_Tabular, P_Preds_Tabular = AB_Mord_Transfer_Eval (Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI, Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI, 'Labels_PHYS_NoI')
O_smiles, O_Labels_Tabular, O_Preds_Tabular = AB_Mord_Transfer_Eval (Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI, Mordred_OCHEM_NoI, AB_Features_OCHEM_NoI, Labels_OCHEM_NoI, 'Labels_PHYS_NoI')
EOAP_smiles, EOAP_Labels, EOAP_Preds_Tabular = AB_Mord_Transfer_Eval (Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI, Mordred_EOAP_NoI, AB_Features_EOAP_NoI, Labels_EOAP_NoI, 'Labels_PHYS_NoI')


""" Graph """
class MPNNModel(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNNModel, self).__init__(aggr='mean')  # 'mean' aggregation for global pooling
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, edge_index, batch):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)

        # Global Pooling (stack different aggregations)
        x = torch.cat([global_mean_pool(x, batch), 
                        global_max_pool(x, batch)], dim=1)

        x = self.fc(x)
        return x


def GNN_Fun_Transfer_Eval(Test_Data_root_path, Test_file_name_csv, model_file_name_csv):


    root_path = Test_Data_root_path
    filename_csv = Test_file_name_csv
    dataset_molnet = MoleculeDataset(root=root_path, filename=filename_csv)
    dataset_graphs = dataset_molnet
    Data_Name = model_file_name_csv.split('_')[1]
    
    model_path = f"{save_directory}/GNN_best_model_{Data_Name}.pth"  # Specify your desired path

    # Prepare the data
    data_size = len(dataset_graphs)
    NUM_GRAPHS_PER_BATCH = 64  # batch size
    data_loader = DataLoader(dataset_graphs)
    
    features_test = [graph.x for graph in data_loader.dataset]
    edge_indices_test = [graph.edge_index for graph in data_loader.dataset]
    edge_attrs_test = [graph.edge_attr for graph in data_loader.dataset]
    labels_test = [graph.y for graph in data_loader.dataset]
    smiles_test = [graph.smiles for graph in data_loader.dataset]
    
    # features_train, features_test, edge_indices_train, edge_indices_test, \
    # edge_attrs_train, edge_attrs_test, labels_train, labels_test, \
    # smiles_train, smiles_test = train_test_split(
    #     features, edge_indices, edge_attrs, labels, smiles_info,
    #     random_state=104, test_size=0.2, shuffle=True)
    
    data_test = [Data(x=features_test[i], edge_index=edge_indices_test[i], edge_attr=edge_attrs_test[i], y=labels_test[i], smiles=smiles_test[i]) for i in range(len(features_test))]
    test_loader = DataLoader(data_test, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
    
    # Initialize the GNN model
    input_dim = dataset_graphs.num_features
    hidden_dim = 128
    output_dim = 1  # Single scalar output for solubility prediction
    model = MPNNModel(input_dim, hidden_dim, output_dim)
    
    # Load the pre-trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression task

    start_time = time.time()  # Record start time before training loop

    
    total_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y.view(-1, 1).float())
            total_loss += loss.item() * batch.num_graphs
    
            # Convert output and true labels to numpy arrays
            predictions.extend(output.cpu().numpy())
            true_labels.extend(batch.y.view(-1).cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}")
    total_Preds_time = time.time() - start_time

    # Compute evaluation metrics
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print(f'Average prediction time: {total_Preds_time/len(true_labels):.4f} seconds')

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    r_squared = r2_score(true_labels, predictions)
    print(f"R-squared: {r_squared:.4f}")
    
    predictions_flat = np.array(predictions).flatten()
    return true_labels, predictions_flat






"""Prediction Results Graph"""

root_path_P = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_PHYS"
file_name_csv_P = "Labels_PHYS_NoI.csv"

root_path_E =  r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_ESOL"
file_name_csv_E = "Labels_ESOL_NoI.csv"

root_path_A = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_AQUA"
file_name_csv_A = "Labels_AQUA_NoI.csv"

root_path_O = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_OCHEM"
file_name_csv_O = "Labels_OCHEM_NoI.csv"

root_path_EOAP = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_EAOP"
file_name_csv_EOAP = "Labels_EOAP_NoI.csv"


E_Labels_Graph, E_Preds_Graph  = GNN_Fun_Transfer_Eval(root_path_E,file_name_csv_E , file_name_csv_P)
A_Labels_Graph, A_Preds_Graph  = GNN_Fun_Transfer_Eval(root_path_A, file_name_csv_A, file_name_csv_P)
P_Labels_Graph, P_Preds_Graph  = GNN_Fun_Transfer_Eval(root_path_P, file_name_csv_P, file_name_csv_P)
O_Labels_Graph, O_Preds_Graph  = GNN_Fun_Transfer_Eval(root_path_O, file_name_csv_O, file_name_csv_P)
EOAP_Labels_Graph, EOAP_Preds_Graph  = GNN_Fun_Transfer_Eval(root_path_EOAP, file_name_csv_EOAP, file_name_csv_P)







""" ====== ESP ====== """

import torch
import torch.nn as nn
import os
import torch.optim as optim
import pandas as pd
import datetime
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from random import sample, randint
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

from Utils_PC_Processing import apply_random_rotation, augment_rot
from Utils_PC_Processing import normalize_train_test_xyz_esp
from Utils_PC_Processing import plot_comparison




def Fix_size(data_PointCloud, Min_points_No):
    
     PointCloud_fixedSize = []
     for count , my_data in enumerate(data_PointCloud):
        num_points = np.shape(my_data)[0]
        rand_idx = sample(range(0, num_points), Min_points_No)
        my_data = my_data[rand_idx,:]
        PointCloud_fixedSize.append(my_data)
    
     return PointCloud_fixedSize


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x) #(batch_size, Num_points, Num_points)
    xx = torch.sum(x**2, dim=1, keepdim=True) #(batch_size, 1, Num_points)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #(batch_size, Num_points, Num_points)
    
    #print(f"Input tensor shape: {x.shape}")
    #print(f"Pairwise distance shape: {pairwise_distance.shape}")
    #numpy_pair_dist = pairwise_distance.numpy()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k) where k is the indices of the k nearest neighbors for each point in the point cloud
    return idx



def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
            
    device = torch.device('cpu')
    #device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx_base = idx_base.expand(-1, num_points, k)  # Expand the dimensions to match the shape of idx

    idx = idx + idx_base     # Replace view with reshape # Convert idx_base to a 1D tensor and then add to idx  

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)

    feature = x.view(batch_size*num_points, -1)[idx, :]

    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


    return feature  # (batch_size, 2*num_dims, num_points, k)



class FeatureExtraction(nn.Module):
    def __init__(self, k=20):
        super(FeatureExtraction, self).__init__()
        #self.args = args
        self.k = k
        dropout = 0.4
        emb_dims = 512
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(2*4, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512*2, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
   
        batch_size = x.size(0)
        x=x.permute(0,2,1)  # Permute to (batch_size, channels, sequence_length)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        return x


class EdgeConv(nn.Module):
    def __init__(self, k=20, output_channels=1):
        super(EdgeConv, self).__init__()
        
        self.k = k
        dropout = 0.4
        emb_dims = 512
        
        self.feature_extraction = FeatureExtraction(k=k)
        
        self.linear1 = nn.Linear(emb_dims*1*2, 1*256, bias=False) # Please note that concatenation doubles the required neurons
        self.bn6 = nn.BatchNorm1d(1*256)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(1*256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Sequential(nn.Linear(128, output_channels))  #

    def forward(self, x):
        x = self.feature_extraction(x)

        x = x.view(x.size(0), -1)

        #print(f"Input tensor shape: {x.shape}")
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 256, num_points, k)
        return x



def EdgeConv_Fun_Transfer_Eval(Test_file_name_csv, Main_file_name_csv):
    
    Model_Data_Name = Main_file_name_csv.split('_')[1]
    Test_Data_Name = Test_file_name_csv.split('_')[1]

    Main_PC_path = fr"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Numpy_Arrays_Datasets_solvent\PC_{Model_Data_Name}_NoI.npy"
    Main_labels_path = fr"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Numpy_Arrays_Datasets_solvent\Labels_{Model_Data_Name}_NoI.csv"
    Main_model_path = f"{save_directory}/EdgeConv_best_model_{Model_Data_Name}_cpu.pth"  # Specify your desired path
    Main_PC = np.load(Main_PC_path)
    Main_Labels_stand = pd.read_csv(Main_labels_path)
    Main_Labels = Main_Labels_stand['logS'].to_numpy()
    
    
    Test_PC_path = fr"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Numpy_Arrays_Datasets_solvent\PC_{Test_Data_Name}_NoI.npy"
    Test_labels_path = fr"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Numpy_Arrays_Datasets_solvent\Labels_{Test_Data_Name}_NoI.csv"
    Test_PC = np.load(Test_PC_path)
    Test_Labels_stand = pd.read_csv(Test_labels_path)
    Test_Labels = Test_Labels_stand['logS'].to_numpy()
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixing the size of pointcloud data to minimum number of Points amoung all point clouds
    Min_points_No = 3000
    Main_PC = Fix_size(Main_PC, Min_points_No)
    Main_PC = np.array(Main_PC)
    
    Test_PC = Fix_size(Test_PC, Min_points_No)
    Test_PC = np.array(Test_PC)
    
    # Train-test split
    PC_train, _, Labels_train, _ = train_test_split(
        Main_PC, Main_Labels, random_state=104, test_size=0.2, shuffle=True)
    
    Labels_test = Test_Labels
    PC_train_norm, PC_test_norm = normalize_train_test_xyz_esp(PC_train, Test_PC)
    
    batch_size = 4 
    num_full_batches = len(PC_train_norm) // batch_size
    subselected_size = num_full_batches * batch_size
    
    PC_train_norm_sampled = PC_train_norm[:]
    Labels_train_sampled = Labels_train[:]
    
    PC_test_norm_sampled = PC_test_norm[:]
    Labels_test_sampled = Labels_test[:]
    
    print("Sizes of Sampled Training Data:")
    print("PC_train_norm_sampled:", np.shape(PC_train_norm))
    
    print("Sizes of Full Test Data:")
    print("PC_test_norm:", np.shape(PC_test_norm))
    
    PC_train_norm = PC_train_norm_sampled
    Labels_train = Labels_train_sampled
    
    PC_test_norm = PC_test_norm_sampled
    Labels_test = Labels_test_sampled
    
    Num_classes = 1
    Num_Points = np.shape(Main_PC)[1]
    Batch_Size = 4
    num_features = 4
    num_points = np.shape(Main_PC)[1]
    print(f'batch Size is: {Batch_Size}')
    
    # Convert data to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert data to PyTorch tensors and send them to the specified device
    PC_train_norm_tensor = torch.FloatTensor(PC_train_norm).to(device)
    Labels_train_tensor = torch.FloatTensor(Labels_train).to(device)
    
    PC_test_norm_tensor = torch.FloatTensor(PC_test_norm).to(device)
    Labels_test_tensor = torch.FloatTensor(Labels_test).to(device)
    
    # Create DataLoader for training and testing
    train_dataset = TensorDataset(PC_train_norm_tensor, Labels_train_tensor)
    test_dataset = TensorDataset(PC_test_norm_tensor, Labels_test_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)
    
    data_iterator = iter(train_dataloader)  # Assuming train_dataloader is your DataLoader
    batch = next(data_iterator)  # Get the first batch
    x_data_1, labels_1 = batch  # Extract features and labels from the batch
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EdgeConv().to(device)
    #print(model)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    best_model = EdgeConv().to(device)
    best_model.load_state_dict(torch.load(Main_model_path))
    
    model = best_model
    model.eval()
    predictions = []
    true_labels = []
    start_time = time.time()  # Record start time before training loop

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_1, labels = batch
            inputs_1, labels = inputs_1.to(device), labels.to(device)
            pred = model(inputs_1)
            
            # Append predictions and true labels
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    total_Preds_time = time.time() - start_time

    mse = mean_squared_error(true_labels, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_labels, predictions)
    r_squared = r2_score(true_labels, predictions)
    
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print(f'Average prediction time: {total_Preds_time/len(true_labels):.4f} seconds')
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    predictions_flat = np.array(predictions).flatten()

    return true_labels, predictions_flat
    





""" Prediction Results ESP"""

E_Labels_EdgeConv, E_Preds_EdgeConv = EdgeConv_Fun_Transfer_Eval(file_name_csv_E, file_name_csv_P)
A_Labels_EdgeConv, A_Preds_EdgeConv = EdgeConv_Fun_Transfer_Eval(file_name_csv_A, file_name_csv_P)
P_Labels_EdgeConv, P_Preds_EdgeConv = EdgeConv_Fun_Transfer_Eval(file_name_csv_P, file_name_csv_P)
O_Labels_EdgeConv, O_Preds_EdgeConv =EdgeConv_Fun_Transfer_Eval(file_name_csv_O, file_name_csv_P)
EOAP_Labels_EdgeConv, EOAP_Preds_EdgeConv = EdgeConv_Fun_Transfer_Eval(file_name_csv_EOAP, file_name_csv_P)



def extract_preds(dataframe, true_labels_name, preds_name):
    true_labels = dataframe[f'{true_labels_name}'].to_numpy()
    predictions = dataframe[f'{preds_name}'].to_numpy()
    valid_indices = ~np.isnan(true_labels) & ~np.isnan(predictions)
    true_labels = true_labels[valid_indices]
    predictions = predictions[valid_indices]
    errors = predictions - true_labels
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    r_squared = r2_score(true_labels, predictions)
    return true_labels, predictions, errors, mae, rmse, r_squared


# # Combine all predictions into a DataFrame
# df = pd.DataFrame({
#     'E_True_Labels': E_Labels_Tabular, 'E_Tabular_Predictions': E_Preds_Tabular, 'E_GNN_Predictions': E_Preds_Graph,'E_EdgeConv_Predictions': E_Preds_EdgeConv,
#     'A_True_Labels': A_Labels_Tabular, 'A_Tabular_Predictions': A_Preds_Tabular,'A_GNN_Predictions': A_Preds_Graph,'A_EdgeConv_Predictions': A_Preds_EdgeConv,
#     'P_True_Labels': P_Labels_Tabular, 'P_Tabular_Predictions': P_Preds_Tabular,'P_GNN_Predictions': P_Preds_Graph,'P_EdgeConv_Predictions': P_Preds_EdgeConv,
#     'O_True_Labels': O_Labels_Tabular, 'O_Tabular_Predictions': O_Preds_Tabular,'O_GNN_Predictions': O_Preds_Graph,'O_EdgeConv_Predictions': O_Preds_EdgeConv,
#     'EOAP_True_Labels': EOAP_Labels, 'EOAP_Tabular_Predictions': EOAP_Preds_Tabular,'EOAP_GNN_Predictions': EOAP_Preds_Graph,'EOAP_EdgeConv_Predictions': EOAP_Preds_EdgeConv,
# })



data_dict_transfer_noENS = {
    'E_smiles': E_smiles, 'E_True_Labels': E_Labels_Tabular, 'E_Tabular_Predictions': E_Preds_Tabular, 'E_GNN_Predictions': E_Preds_Graph,'E_EdgeConv_Predictions': E_Preds_EdgeConv,
    'A_smiles': A_smiles, 'A_True_Labels': A_Labels_Tabular, 'A_Tabular_Predictions': A_Preds_Tabular,'A_GNN_Predictions': A_Preds_Graph,'A_EdgeConv_Predictions': A_Preds_EdgeConv,
    'P_smiles': P_smiles, 'P_True_Labels': P_Labels_Tabular, 'P_Tabular_Predictions': P_Preds_Tabular,'P_GNN_Predictions': P_Preds_Graph,'P_EdgeConv_Predictions': P_Preds_EdgeConv,
    'O_smiles': O_smiles, 'O_True_Labels': O_Labels_Tabular, 'O_Tabular_Predictions': O_Preds_Tabular,'O_GNN_Predictions': O_Preds_Graph,'O_EdgeConv_Predictions': O_Preds_EdgeConv,
    'EOAP_smiles': EOAP_smiles, 'EOAP_True_Labels': EOAP_Labels, 'EOAP_Tabular_Predictions': EOAP_Preds_Tabular,'EOAP_GNN_Predictions': EOAP_Preds_Graph,'EOAP_EdgeConv_Predictions': EOAP_Preds_EdgeConv,
}

Saved_Results_dir = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Manuscript_codes\Saved_Results'


with open(f'{Saved_Results_dir}/all_transferability_predictions.pkl', 'wb') as f:
    pickle.dump(data_dict_transfer_noENS, f)
print(f"All predictions saved to {Saved_Results_dir}/all_transferability_predictions.pkl")

# Load the dictionary from the file
with open(f'{Saved_Results_dir}/all_transferability_predictions.pkl', 'rb') as f:
    data_dict_transfer_noENS = pickle.load(f)

# Extract predictions and calculate metrics
def extract_preds(smiles, true_labels, predictions):
    valid_indices = ~np.isnan(true_labels) & ~np.isnan(predictions)
    true_labels = true_labels[valid_indices]
    predictions = predictions[valid_indices]
    errors = predictions - true_labels
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    r_squared = r2_score(true_labels, predictions)
    return true_labels, predictions, errors, mae, rmse, r_squared, smiles

def extract_all_metrics(data_dict):
    results = {}
    columns = [
        ('E_smiles', 'E_True_Labels', 'E_Tabular_Predictions'), ('E_smiles','E_True_Labels', 'E_GNN_Predictions'), ('E_smiles','E_True_Labels', 'E_EdgeConv_Predictions'),
        ('A_smiles', 'A_True_Labels', 'A_Tabular_Predictions'), ('A_smiles','A_True_Labels', 'A_GNN_Predictions'), ('A_smiles','A_True_Labels', 'A_EdgeConv_Predictions'),
        ('P_smiles', 'P_True_Labels', 'P_Tabular_Predictions'), ('P_smiles', 'P_True_Labels', 'P_GNN_Predictions'), ('P_smiles', 'P_True_Labels', 'P_EdgeConv_Predictions'),
        ('O_smiles', 'O_True_Labels', 'O_Tabular_Predictions'), ('O_smiles', 'O_True_Labels', 'O_GNN_Predictions'), ('O_smiles','O_True_Labels', 'O_EdgeConv_Predictions'),
        ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_Tabular_Predictions'), ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_GNN_Predictions'), ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_EdgeConv_Predictions')
    ]
    
    for smiles, true_label, pred in columns:
        true_labels, predictions, errors, mae, rmse, r_squared, smiles = extract_preds(data_dict[smiles] , data_dict[true_label], data_dict[pred])
        results[pred] = {
            'true_labels': true_labels, 'predictions': predictions, 'errors': errors, 'mae': mae, 'rmse': rmse, 'r_squared': r_squared, 'smiles':smiles
        }
    return results


metrics_transfer = extract_all_metrics(data_dict_transfer_noENS)


def calculate_ensemble(predictions, rmses):
    weights = 1 / rmses
    normalized_weights = weights / weights.sum()
    ensemble_preds = np.sum(predictions * normalized_weights[:, None], axis=0)
    return ensemble_preds

"""New """
def calculate_ensemble(predictions, rmses, tabular_weight=2):
    # Increase the weight for Tabular predictions
    weights = np.ones(len(rmses))
    weights[0] = tabular_weight
    rmses = rmses / rmses[0]  # Normalize rmses by Tabular rmse to balance
    weights /= rmses  # Inverse rmses for weights
    normalized_weights = weights / weights.sum()
    ensemble_preds = np.sum(predictions * normalized_weights[:, None], axis=0)
    return ensemble_preds



def add_ensemble_predictions(data_dict, metrics):
    for prefix in ['E', 'A', 'P', 'O', 'EOAP']:
        preds = np.array([
            metrics[f'{prefix}_Tabular_Predictions']['predictions'],
            metrics[f'{prefix}_GNN_Predictions']['predictions'],
            metrics[f'{prefix}_EdgeConv_Predictions']['predictions']
        ])
        rmses = np.array([
            metrics[f'{prefix}_Tabular_Predictions']['rmse'],
            metrics[f'{prefix}_GNN_Predictions']['rmse'],
            metrics[f'{prefix}_EdgeConv_Predictions']['rmse']
        ])
        ensemble_preds = calculate_ensemble(preds, rmses)
        data_dict[f'{prefix}_Ensemble_Predictions'] = ensemble_preds
    return data_dict



data_dict_transfer = add_ensemble_predictions(data_dict_transfer_noENS, metrics_transfer)



with open(f'{Saved_Results_dir}/all_transfer_predictions_with_ensemble.pkl', 'wb') as f:
    pickle.dump(data_dict_transfer, f)
print(f"All predictions with ensemble saved to {Saved_Results_dir}/all_transfer_predictions_with_ensemble.pkl")

with open(f'{Saved_Results_dir}/all_transfer_predictions_with_ensemble.pkl', 'rb') as f:
    data_dict_transfer_loaded = pickle.load(f)



def extract_all_metrics(data_dict):
    results = {}
    columns = [
        ('E_smiles', 'E_True_Labels', 'E_Tabular_Predictions'), ('E_smiles','E_True_Labels', 'E_GNN_Predictions'), ('E_smiles','E_True_Labels', 'E_EdgeConv_Predictions'),
        ('A_smiles', 'A_True_Labels', 'A_Tabular_Predictions'), ('A_smiles','A_True_Labels', 'A_GNN_Predictions'), ('A_smiles','A_True_Labels', 'A_EdgeConv_Predictions'),
        ('P_smiles', 'P_True_Labels', 'P_Tabular_Predictions'), ('P_smiles', 'P_True_Labels', 'P_GNN_Predictions'), ('P_smiles', 'P_True_Labels', 'P_EdgeConv_Predictions'),
        ('O_smiles', 'O_True_Labels', 'O_Tabular_Predictions'), ('O_smiles', 'O_True_Labels', 'O_GNN_Predictions'), ('O_smiles','O_True_Labels', 'O_EdgeConv_Predictions'),
        ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_Tabular_Predictions'), ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_GNN_Predictions'), ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_EdgeConv_Predictions')
    ]
    
    for smiles, true_label, pred in columns:
        true_labels, predictions, errors, mae, rmse, r_squared, smiles = extract_preds(data_dict[smiles], data_dict[true_label] , data_dict[true_label], data_dict[pred])
        results[pred] = {
            'true_labels': true_labels, 'predictions': predictions, 'errors': errors, 'mae': mae, 'rmse': rmse, 'r_squared': r_squared, 'smiles':smiles
        }
    return results


def extract_all_metrics_with_ensemble(data_dict):
    results = {}
    columns = [
        ('E_smiles', 'E_True_Labels', 'E_Tabular_Predictions'), ('E_smiles','E_True_Labels', 'E_GNN_Predictions'), ('E_smiles','E_True_Labels', 'E_EdgeConv_Predictions'), ('E_smiles','E_True_Labels', 'E_Ensemble_Predictions'),
        ('A_smiles', 'A_True_Labels', 'A_Tabular_Predictions'), ('A_smiles', 'A_True_Labels', 'A_GNN_Predictions'), ('A_smiles','A_True_Labels', 'A_EdgeConv_Predictions'), ('A_smiles','A_True_Labels', 'A_Ensemble_Predictions'),
        ('P_smiles', 'P_True_Labels', 'P_Tabular_Predictions'), ('P_smiles','P_True_Labels', 'P_GNN_Predictions'), ('P_smiles','P_True_Labels', 'P_EdgeConv_Predictions'), ('P_smiles', 'P_True_Labels', 'P_Ensemble_Predictions'),
        ('O_smiles', 'O_True_Labels', 'O_Tabular_Predictions'), ('O_smiles','O_True_Labels', 'O_GNN_Predictions'), ('O_smiles','O_True_Labels', 'O_EdgeConv_Predictions'), ('O_smiles','O_True_Labels', 'O_Ensemble_Predictions'),
        ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_Tabular_Predictions'), ('EOAP_smiles','EOAP_True_Labels', 'EOAP_GNN_Predictions'), ('EOAP_smiles','EOAP_True_Labels', 'EOAP_EdgeConv_Predictions'), ('EOAP_smiles', 'EOAP_True_Labels', 'EOAP_Ensemble_Predictions')
    ]
    
    for smiles, true_label, pred in columns:
        true_labels, predictions, errors, mae, rmse, r_squared, smiles = extract_preds(data_dict[smiles], data_dict[true_label], data_dict[pred])
        results[pred] = {
            'true_labels': true_labels, 'predictions': predictions, 'errors': errors, 'mae': mae, 'rmse': rmse, 'r_squared': r_squared, 'smiles': smiles
        }
    return results




metrics = extract_all_metrics_with_ensemble(data_dict_transfer_loaded)


# Extract all elements into variables
for prefix in ['E', 'A', 'P', 'O', 'EOAP']:
    tab_pred = f'{prefix}_Tabular_Predictions'
    gnn_pred = f'{prefix}_GNN_Predictions'
    edge_pred = f'{prefix}_EdgeConv_Predictions'
    ens_pred = f'{prefix}_Ensemble_Predictions'

    # Individual model metrics and predictions
    globals()[f'{prefix}_True_Labels'] = metrics[tab_pred]['true_labels']
    globals()[f'{prefix}_Tabular_Predictions'] = metrics[tab_pred]['predictions']
    globals()[f'{prefix}_Tabular_MAE'] = metrics[tab_pred]['mae']
    globals()[f'{prefix}_Tabular_RMSE'] = metrics[tab_pred]['rmse']
    globals()[f'{prefix}_Tabular_R_squared'] = metrics[tab_pred]['r_squared']
    
    globals()[f'{prefix}_GNN_Predictions'] = metrics[gnn_pred]['predictions']
    globals()[f'{prefix}_GNN_MAE'] = metrics[gnn_pred]['mae']
    globals()[f'{prefix}_GNN_RMSE'] = metrics[gnn_pred]['rmse']
    globals()[f'{prefix}_GNN_R_squared'] = metrics[gnn_pred]['r_squared']
    
    globals()[f'{prefix}_EdgeConv_Predictions'] = metrics[edge_pred]['predictions']
    globals()[f'{prefix}_EdgeConv_MAE'] = metrics[edge_pred]['mae']
    globals()[f'{prefix}_EdgeConv_RMSE'] = metrics[edge_pred]['rmse']
    globals()[f'{prefix}_EdgeConv_R_squared'] = metrics[edge_pred]['r_squared']
    
    # Ensemble model metrics and predictions
    globals()[f'{prefix}_Ensemble_True_Labels'] = metrics[ens_pred]['true_labels']
    globals()[f'{prefix}_Ensemble_Predictions'] = metrics[ens_pred]['predictions']
    globals()[f'{prefix}_Ensemble_MAE'] = metrics[ens_pred]['mae']
    globals()[f'{prefix}_Ensemble_RMSE'] = metrics[ens_pred]['rmse']
    globals()[f'{prefix}_Ensemble_R_squared'] = metrics[ens_pred]['r_squared']

