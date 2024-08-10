
import torch 
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Dataset, Data

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import numpy as np
from Graph_Featurizer import MoleculeDataset
from sklearn.model_selection import train_test_split


def GNN_Fun_Train(Data_root_path, file_name_csv):
    
    root_path = Data_root_path
    filename_csv = file_name_csv
    dataset_molnet = MoleculeDataset(root=root_path, filename=filename_csv)
    dataset_graphs = dataset_molnet
    
    
    Data_Name = filename_csv.split('_')[1]



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
    
    # Prepare the data
    data_size = len(dataset_graphs)
    NUM_GRAPHS_PER_BATCH = 64  # batch size
    

    data_loader = DataLoader(dataset_graphs)
    
    # Extract features, edge indices, edge attributes, labels, and SMILES information from the DataLoader
    features = [graph.x for graph in data_loader.dataset]
    edge_indices = [graph.edge_index for graph in data_loader.dataset]
    edge_attrs = [graph.edge_attr for graph in data_loader.dataset]
    labels = [graph.y for graph in data_loader.dataset]
    smiles_info = [graph.smiles for graph in data_loader.dataset]
    
    # Perform train-test split while keeping the same order
    features_train, features_test, edge_indices_train, edge_indices_test, \
    edge_attrs_train, edge_attrs_test, labels_train, labels_test, \
    smiles_train, smiles_test = train_test_split(
        features, edge_indices, edge_attrs, labels, smiles_info,
        random_state=104, test_size=0.2, shuffle=True)
    
    
    data_train = [Data(x=features_train[i], edge_index=edge_indices_train[i], edge_attr=edge_attrs_train[i], y=labels_train[i], smiles=smiles_train[i]) for i in range(len(features_train))]
    
    data_test = [Data(x=features_test[i], edge_index=edge_indices_test[i], edge_attr=edge_attrs_test[i], y=labels_test[i], smiles=smiles_test[i]) for i in range(len(features_test))]
    
    train_loader = DataLoader(data_train, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    val_loader = test_loader
    
    
    # Initialize the GNN model
    input_dim = dataset_graphs.num_features
    hidden_dim = 128
    output_dim = 1  # Single scalar output for solubility prediction
    model = MPNNModel(input_dim, hidden_dim, output_dim)
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression task
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    train_iter = iter(train_loader)
    first_sample = next(train_iter)[10].x.tolist()
    
    
    
    best_model = None
    best_val_rmse = float('inf')
    
    num_epochs = 1000
    
    

    # Other parameters
    patience = 50  # Number of epochs with no improvement to wait before early stopping
    early_stop_counter = 0
    
    
    best_model = None
    best_val_rmse = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
    
        # Training loop
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
    
        # Calculate training loss
        avg_loss = total_loss / len(train_loader.dataset)
        rmse_train = torch.sqrt(torch.tensor(avg_loss))
    
        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                val_output = model(val_batch.x, val_batch.edge_index, val_batch.batch)
                val_loss = criterion(val_output, val_batch.y.view(-1, 1).float())
                val_losses.append(val_loss.item())
    
        # Calculate validation RMSE
        avg_val_loss = sum(val_losses) / len(val_losses)
        rmse_val = torch.sqrt(torch.tensor(avg_val_loss))  # Convert to tensor before applying square root
    
        # Update best model if the current validation RMSE is lower
        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            best_model = model.state_dict()
            early_stop_counter = 0  # Reset the counter when a better model is found
        else:
            early_stop_counter += 1
    
      
    
        # Print and log to TensorBoard every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss (MSE): {avg_loss:.4f}, Training RMSE: {rmse_train:.4f}, Validation Loss (MSE): {avg_val_loss:.4f}, Validation Loss (RMSE): {rmse_val:.4f}")
    
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as there is no improvement for {patience} consecutive epochs.")
            break
    
    # Save the best model
    if best_model is not None:
        model.load_state_dict(best_model)
        save_path = "./best_model.pth"  # Specify your desired path
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")
    
    # Close TensorBoard writer
    
    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    
    # Set the model to evaluation mode
    model.eval()
    
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
    
    # Compute evaluation metrics
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    r_squared = r2_score(true_labels, predictions)
    print(f"R-squared: {r_squared:.4f}")
    
    predictions_flat = np.array(predictions).flatten()
    
    errors = predictions_flat - np.array(true_labels)
    
    return true_labels , predictions_flat
    
