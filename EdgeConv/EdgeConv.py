import os
import glob
import numpy as np
import math
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from random import sample, randint
# Functions from other files
from Utils_PC_Processing import normalize_train_test_xyz_esp
from Utils_PC_Processing import plot_comparison
from torch.utils.data import DataLoader, TensorDataset
from random import sample, randint

"""
 Path to ESP maps (PC) data
"""
PC_path = "/project/6033373/mghanava/Deep-learning/EdgeConv_solubility/Paper_Results/PC_PHYS_NoI.npy"
labels_path = "/project/6033373/mghanava/Deep-learning/EdgeConv_solubility/Paper_Results/Labels_PHYS_NoI.csv"

PC = np.load(PC_path)
Labels_stand = pd.read_csv(labels_path)
Labels = Labels_stand['logS'].to_numpy()

esol_string = os.path.splitext(os.path.basename(labels_path))[0]
Data_Name = esol_string.split('_')[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Fixing the size of pointcloud data to minimum number of Points amoung all point clouds
Min_points_No = 3000

def Fix_size(data_PointCloud, Min_points_No):
    
     PointCloud_fixedSize = []
     for count , my_data in enumerate(data_PointCloud):
        num_points = np.shape(my_data)[0]
        rand_idx = sample(range(0, num_points), Min_points_No)
        my_data = my_data[rand_idx,:]
        PointCloud_fixedSize.append(my_data)
    
     return PointCloud_fixedSize


PC = Fix_size(PC, Min_points_No)

PC = np.array(PC)


# Train-test split
PC_train, PC_test, Labels_train, Labels_test = train_test_split(
    PC, Labels, random_state=104, test_size=0.2, shuffle=True)




PC_train_norm, PC_test_norm = normalize_train_test_xyz_esp(PC_train, PC_test)



batch_size = 4 

num_full_batches = len(PC_train_norm) // batch_size

subselected_size = num_full_batches * batch_size


# Take 1000 samples from training set
# Take 1000 samples from training set
PC_train_norm_sampled = PC_train_norm[:]
Labels_train_sampled = Labels_train[:]

# Take 1000 samples from test set
PC_test_norm_sampled = PC_test_norm[:]
Labels_test_sampled = Labels_test[:]


print("Sizes of Sampled Training Data:")
print("PC_train_norm_sampled:", np.shape(PC_train_norm))


print("Sizes of Full Test Data:")
print("PC_test_norm:", np.shape(PC_test_norm))


PC_train_norm = PC_train_norm_sampled
Labels_train = Labels_train_sampled

# Take 1000 samples from test set
PC_test_norm = PC_test_norm_sampled
Labels_test = Labels_test_sampled


Num_classes = 1
Num_Points = np.shape(PC)[1]
Batch_Size = 4
num_features = 4
num_points = np.shape(PC)[1]

print(f'batch Size is: {Batch_Size}')



# Convert data to PyTorch tensors

# Specify the device (cuda if GPU, otherwise cpu)
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
            
    device = torch.device('cuda')
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

knn_out_1 = knn(x_data_1.permute(0,2,1), 20)

DynGraph = get_graph_feature(x_data_1.permute(0,2,1), k=20)



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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Instantiate the model and move it to GPU
model = EdgeConv().to(device)


print(model)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
#optimizer = optim.Adam(model.parameters(), lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
     


def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    train_losses = []
    for batch_idx, batch in enumerate(dataloader):
        inputs_1, labels = batch
        # Move inputs and labels to GPU
        inputs_1, labels = inputs_1.to(device), labels.to(device)
        
        pred = model(inputs_1)
        loss = loss_fn(pred, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
      

        if batch_idx % 100 == 0:
           print(f"Current MSE loss: {loss.item()}")

    # Calculate and print Mean Squared Error (MSE) after training
    rmse = math.sqrt(sum(train_losses) / len(train_losses))
    print(f"RMSE for train: {rmse}")

    return [sum(train_losses) / len(dataloader)], [rmse]



def test_epoch(dataloader, model, loss_fn):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in dataloader:
            inputs_1, labels = batch
            # Move inputs and labels to GPU
            inputs_1, labels = inputs_1.to(device), labels.to(device)
            pred = model(inputs_1)
            test_losses.append(loss_fn(pred, labels.unsqueeze(1)).item())


    if len(test_losses) == 0:
        print("No test data.")
        return 0, 0

    # Calculate and print Root Mean Squared Error (RMSE) after testing
    rmse = math.sqrt(sum(test_losses) / len(test_losses))
    print(f"RMSE for Test RMSE: {rmse}")

    return [sum(test_losses) / len(dataloader)], [rmse]

EPOCHS = 120

train_losses = []
test_losses = []
early_stop_patience = 30 # Number of epochs to wait for improvement before early stopping

best_test_rmse = 100
best_epoch = 0
stop_counter = 0
epoch_list = []
train_loss_list = []
train_balanced_acc_list = []

test_loss_list = []
test_balanced_acc_list = []
train_rmse_list =[]
test_rmse_list =[]

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

best_model_path = f"best_model_sol_{Data_Name}_{current_time}.pth"

for epoch in range(EPOCHS):
    print(f"Epoch Nr. {epoch+1} -----------------------")
    train_epoch_results = train_epoch(train_dataloader, model, loss_fn, optimizer)
    
    train_loss = train_epoch_results[0]
    train_rmse = train_epoch_results[1]
    test_loss, test_rmse = test_epoch(test_dataloader, model, loss_fn)
    
    print(f'test RMSE: {test_rmse}')
    print(f'best RMSE: {best_test_rmse}')
    
    # Check for improvement in test Error
    if test_rmse[0] < best_test_rmse:
        best_test_rmse = test_rmse[0]
        best_epoch = epoch
        stop_counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
    else:
        stop_counter += 1
    
    # Check for early stopping
    if stop_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1} as test error did not improve.")
        break
    
    lr_scheduler.step()
    
    epoch_list.append(epoch + 1)
    train_loss_list.append(train_loss)
    train_rmse_list.append(train_rmse)
    test_loss_list.append(test_loss)
    test_rmse_list.append(test_rmse)


# Load the best model for evaluation

best_model = EdgeConv().to(device)

best_model.load_state_dict(torch.load(best_model_path))



# Create a DataFrame
data = {
    'Epoch': epoch_list,
    'Train Loss': train_loss_list,
    'Train RMSE': train_rmse_list,
    'Test Loss': test_loss_list,
    'Test RMSE': test_rmse_list,
}

df = pd.DataFrame(data)


# Save the DataFrame to an Excel file
df.to_excel(f'metrics_{Data_Name}_{current_time}.xlsx', index=False)


model = best_model
# Evaluate the model on the test set
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs_1, labels = batch
        inputs_1, labels = inputs_1.to(device), labels.to(device)
        pred = model(inputs_1)
        
        # Append predictions and true labels
        predictions.extend(pred.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
mse = mean_squared_error(true_labels, predictions)
rmse = math.sqrt(mse)
mae = mean_absolute_error(true_labels, predictions)
r_squared = r2_score(true_labels, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


results_df = pd.DataFrame({
    'MSE': [mse] * len(true_labels),
    'RMSE': [rmse] * len(true_labels),
    'MAE': [mae] * len(true_labels),
    'R-squared': [r_squared] * len(true_labels),
    'True Labels': true_labels,
    'Predictions': predictions
})


# Save the DataFrame to an Excel file
results_df.to_excel(f'results_sol_{Data_Name}_{current_time}.xlsx', index=False)

# Print additional evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Print confirmation
print("Results saved successfully.")
