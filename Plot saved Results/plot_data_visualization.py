
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import deepchem as dc




path_save_data = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Revised_Processed_Saved_Data_solvent'
# Save the figure
save_path = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\ML based on Tabular Features\Plot Reults'
dpi_value = 300  # Set DPI value


path_Numpy = path_save_data

AB_Features_ESOL_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_ESOL_NoI.npy")
Labels_ESOL_NoI= pd.read_csv(f"{path_Numpy}/Labels_ESOL_NoI.csv")

AB_Features_PHYS_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_PHYS_NoI.npy")
Labels_PHYS_NoI= pd.read_csv(f"{path_Numpy}/Labels_PHYS_NoI.csv")


AB_Features_AQUA_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_AQUA_NoI.npy")
Labels_AQUA_NoI= pd.read_csv(f"{path_Numpy}/Labels_AQUA_NoI.csv")


AB_Features_OCHEM_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_OCHEM_NoI.npy")
Labels_OCHEM_NoI= pd.read_csv(f"{path_Numpy}/Labels_OCHEM_NoI.csv")


AB_Features_E_A_P_O_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_E_A_P_O_NoI.npy")
Labels_E_A_P_O_NoI= pd.read_csv(f"{path_Numpy}/Labels_E_A_P_O_NoI.csv")


smiles_E = Labels_ESOL_NoI['smiles'].tolist()
smiles_A = Labels_AQUA_NoI['smiles'].tolist()
smiles_P = Labels_PHYS_NoI['smiles'].tolist()
smiles_O = Labels_OCHEM_NoI['smiles'].tolist()
smiles_EAPO = Labels_E_A_P_O_NoI['smiles'].tolist()



# Find SMILES in smiles_E that are not in smiles_EAPO
missing_in_E = [smile for smile in smiles_E if smile not in smiles_EAPO]



AB_Features_data = AB_Features_E_A_P_O_NoI
Labels_data = Labels_E_A_P_O_NoI


name = Labels_data['name'].tolist()
smiles = Labels_data['smiles'].tolist()
y_real_list = Labels_data['logS'].tolist()
y = np.array(y_real_list)


a = Labels_ESOL_NoI['logS'].tolist()

"""RDkit_Descriptors"""



""" Mordred"""
import deepchem as dc

save_directory = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\ML based on Tabular Features'


loaded_arrays = np.load(f'{save_directory}\\data_arrays.npz')

# Access individual arrays
X_E_A_P_O_NoI = loaded_arrays['X_E_A_P_O_NoI']
X_ESOL_NoI = loaded_arrays['X_ESOL_NoI']
X_PHYS_NoI = loaded_arrays['X_PHYS_NoI']
X_AQUA_NoI = loaded_arrays['X_AQUA_NoI']
X_OCHEM_NoI = loaded_arrays['X_OCHEM_NoI']



# Concatenate all datasets
X_combined = np.concatenate((X_E_A_P_O_NoI, X_ESOL_NoI, X_PHYS_NoI, X_AQUA_NoI,X_OCHEM_NoI), axis=0)
y_combined = np.concatenate((np.zeros(X_E_A_P_O_NoI.shape[0]),
                             np.ones(X_ESOL_NoI.shape[0]),
                             2 * np.ones(X_PHYS_NoI.shape[0]),
                             3 * np.ones(X_AQUA_NoI.shape[0]),
                             4 * np.ones(X_OCHEM_NoI.shape[0])), axis=0)

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42)

X_combined_tsne = tsne.fit_transform(X_combined)

# Separate the transformed data for individual datasets
X_E_A_P_O_NoI_tsne = X_combined_tsne[:X_E_A_P_O_NoI.shape[0], :]
X_ESOL_NoI_tsne = X_combined_tsne[X_E_A_P_O_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0], :]
X_PHYS_NoI_tsne = X_combined_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0], :]
X_AQUA_NoI_tsne = X_combined_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]+X_AQUA_NoI.shape[0], :]
X_OCHEM_NoI_tsne = X_combined_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]+X_AQUA_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]+X_AQUA_NoI.shape[0]+X_OCHEM_NoI.shape[0], :]




print("X_E_A_P_O_NoI_tsne shape:", X_E_A_P_O_NoI_tsne.shape)
print("X_ESOL_NoI_tsne shape:", X_ESOL_NoI_tsne.shape)
print("X_PHYS_NoI_tsne shape:", X_PHYS_NoI_tsne.shape)
print("X_AQUA_NoI_tsne shape:", X_AQUA_NoI_tsne.shape)
print("X_OCHEM_NoI_tsne shape:", X_OCHEM_NoI_tsne.shape)



plot_path = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Plot results"

"""Hexbin plot Density for ESOL, AQUA, PHYS, etc on whole molecules""" 
"""In the context of a hexbin plot, "Density" typically refers to the number of data points within each hexagon. In the provided code:"""

# Hexbin plot
fig, axes = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
scatter1 = axes.hexbin(X_E_A_P_O_NoI_tsne[:, 0], X_E_A_P_O_NoI_tsne[:, 1], gridsize=40, cmap='Blues', mincnt=1, label='All Data')
axes.set_title('')
axes.set_xlabel('t-SNE 1')
axes.set_ylabel('t-SNE 2')


"""Scatter ESOL"""
scatter_ESOL = axes.scatter(X_ESOL_NoI_tsne[:, 0], X_ESOL_NoI_tsne[:, 1], s=10, color='green', alpha=0.5, label='ESOL')
plt.legend()
plt.savefig(os.path.join(save_path, 't-SNE_Plot_Overlay.png'), bbox_inches='tight', dpi=300)



"""Scatter PHYS"""
scatter_PHYS = axes.scatter(X_PHYS_NoI_tsne[:, 0], X_PHYS_NoI_tsne[:, 1], s=10, color='orange', alpha=0.5, label='PHYS')
plt.legend()
plt.savefig(os.path.join(save_path, 't-SNE_Plot_Overlay.png'), bbox_inches='tight', dpi=300)


"""Scatter AQUA"""
scatter_AQUA = axes.scatter(X_AQUA_NoI_tsne[:, 0], X_AQUA_NoI_tsne[:, 1], s=10, color='purple', alpha=0.5, label='AQUA')
plt.legend()
plt.savefig(os.path.join(plot_path, 't-SNE_Plot_Overlay.png'), bbox_inches='tight', dpi=300)
plt.show()






y_E = np.array(Labels_ESOL_NoI['logS'].tolist())
y_A = np.array(Labels_AQUA_NoI['logS'].tolist())
y_P = np.array(Labels_PHYS_NoI['logS'].tolist())
y_EAPO =  np.array(Labels_E_A_P_O_NoI['logS'].tolist())


# Split the data into train and test
X_train_E_A_P_O_NoI, X_test_E_A_P_O_NoI, y_train_EAPO, y_test_EAPO = train_test_split(X_E_A_P_O_NoI, y_EAPO, test_size=0.2, random_state=104)
X_train_ESOL_NoI, X_test_ESOL_NoI, y_train_E, y_test_E = train_test_split(X_ESOL_NoI,y_E, test_size=0.2, random_state=104)
X_train_PHYS_NoI, X_test_PHYS_NoI, y_train_P, y_test_P = train_test_split(X_PHYS_NoI,y_P, test_size=0.2, random_state=104)
X_train_AQUA_NoI, X_test_AQUA_NoI, y_train_A, y_test_A = train_test_split(X_AQUA_NoI, y_A, test_size=0.2, random_state=104)

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform t-SNE for X_train


X_train_combined = np.concatenate((X_train_E_A_P_O_NoI, X_train_ESOL_NoI, X_train_PHYS_NoI, X_train_AQUA_NoI), axis = 0)
X_test_combined = np.concatenate((X_test_E_A_P_O_NoI, X_test_ESOL_NoI, X_test_PHYS_NoI, X_test_AQUA_NoI), axis = 0)

X_combined_train_test = np.concatenate((X_train_combined, X_test_combined), axis = 0)
X_combined_train_test_tsne = tsne.fit_transform(X_combined_train_test)


X_combined_tsne_E_A_P_O_NoI = X_combined_train_test_tsne[:X_E_A_P_O_NoI.shape[0], :]
X_combined_tsne_ESOL_NoI = X_combined_train_test_tsne[X_E_A_P_O_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0], :]
X_combined_tsne_PHYS_NoI = X_combined_train_test_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0], :]
X_combined_tsne_AQUA_NoI = X_combined_train_test_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]+X_AQUA_NoI.shape[0], :]

y_combined_E_A_P_O_NoI = np.concatenate((np.zeros(X_train_E_A_P_O_NoI.shape[0]), np.ones(X_test_E_A_P_O_NoI.shape[0])), axis=0)
y_combined_ESOL_NoI = np.concatenate((np.zeros(X_train_ESOL_NoI.shape[0]), np.ones(X_test_ESOL_NoI.shape[0])), axis=0)
y_combined_PHYS_NoI = np.concatenate((np.zeros(X_train_PHYS_NoI.shape[0]), np.ones(X_test_PHYS_NoI.shape[0])), axis=0)
y_combined_AQUA_NoI = np.concatenate((np.zeros(X_train_AQUA_NoI.shape[0]), np.ones(X_test_AQUA_NoI.shape[0])), axis=0)


X_train_combined_tsne = tsne.fit_transform(X_train_combined)

X_train_tsne_E_A_P_O_NoI = X_train_combined_tsne[:X_train_E_A_P_O_NoI.shape[0], :]
X_train_tsne_ESOL_NoI = X_train_tsne_E_A_P_O_NoI[X_train_E_A_P_O_NoI.shape[0]:X_train_E_A_P_O_NoI.shape[0] + X_train_ESOL_NoI.shape[0], :]
X_train_tsne_PHYS_NoI = X_train_tsne_E_A_P_O_NoI[X_train_E_A_P_O_NoI.shape[0] + X_train_ESOL_NoI.shape[0]:X_train_E_A_P_O_NoI.shape[0] + X_train_ESOL_NoI.shape[0] + X_train_PHYS_NoI.shape[0], :]
X_train_tsne_AQUA_NoI = X_train_tsne_E_A_P_O_NoI[X_train_E_A_P_O_NoI.shape[0] + X_train_ESOL_NoI.shape[0] + X_train_PHYS_NoI.shape[0]:X_train_E_A_P_O_NoI.shape[0] + X_train_ESOL_NoI.shape[0] + X_train_PHYS_NoI.shape[0]+X_train_AQUA_NoI.shape[0], :]


# Fit and transform t-SNE for X_test

X_test_combined_tsne = tsne.fit_transform(X_test_combined)

X_test_tsne_E_A_P_O_NoI = X_test_combined_tsne[:X_test_E_A_P_O_NoI.shape[0], :]
X_test_tsne_ESOL_NoI = X_test_tsne_E_A_P_O_NoI[X_test_E_A_P_O_NoI.shape[0]:X_test_E_A_P_O_NoI.shape[0] + X_test_ESOL_NoI.shape[0], :]
X_test_tsne_PHYS_NoI = X_test_tsne_E_A_P_O_NoI[X_test_E_A_P_O_NoI.shape[0] + X_test_ESOL_NoI.shape[0]:X_test_E_A_P_O_NoI.shape[0] + X_test_ESOL_NoI.shape[0] + X_test_PHYS_NoI.shape[0], :]
X_test_tsne_AQUA_NoI = X_test_tsne_E_A_P_O_NoI[X_test_E_A_P_O_NoI.shape[0] + X_test_ESOL_NoI.shape[0] + X_test_PHYS_NoI.shape[0]:X_test_E_A_P_O_NoI.shape[0] + X_test_ESOL_NoI.shape[0] + X_test_PHYS_NoI.shape[0] + X_test_AQUA_NoI.shape[0], :]





X_train_combined = np.concatenate((X_train_E_A_P_O_NoI, X_train_ESOL_NoI, X_train_PHYS_NoI, X_train_AQUA_NoI), axis = 0)
X_test_combined = np.concatenate((X_test_E_A_P_O_NoI, X_test_ESOL_NoI, X_test_PHYS_NoI, X_test_AQUA_NoI), axis = 0)

X_combined_train_test = np.concatenate((X_train_combined, X_test_combined), axis = 0)
X_combined_train_test_tsne = tsne.fit_transform(X_combined_train_test)


X_combined_tsne_E_A_P_O_NoI = X_combined_train_test_tsne[:X_E_A_P_O_NoI.shape[0], :]
X_combined_tsne_ESOL_NoI = X_combined_train_test_tsne[X_E_A_P_O_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0], :]
X_combined_tsne_PHYS_NoI = X_combined_train_test_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0], :]
X_combined_tsne_AQUA_NoI = X_combined_train_test_tsne[X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]:X_E_A_P_O_NoI.shape[0] + X_ESOL_NoI.shape[0] + X_PHYS_NoI.shape[0]+X_AQUA_NoI.shape[0], :]

y_combined_E_A_P_O_NoI = np.concatenate((np.zeros(X_train_E_A_P_O_NoI.shape[0]), np.ones(X_test_E_A_P_O_NoI.shape[0])), axis=0)
y_combined_ESOL_NoI = np.concatenate((np.zeros(X_train_ESOL_NoI.shape[0]), np.ones(X_test_ESOL_NoI.shape[0])), axis=0)
y_combined_PHYS_NoI = np.concatenate((np.zeros(X_train_PHYS_NoI.shape[0]), np.ones(X_test_PHYS_NoI.shape[0])), axis=0)
y_combined_AQUA_NoI = np.concatenate((np.zeros(X_train_AQUA_NoI.shape[0]), np.ones(X_test_AQUA_NoI.shape[0])), axis=0)




# Arrange all plots in a 2 by 2 subplot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), dpi=dpi_value)

# Plot t-SNE for X_combined with different colors for train and test (E_A_P_O_NoI)
scatter_combined_E_A_P_O_NoI = axes[0, 0].scatter(X_combined_tsne_E_A_P_O_NoI[:, 0], X_combined_tsne_E_A_P_O_NoI[:, 1], c=y_combined_E_A_P_O_NoI, cmap='copper', alpha=0.8)
axes[0, 0].set_title('All Data', fontsize=15)
axes[0, 0].set_xlabel('t-SNE 1', fontsize=15)
axes[0, 0].set_ylabel('t-SNE 2', fontsize=15)
axes[0, 0].legend(handles=scatter_combined_E_A_P_O_NoI.legend_elements()[0], labels=['Train', 'Test'], fontsize=15)

# Plot t-SNE for X_combined with different colors for train and test (ESOL_NoI)
scatter_combined_ESOL_NoI = axes[0, 1].scatter(X_combined_tsne_ESOL_NoI[:, 0], X_combined_tsne_ESOL_NoI[:, 1], c=y_combined_ESOL_NoI, cmap='copper', alpha=0.8)
axes[0, 1].set_title('ESOL', fontsize=15)
axes[0, 1].set_xlabel('t-SNE 1', fontsize=15)
axes[0, 1].set_ylabel('t-SNE 2', fontsize=15)
axes[0, 1].legend(handles=scatter_combined_ESOL_NoI.legend_elements()[0], labels=['Train', 'Test'], fontsize=15)

# Plot t-SNE for X_combined with different colors for train and test (PHYS_NoI)
scatter_combined_PHYS_NoI = axes[1, 0].scatter(X_combined_tsne_PHYS_NoI[:, 0], X_combined_tsne_PHYS_NoI[:, 1], c=y_combined_PHYS_NoI, cmap='copper', alpha=0.8)
axes[1, 0].set_title('PHYS', fontsize=15)
axes[1, 0].set_xlabel('t-SNE 1', fontsize=15)
axes[1, 0].set_ylabel('t-SNE 2', fontsize=15)
axes[1, 0].legend(handles=scatter_combined_PHYS_NoI.legend_elements()[0], labels=['Train', 'Test'], fontsize=15)

# Plot t-SNE for X_combined with different colors for train and test (AQUA_NoI)
scatter_combined_AQUA_NoI = axes[1, 1].scatter(X_combined_tsne_AQUA_NoI[:, 0], X_combined_tsne_AQUA_NoI[:, 1], c=y_combined_AQUA_NoI, cmap='copper', alpha=0.8)
axes[1, 1].set_title('AQUA', fontsize=15)
axes[1, 1].set_xlabel('t-SNE 1', fontsize=15)
axes[1, 1].set_ylabel('t-SNE 2', fontsize=15)
axes[1, 1].legend(handles=scatter_combined_AQUA_NoI.legend_elements()[0], labels=['Train', 'Test'], fontsize=15)

plt.savefig(os.path.join(save_path, 'Train_test_TSNE.png'), bbox_inches='tight', dpi=dpi_value)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



"""Histogram Distribution"""

y_E = np.array(Labels_ESOL_NoI['logS'].tolist())
y_A = np.array(Labels_AQUA_NoI['logS'].tolist())
y_P = np.array(Labels_PHYS_NoI['logS'].tolist())
y_EAPO =  np.array(Labels_E_A_P_O_NoI['logS'].tolist())

# Create subplots for histograms
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), dpi=300)

# Plot histograms for logS values with edged bars
axes[0, 0].hist(y_E, bins=30, edgecolor='black', linewidth=1.2, color='blue', alpha=0.7)
axes[0, 0].set_title('Histogram of logS (ESOL_NoI)')
axes[0, 0].set_xlabel('logS')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(y_A, bins=30, edgecolor='black', linewidth=1.2, color='green', alpha=0.7)
axes[0, 1].set_title('Histogram of logS (AQUA_NoI)')
axes[0, 1].set_xlabel('logS')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(y_P, bins=30, edgecolor='black', linewidth=1.2, color='red', alpha=0.7)
axes[1, 0].set_title('Histogram of logS (PHYS_NoI)')
axes[1, 0].set_xlabel('logS')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(y_EAPO, bins=30, edgecolor='black', linewidth=1.2, color='purple', alpha=0.7)
axes[1, 1].set_title('Histogram of logS (E_A_P_O_NoI)')
axes[1, 1].set_xlabel('logS')
axes[1, 1].set_ylabel('Frequency')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Create a single subplot for overlaid histograms
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot histograms for logS values with edged bars and transparency
ax.hist(y_E, bins=30, edgecolor='black', linewidth=1.2, color='royalblue', alpha=0.99, label='ESOL')
ax.hist(y_A, bins=30, edgecolor='black', linewidth=1.2, color='skyblue', alpha=0.7, label='AQUA')
ax.hist(y_P, bins=30, edgecolor='black', linewidth=1.2, color='royalblue', alpha=0.4, label='PHYS')
ax.hist(y_EAPO, bins=30, edgecolor='black', linewidth=1.2, color='deepskyblue', alpha=0.2, label='All Data')

# Add labels and legend
ax.set_title('')
ax.set_xlabel('logS', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.legend(fontsize=15)

plt.savefig(os.path.join(save_path, 'LogS_Histogram.png'), bbox_inches='tight', dpi=dpi_value)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()





