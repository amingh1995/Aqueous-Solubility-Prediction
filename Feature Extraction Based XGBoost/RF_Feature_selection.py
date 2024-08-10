

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import os
from sklearn.preprocessing import  MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from mordred import Calculator, descriptors
import deepchem as dc


path_save_data = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Revised_Processed_Saved_Data_solvent'
path_save_model = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\ML based on Tabular Features'

path_Numpy = path_save_data

"""
Loading alpha beta Volume, Area, Sphericity features extracted from ESP maps
"""

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




AB_Features_data = AB_Features_E_A_P_O_NoI
Labels_data = Labels_E_A_P_O_NoI


smiles = Labels_data['smiles'].tolist()
y_real_list = Labels_data['logS'].tolist()
y = np.array(y_real_list)


a = Labels_ESOL_NoI['logS'].tolist()


"""RDkit_Descriptors"""


""" Mordred"""

featurizer = dc.feat.MordredDescriptors(ignore_3D=False)
smiles_EAPO = Labels_E_A_P_O_NoI['smiles'].tolist()
Mordred_EAPO = featurizer.featurize(smiles_EAPO)


smiles_ESOL = Labels_ESOL_NoI['smiles'].tolist()
Mordred_ESOL_NoI = featurizer.featurize(smiles_ESOL)

smiles_PHYS = Labels_PHYS_NoI['smiles'].tolist()
Mordred_PHYS_NoI = featurizer.featurize(smiles_PHYS)

smiles_AQUA = Labels_AQUA_NoI['smiles'].tolist()
Mordred_AQUA_NoI = featurizer.featurize(smiles_AQUA)

smiles_OCHEM = Labels_OCHEM_NoI['smiles'].tolist()
Mordred_OCHEM_NoI = featurizer.featurize(smiles_OCHEM)



XـEAPO = np.concatenate((Mordred_EAPO, AB_Features_E_A_P_O_NoI), axis=1)
X_ESOL_NoI = np.concatenate((Mordred_ESOL_NoI, AB_Features_ESOL_NoI), axis=1)
X_PHYS_NoI = np.concatenate((Mordred_PHYS_NoI, AB_Features_PHYS_NoI), axis=1)
X_AQUA_NoI = np.concatenate((Mordred_AQUA_NoI, AB_Features_AQUA_NoI), axis=1)
X_OCHEM_NoI = np.concatenate((Mordred_OCHEM_NoI, AB_Features_OCHEM_NoI), axis=1)



calc = Calculator(descriptors, ignore_3D=False)
Mordred_feature_names = [str(descriptor) for descriptor in calc.descriptors]

X = XـEAPO
X = Mordred_EAPO
X = AB_Features_E_A_P_O_NoI
X = X
y = y


AB_feature_names = [f'Alpha_{i}' for i in range(1, 55)] + [f'Beta_{i}' for i in range(1, 23)] + ['V', 'Area', 'sph']

X_df = pd.DataFrame(X, columns=Mordred_feature_names+AB_feature_names)
X_df = pd.DataFrame(X, columns=AB_feature_names)
X_df = pd.DataFrame(X, columns=Mordred_feature_names)

y_df = pd.DataFrame(y, columns=['target'])

data_df = pd.concat([X_df, y_df], axis=1)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.2,
    random_state=104
)




# Instantiate the MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Instantiate the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train_scaled, y_train)


# Get feature importances
feature_importances = rf_model.feature_importances_


# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
top_features = feature_importance_df.nlargest(500, 'Importance')

# Plot the feature importance diagram
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 5 Feature Importance')
plt.show()



AB_top_n_feature_indices = feature_importance_df.nlargest(25, 'Importance')['Feature'].index
X_train_selected_AB = X_train_scaled[:, AB_top_n_feature_indices]
X_test_selected_AB = X_test_scaled[:, AB_top_n_feature_indices]


M_top_n_feature_indices = feature_importance_df.nlargest(500, 'Importance')['Feature'].index
X_train_selected_M = X_train_scaled[:, M_top_n_feature_indices]
X_test_selected_M = X_test_scaled[:, M_top_n_feature_indices]


AB_top_n_feature_indices = np.load(os.path.join(path_save_model, 'AB_top_n_feature_indices.npy'))
M_top_n_feature_indices = np.load(os.path.join(path_save_model, 'M_top_n_feature_indices.npy'))

