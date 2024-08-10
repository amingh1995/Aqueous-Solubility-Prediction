
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


AB_Features_EOAP_NoI  = np.load(f"{path_Numpy}/AB_VAS_Features_E_A_P_O_NoI.npy")
Labels_EOAP_NoI= pd.read_csv(f"{path_Numpy}/Labels_E_A_P_O_NoI.csv")

smiles_E = Labels_ESOL_NoI['smiles'].tolist()
smiles_A = Labels_AQUA_NoI['smiles'].tolist()
smiles_P = Labels_PHYS_NoI['smiles'].tolist()
smiles_O = Labels_OCHEM_NoI['smiles'].tolist()
smiles_EAPO = Labels_EOAP_NoI['smiles'].tolist()



AB_Features_data = AB_Features_EOAP_NoI
Labels_data = Labels_OCHEM_NoI


smiles = Labels_data['smiles'].tolist()
y_real_list = Labels_data['logS'].tolist()
y = np.array(y_real_list)


a = Labels_ESOL_NoI['logS'].tolist()


"""RDkit_Descriptors"""


""" Mordred"""

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



X_EOAP = np.concatenate((Mordred_EOAP_NoI, AB_Features_EOAP_NoI), axis=1)
X_ESOL_NoI = np.concatenate((Mordred_ESOL_NoI, AB_Features_ESOL_NoI), axis=1)
X_PHYS_NoI = np.concatenate((Mordred_PHYS_NoI, AB_Features_PHYS_NoI), axis=1)
X_AQUA_NoI = np.concatenate((Mordred_AQUA_NoI, AB_Features_AQUA_NoI), axis=1)
X_OCHEM_NoI = np.concatenate((Mordred_OCHEM_NoI, AB_Features_OCHEM_NoI), axis=1)



calc = Calculator(descriptors, ignore_3D=False)
Mordred_feature_names = [str(descriptor) for descriptor in calc.descriptors]

X = X_EOAP
X = Mordred_EOAP_NoI
X = AB_Features_EOAP_NoI
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
threshold = 1e-4



# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})




def AB_Mord_ML (Mordred_Feat, AB_Feat, Labels_data_csv):
      
    
    calc = Calculator(descriptors, ignore_3D=False)
    Mordred_feature_names = [str(descriptor) for descriptor in calc.descriptors]
    AB_feature_names = [f'Alpha_{i}' for i in range(1, 55)] + [f'Beta_{i}' for i in range(1, 23)] + ['V', 'Area', 'sph']
    
    
    AB_top_n_feature_indices = np.load(os.path.join(path_save_model, 'AB_top_n_feature_indices.npy'))
    M_top_n_feature_indices = np.load(os.path.join(path_save_model, 'M_top_n_feature_indices.npy'))
    
    
    """
    ESOL
    """
    
    X_M = Mordred_Feat
    X_AB = AB_Feat
    y_real_list = Labels_data_csv['logS'].tolist()
    y = np.array(y_real_list)
    
    
    X_df_AB = pd.DataFrame(X_AB, columns=AB_feature_names)
    X_df_M = pd.DataFrame(X_M, columns=Mordred_feature_names)
    
    y_df = pd.DataFrame(y, columns=['target'])
    
    
    # Split the dataset into training and testing sets
    X_train_AB, X_test_AB, y_train_AB, y_test_AB = train_test_split(
        X_df_AB,
        y_df,
        test_size=0.2,
        random_state=104
    )
    
    X_train_M, X_test_M, y_train_M, y_test_M = train_test_split(
        X_df_M,
        y_df,
        test_size=0.2,
        random_state=104
    )
    
    
    
    X_train_selected_AB = X_train_AB.iloc[:, AB_top_n_feature_indices]
    X_test_selected_AB = X_test_AB.iloc[:, AB_top_n_feature_indices]
    
    X_train_selected_M = X_train_M.iloc[:, M_top_n_feature_indices]
    X_test_selected_M = X_test_M.iloc[:, M_top_n_feature_indices]
    
    X_train_selected_nonscaled = np.concatenate((X_train_selected_M, X_train_selected_AB), axis =1)
    X_test_selected_nonscaled = np.concatenate((X_test_selected_M, X_test_selected_AB), axis = 1)
    
    # Instantiate the MinMaxScaler
    scaler = MinMaxScaler()
    X_train_selected = scaler.fit_transform(X_train_selected_nonscaled)
    X_test_selected = scaler.transform(X_test_selected_nonscaled)
    
    
    y_train = y_train_M
    y_test = y_test_M
    
    
    
    """ Train """
    # Select features with importance above the threshold
    #selected_features = feature_names[feature_importances > threshold]
    
    
    # Set hyperparameters
    hyperparameters = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}
    model = GradientBoostingRegressor(**hyperparameters, random_state=42)
    model.fit(X_train_selected, y_train)
    
    
    
    test_predictions_selected = model.predict(X_test_selected)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, test_predictions_selected)
    mae = mean_absolute_error(y_test, test_predictions_selected)
    r2 = r2_score(y_test, test_predictions_selected)
    rmse_knn = np.sqrt(mean_squared_error(y_test, test_predictions_selected))
    
    output_range_test = y_test.max() - y_test.min()
    
    rRMSE = np.sqrt(mse) / output_range_test
    
    print("Relative RMSE (rRMSE):", rRMSE)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)
    print("RMSE:", rmse_knn)
    
    
    return y_test, test_predictions_selected


E_True_Labels, E_Predictions = AB_Mord_ML(Mordred_ESOL_NoI, AB_Features_ESOL_NoI, Labels_ESOL_NoI)
A_True_Labels, A_Predictions = AB_Mord_ML(Mordred_AQUA_NoI, AB_Features_AQUA_NoI, Labels_AQUA_NoI)
P_True_Labels, P_Predictions = AB_Mord_ML(Mordred_PHYS_NoI, AB_Features_PHYS_NoI, Labels_PHYS_NoI)
O_True_Labels, O_Predictions = AB_Mord_ML(Mordred_OCHEM_NoI, AB_Features_OCHEM_NoI, Labels_OCHEM_NoI)
EOAP_True_Labels, EOAP_Predictions = AB_Mord_ML(Mordred_EOAP_NoI, AB_Features_EOAP_NoI, Labels_EOAP_NoI)


E_True_Labels, A_True_Labels, P_True_Labels, O_True_Labels, EOAP_True_Labels = E_True_Labels.values.flatten(), A_True_Labels.values.flatten(), P_True_Labels.values.flatten(),  O_True_Labels.values.flatten(), EOAP_True_Labels.values.flatten()
E_True_Labels = E_True_Labels.values.flatten()
A_True_Labels = A_True_Labels.values.flatten()
P_True_Labels = P_True_Labels.values.flatten()
O_True_Labels = O_True_Labels.values.flatten()
EOAP_True_Labels = EOAP_True_Labels.values.flatten()


"""Save them"""
# Create DataFrames for each case
df_E = pd.DataFrame({
    'True_Labels_E': E_True_Labels,
    'Predictions_E': E_Predictions
})

df_A = pd.DataFrame({
    'True_Labels_A': A_True_Labels,
    'Predictions_A': A_Predictions
})

df_P = pd.DataFrame({
    'True_Labels_P': P_True_Labels,
    'Predictions_P': P_Predictions
})

df_O = pd.DataFrame({
    'True_Labels_O': O_True_Labels,
    'Predictions_O': O_Predictions
})

df_EOAP = pd.DataFrame({
    'True_Labels_EOAP': EOAP_True_Labels,
    'Predictions_EOAP': EOAP_Predictions
})

# Combine DataFrames along columns
combined_df = pd.concat([df_E, df_A, df_P, df_O, df_EOAP], axis=1)

# Save combined DataFrame to Excel
combined_df.to_excel(f'{path_save_model}/AB_M_feat_test_Pred_results.xlsx', index=False)





