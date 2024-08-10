
import numpy as np
import seaborn as sns
import pandas as pd 
import os
from Functionalized_Train_datasets import GNN_Fun_Train




save_directory = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets"

root_path_P = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_PHYS"
file_name_csv_P = "Labels_PHYS_NoI.csv"


root_path_E =  r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_ESOL"
file_name_csv_E = "Labels_ESOL_NoI.csv"

root_path_A = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_AQUA"
file_name_csv_A = "Labels_AQUA_NoI.csv"

root_path_O = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_OCHEM"
file_name_csv_O = "Labels_OCHEM_NoI.csv"

root_path_EOAP = r"C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\GNN_all_Datasets\data_EAOP"
file_name_csv_EOAP = "Labels_E_A_P_O_NoI.csv"



"""
Train on Datasets
"""

true_labels_E , predictions_E  = GNN_Fun_Train(root_path_E, file_name_csv_E)
true_labels_A , predictions_A  = GNN_Fun_Train(root_path_A, file_name_csv_A)
true_labels_P , predictions_P  = GNN_Fun_Train(root_path_P, file_name_csv_P)
true_labels_O , predictions_O  = GNN_Fun_Train(root_path_O, file_name_csv_O)
true_labels_EOAP , predictions_EOAP  = GNN_Fun_Train(root_path_EOAP, file_name_csv_EOAP)


combined_df = pd.concat([
    pd.DataFrame({
        'True Labels': true_labels_E,
        'Predictions': predictions_E
    }),
    pd.DataFrame({
        'True Labels': true_labels_A,
        'Predictions': predictions_A
    }),
    pd.DataFrame({
        'True Labels': true_labels_P,
        'Predictions': predictions_P
    }),
    pd.DataFrame({
        'True Labels': true_labels_O,
        'Predictions': predictions_O
    }),
    pd.DataFrame({
        'True Labels': true_labels_EOAP,
        'Predictions': predictions_EOAP
    }),
], axis=1, keys=['E', 'A', 'P', 'O', 'EOAP'], names=['Category'])




df_E_columns = combined_df['E']

true_labels_E = df_E_columns['True Labels'].to_numpy()
predictions_E = df_E_columns['Predictions'].to_numpy()
errors_E = predictions_E - true_labels_E


# Flatten the MultiIndex in the columns
flatten_combined_df = combined_df
flatten_combined_df.columns = [f'{col[0]}_{col[1]}' for col in combined_df.columns]


flatten_combined_df.to_excel(f'{save_directory}/GNN_test_Pred_results.xlsx', index=False)

flatten_combined_df.columns

true_labels_E = flatten_combined_df['E_True Labels'].to_numpy()
predictions_E = flatten_combined_df['E_Predictions'].to_numpy()


valid_indices = ~np.isnan(true_labels_E) & ~np.isnan(predictions_E)
true_labels_E = true_labels_E[valid_indices]
predictions_E = predictions_E[valid_indices]

errors_E = predictions_E - true_labels_E


