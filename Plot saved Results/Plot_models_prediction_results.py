
import os
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import ast
import pickle


"""
Results from three model training files
"""

GNN_results_dir = "GNN_test_Pred_results.xlsx"
EdgeConv_results_dir = "EdgeConv_test_Pred_results.xlsx"
AB_M_feat_results_dir = "AB_M_feat_test_Pred_results.xlsx"  # results from the method was based on extracted features from ESP and Mordred 


"""
Load Dataseet
"""

dataframe_GNN = pd.read_excel(GNN_results_dir)
dataframe_Edge = pd.read_excel(EdgeConv_results_dir)
dataframe_AB_M_feat = pd.read_excel(AB_M_feat_results_dir)

print ('hello world')

#ESOL
def extract_preds(dataframe, true_labels_name, preds_name):
    GNN_true_labels_E = dataframe[f'{true_labels_name}'].to_numpy()
    GNN_predictions_E = dataframe[f'{preds_name}'].to_numpy()
    GNN_valid_indices = ~np.isnan(GNN_true_labels_E) & ~np.isnan(GNN_predictions_E)
    GNN_true_labels_E = GNN_true_labels_E[GNN_valid_indices]
    GNN_predictions_E = GNN_predictions_E[GNN_valid_indices]
    GNN_errors_E = GNN_predictions_E - GNN_true_labels_E
    mae = mean_absolute_error(GNN_true_labels_E, GNN_predictions_E)
    mse = mean_squared_error(GNN_true_labels_E, GNN_predictions_E)
    rmse = np.sqrt(mse)
    r_squared = r2_score(GNN_true_labels_E, GNN_predictions_E)
    return GNN_true_labels_E, GNN_predictions_E, GNN_errors_E, mae, rmse, r_squared


def Edge_extract_preds(dataframe, true_labels_name, preds_name):
    GNN_true_labels_E = dataframe[f'{true_labels_name}'].to_numpy()
    GNN_predictions_E_1 = dataframe[f'{preds_name}'].to_numpy()
    GNN_valid_indices = ~np.isnan(GNN_true_labels_E)
    GNN_true_labels_E = GNN_true_labels_E[GNN_valid_indices]
    GNN_predictions_E_1 = GNN_predictions_E_1[GNN_valid_indices]
    converted_values = [float(ast.literal_eval(element)[0]) for element in GNN_predictions_E_1]
    GNN_predictions_E = np.array(converted_values)
    GNN_errors_E = GNN_predictions_E - GNN_true_labels_E
    mae = mean_absolute_error(GNN_true_labels_E, GNN_predictions_E)
    mse = mean_squared_error(GNN_true_labels_E, GNN_predictions_E)
    rmse = np.sqrt(mse)
    r_squared = r2_score(GNN_true_labels_E, GNN_predictions_E)
    
    return GNN_true_labels_E, GNN_predictions_E, GNN_errors_E, mae, rmse, r_squared


"""GNN results"""
GNN_true_labels_E, GNN_predictions_E, GNN_errors_E, GNN_E_mae, GNN_E_rmse, GNN_E_r_squared = extract_preds(dataframe_GNN, 'E_True_Labels', 'E_Predictions')
GNN_true_labels_A, GNN_predictions_A, GNN_errors_A, GNN_A_mae, GNN_A_rmse, GNN_A_r_squared = extract_preds(dataframe_GNN, 'A_True_Labels', 'A_Predictions')
GNN_true_labels_P, GNN_predictions_P, GNN_errors_P, GNN_P_mae, GNN_P_rmse, GNN_P_r_squared = extract_preds(dataframe_GNN, 'P_True_Labels', 'P_Predictions')
GNN_true_labels_O, GNN_predictions_O, GNN_errors_O, GNN_O_mae, GNN_O_rmse, GNN_O_r_squared = extract_preds(dataframe_GNN, 'O_True_Labels', 'O_Predictions')
GNN_true_labels_EOAP, GNN_predictions_EOAP, GNN_errors_EOAP, GNN_EOAP_mae, GNN_EOAP_rmse, GNN_EOAP_r_squared = extract_preds(dataframe_GNN, 'EOAP_True_Labels', 'EOAP_Predictions')


"""Edge results"""
Edge_true_labels_E, Edge_predictions_E, Edge_errors_E, Edge_E_mae, Edge_E_rmse, Edge_E_r_squared = Edge_extract_preds(dataframe_Edge, 'E_True_Labels', 'E_Predictions')
Edge_true_labels_A, Edge_predictions_A, Edge_errors_A, Edge_A_mae, Edge_A_rmse, Edge_A_r_squared = Edge_extract_preds(dataframe_Edge, 'A_True_Labels', 'A_Predictions')
Edge_true_labels_P, Edge_predictions_P, Edge_errors_P, Edge_P_mae, Edge_P_rmse, Edge_P_r_squared = Edge_extract_preds(dataframe_Edge, 'P_True_Labels', 'P_Predictions')
Edge_true_labels_O, Edge_predictions_O, Edge_errors_O, Edge_O_mae, Edge_O_rmse, Edge_O_r_squared = Edge_extract_preds(dataframe_Edge, 'O_True_Labels', 'O_Predictions')
Edge_true_labels_EOAP, Edge_predictions_EOAP, Edge_errors_EOAP, Edge_EOAP_mae, Edge_EOAP_rmse, Edge_EOAP_r_squared = Edge_extract_preds(dataframe_Edge, 'EOAP_True_Labels', 'EOAP_Predictions')


"""AB_M_feat"""
AB_M_feat_true_labels_E, AB_M_feat_predictions_E, AB_M_feat_errors_E, AB_M_feat_E_mae, AB_M_feat_E_rmse, AB_M_feat_E_r_squared = extract_preds(dataframe_AB_M_feat, 'E_True_Labels', 'E_Predictions')
AB_M_feat_true_labels_A, AB_M_feat_predictions_A, AB_M_feat_errors_A, AB_M_feat_A_mae, AB_M_feat_A_rmse, AB_M_feat_A_r_squared = extract_preds(dataframe_AB_M_feat, 'A_True_Labels', 'A_Predictions')
AB_M_feat_true_labels_P, AB_M_feat_predictions_P, AB_M_feat_errors_P, AB_M_feat_P_mae, AB_M_feat_P_rmse, AB_M_feat_P_r_squared = extract_preds(dataframe_AB_M_feat, 'P_True_Labels', 'P_Predictions')
AB_M_feat_true_labels_O, AB_M_feat_predictions_O, AB_M_feat_errors_O, AB_M_feat_O_mae, AB_M_feat_O_rmse, AB_M_feat_O_r_squared = extract_preds(dataframe_AB_M_feat, 'O_True_Labels', 'O_Predictions')
AB_M_feat_true_labels_EOAP, AB_M_feat_predictions_EOAP, AB_M_feat_errors_EOAP, AB_M_feat_EOAP_mae, AB_M_feat_EOAP_rmse, AB_M_feat_EOAP_r_squared = extract_preds(dataframe_AB_M_feat, 'EOAP_True_Labels', 'EOAP_Predictions')



"""New"""
Saved_Results_dir = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Manuscript_codes\Saved_Results'
with open(f'{Saved_Results_dir}/all_test_predictions_with_ensemble.pkl', 'rb') as f:
    data_dict_loaded = pickle.load(f)


def extract_all_metrics_with_ensemble(data_dict):
    results = {}
    columns = [
        ('E_True_Labels', 'E_Tabular_Predictions'), ('E_True_Labels', 'E_GNN_Predictions'), ('E_True_Labels', 'E_EdgeConv_Predictions'), ('E_True_Labels', 'E_Ensemble_Predictions'),
        ('A_True_Labels', 'A_Tabular_Predictions'), ('A_True_Labels', 'A_GNN_Predictions'), ('A_True_Labels', 'A_EdgeConv_Predictions'), ('A_True_Labels', 'A_Ensemble_Predictions'),
        ('P_True_Labels', 'P_Tabular_Predictions'), ('P_True_Labels', 'P_GNN_Predictions'), ('P_True_Labels', 'P_EdgeConv_Predictions'), ('P_True_Labels', 'P_Ensemble_Predictions'),
        ('O_True_Labels', 'O_Tabular_Predictions'), ('O_True_Labels', 'O_GNN_Predictions'), ('O_True_Labels', 'O_EdgeConv_Predictions'), ('O_True_Labels', 'O_Ensemble_Predictions'),
        ('EOAP_True_Labels', 'EOAP_Tabular_Predictions'), ('EOAP_True_Labels', 'EOAP_GNN_Predictions'), ('EOAP_True_Labels', 'EOAP_EdgeConv_Predictions'), ('EOAP_True_Labels', 'EOAP_Ensemble_Predictions')
    ]
    
    for true_label, pred in columns:
        true_labels, predictions, errors, mae, rmse, r_squared = extract_preds(data_dict[true_label], data_dict[pred])
        results[pred] = {
            'true_labels': true_labels, 'predictions': predictions, 'errors': errors, 'mae': mae, 'rmse': rmse, 'r_squared': r_squared
        }
    return results

metrics = extract_all_metrics_with_ensemble(data_dict_loaded)





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


""" End New """



## Print results: 

def print_metrics(label, rmse, mae, r_squared):
    print(f"{label} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r_squared:.4f}")

print("GNN results:")
print_metrics("E", GNN_E_rmse, GNN_E_mae, GNN_E_r_squared)
print_metrics("A", GNN_A_rmse, GNN_A_mae, GNN_A_r_squared)
print_metrics("P", GNN_P_rmse, GNN_P_mae, GNN_P_r_squared)
print_metrics("EOAP", GNN_EOAP_rmse, GNN_EOAP_mae, GNN_EOAP_r_squared)

print("\nEdge results:")
print_metrics("E", Edge_E_rmse, Edge_E_mae, Edge_E_r_squared)
print_metrics("A", Edge_A_rmse, Edge_A_mae, Edge_A_r_squared)
print_metrics("P", Edge_P_rmse, Edge_P_mae, Edge_P_r_squared)
print_metrics("EOAP", Edge_EOAP_rmse, Edge_EOAP_mae, Edge_EOAP_r_squared)

print("\nAB_M_feat results:")
print_metrics("E", AB_M_feat_E_rmse, AB_M_feat_E_mae, AB_M_feat_E_r_squared)
print_metrics("A", AB_M_feat_A_rmse, AB_M_feat_A_mae, AB_M_feat_A_r_squared)
print_metrics("P", AB_M_feat_P_rmse, AB_M_feat_P_mae, AB_M_feat_P_r_squared)
print_metrics("EOAP", AB_M_feat_EOAP_rmse, AB_M_feat_EOAP_mae, AB_M_feat_EOAP_r_squared)



def rgb_to_hex(rgb):
    """Convert RGB to hexadecimal color code."""
    hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
    return hex_color

# Example usage:
rgb_values = (101, 152, 203)
hex_color_code = rgb_to_hex(rgb_values)
print("RGB:", rgb_values)
print("Hexadecimal:", hex_color_code)





# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi = 300)
fig.suptitle('Comparison of MAE and RMSE for GNN, Edge, and Feature-based (ESOL, AQUA, PHYS, ALL-Data)', fontsize=16)

# Subplot 1: ESOL
bar_width = 0.12  # Width of the bars
bar_positions_metrics = np.arange(len(['MAE', 'RMSE'])) / 1.5

# Update bar positions and number of bars
bar_positions_e_GNN = bar_positions_metrics+ bar_width
bar_positions_e_Edge = bar_positions_metrics 
bar_positions_e_AB_M = bar_positions_metrics + bar_width + bar_width

# Define colors
color_GNN = rgb_to_hex((51, 90, 153))
color_Edge = rgb_to_hex((77, 148, 255))
color_AB_M = rgb_to_hex((200, 220, 255))

# Plot MAE and RMSE bars next to each other with borders
axes[0, 0].bar(bar_positions_e_Edge, [Edge_E_mae, Edge_E_rmse], width=bar_width, label='EdgeConv', color=color_Edge, edgecolor='black', linewidth=1)
axes[0, 0].bar(bar_positions_e_GNN, [GNN_E_mae, GNN_E_rmse], width=bar_width, label='GCN', color=color_GNN, edgecolor='black', linewidth=1)
axes[0, 0].bar(bar_positions_e_AB_M, [AB_M_feat_E_mae, AB_M_feat_E_rmse], width=bar_width, label='Feature-based', color=color_AB_M, edgecolor='black', linewidth=1)

# Set font size
axes[0, 0].tick_params(axis='both', which='major', labelsize=15)
axes[0, 0].legend(fontsize=15)
axes[0, 0].set_xlabel('Metrics', fontsize=15)
axes[0, 0].set_ylabel('Error', fontsize=15)
axes[0, 0].set_title('ESOL', fontsize=15)

# Set x-tick positions and labels
axes[0, 0].set_xticks(bar_positions_metrics + bar_width / 1)
axes[0, 0].set_xticklabels(['MAE', 'RMSE'])



# Subplot 2: AQUA
bar_positions_a_Edge = bar_positions_metrics 
bar_positions_a_GNN = bar_positions_metrics + bar_width
bar_positions_a_AB_M = bar_positions_metrics + bar_width + bar_width

# Plot MAE and RMSE bars next to each other with borders
axes[0, 1].bar(bar_positions_a_Edge, [Edge_A_mae, Edge_A_rmse], width=bar_width, label='EdgeConv', color=color_Edge, edgecolor='black', linewidth=1)

axes[0, 1].bar(bar_positions_a_GNN, [GNN_A_mae, GNN_A_rmse], width=bar_width, label='GCN', color=color_GNN, edgecolor='black', linewidth=1)
axes[0, 1].bar(bar_positions_a_AB_M, [AB_M_feat_A_mae, AB_M_feat_A_rmse], width=bar_width, label='Feature-based', color=color_AB_M, edgecolor='black', linewidth=1)

# Set font size
axes[0, 1].tick_params(axis='both', which='major', labelsize=15)
axes[0, 1].legend(fontsize=15)
axes[0, 1].set_xlabel('Metrics', fontsize=15)
axes[0, 1].set_ylabel('Error', fontsize=15)
axes[0, 1].set_title('AQUA', fontsize=15)

# Set x-tick positions and labels
axes[0, 1].set_xticks(bar_positions_metrics + bar_width / 1)
axes[0, 1].set_xticklabels(['MAE', 'RMSE'])

# Subplot 3: PHYS
bar_positions_p_GNN = bar_positions_metrics+ bar_width
bar_positions_p_Edge = bar_positions_metrics 
bar_positions_p_AB_M = bar_positions_metrics + bar_width + bar_width

# Plot MAE and RMSE bars next to each other with borders
axes[1, 0].bar(bar_positions_p_Edge, [Edge_P_mae, Edge_P_rmse], width=bar_width, label='EdgeConv', color=color_Edge, edgecolor='black', linewidth=1)
axes[1, 0].bar(bar_positions_p_GNN, [GNN_P_mae, GNN_P_rmse], width=bar_width, label='GCN', color=color_GNN, edgecolor='black', linewidth=1)
axes[1, 0].bar(bar_positions_p_AB_M, [AB_M_feat_P_mae, AB_M_feat_P_rmse], width=bar_width, label='Feature-based', color=color_AB_M, edgecolor='black', linewidth=1)

# Set font size
axes[1, 0].tick_params(axis='both', which='major', labelsize=15)
axes[1, 0].legend(fontsize=15)
axes[1, 0].set_xlabel('Metrics', fontsize=15)
axes[1, 0].set_ylabel('Error', fontsize=15)
axes[1, 0].set_title('PHYS', fontsize=15)

# Set x-tick positions and labels
axes[1, 0].set_xticks(bar_positions_metrics + bar_width / 1)
axes[1, 0].set_xticklabels(['MAE', 'RMSE'])

# Subplot 4: ALL-Data (EOAP)
bar_positions_all_GNN = bar_positions_metrics+ bar_width
bar_positions_all_Edge = bar_positions_metrics 
bar_positions_all_AB_M = bar_positions_metrics + bar_width + bar_width

# Plot MAE and RMSE bars next to each other with borders
axes[1, 1].bar(bar_positions_all_Edge, [Edge_EOAP_mae, Edge_EOAP_rmse], width=bar_width, label='EdgeConv', color=color_Edge, edgecolor='black', linewidth=1)
axes[1, 1].bar(bar_positions_all_GNN, [GNN_EOAP_mae, GNN_EOAP_rmse], width=bar_width, label='GCN', color=color_GNN, edgecolor='black', linewidth=1)
axes[1, 1].bar(bar_positions_all_AB_M, [AB_M_feat_EOAP_mae, AB_M_feat_EOAP_rmse], width=bar_width, label='Feature-based', color=color_AB_M, edgecolor='black', linewidth=1)

# Set font size
axes[1, 1].tick_params(axis='both', which='major', labelsize=15)
axes[1, 1].legend(fontsize=15)
axes[1, 1].set_xlabel('Metrics', fontsize=15)
axes[1, 1].set_ylabel('Error', fontsize=15)
axes[1, 1].set_title('ALL-Data (EOAP)', fontsize=15)

# Set x-tick positions and labels
axes[1, 1].set_xticks(bar_positions_metrics + bar_width / 1)
axes[1, 1].set_xticklabels(['MAE', 'RMSE'])
# Enable interactive mode
plt.ion()

# Show the plot

file_path = os.path.join(r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Plot results', 'bar_chart.png')

plt.savefig(file_path)

# Show the plot
plt.show()



# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 10), dpi = 300)

# Subplot 1: ESOL on AB_M_feat_
axes[0, 0].scatter(AB_M_feat_true_labels_E, AB_M_feat_predictions_E, alpha=0.5)
axes[0, 0].plot([min(AB_M_feat_true_labels_E), max(AB_M_feat_true_labels_E)], [min(AB_M_feat_true_labels_E), max(AB_M_feat_true_labels_E)], '-', alpha=0.5, color='blue')
axes[0, 0].plot([min(AB_M_feat_true_labels_E), max(AB_M_feat_true_labels_E)], [min(AB_M_feat_true_labels_E) + AB_M_feat_E_rmse, max(AB_M_feat_true_labels_E) + AB_M_feat_E_rmse], '--', alpha=0.5, color='green')
axes[0, 0].plot([min(AB_M_feat_true_labels_E), max(AB_M_feat_true_labels_E)], [min(AB_M_feat_true_labels_E) - AB_M_feat_E_rmse, max(AB_M_feat_true_labels_E) - AB_M_feat_E_rmse], '--', alpha=0.5, color='green')
axes[0, 0].text(0.05, 0.95, f'R² = {AB_M_feat_E_r_squared:.4f}', transform=axes[0, 0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 0].set_xlabel('True Values', fontsize=15)
axes[0, 0].set_ylabel('Predictions', fontsize=15)
axes[0, 0].set_title('ESOL', fontsize=16)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)

# Subplot 2: AQUA on AB_M_feat_
axes[0, 1].scatter(AB_M_feat_true_labels_A, AB_M_feat_predictions_A, alpha=0.5)
axes[0, 1].plot([min(AB_M_feat_true_labels_A), max(AB_M_feat_true_labels_A)], [min(AB_M_feat_true_labels_A), max(AB_M_feat_true_labels_A)], '-', alpha=0.5, color='blue')
axes[0, 1].plot([min(AB_M_feat_true_labels_A), max(AB_M_feat_true_labels_A)], [min(AB_M_feat_true_labels_A) + AB_M_feat_A_rmse, max(AB_M_feat_true_labels_A) + AB_M_feat_A_rmse], '--', alpha=0.5, color='green')
axes[0, 1].plot([min(AB_M_feat_true_labels_A), max(AB_M_feat_true_labels_A)], [min(AB_M_feat_true_labels_A) - AB_M_feat_A_rmse, max(AB_M_feat_true_labels_A) - AB_M_feat_A_rmse], '--', alpha=0.5, color='green')
axes[0, 1].text(0.05, 0.95, f'R² = {AB_M_feat_A_r_squared:.4f}', transform=axes[0, 1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 1].set_xlabel('True Values', fontsize=15)
axes[0, 1].set_ylabel('Predictions', fontsize=15)
axes[0, 1].set_title('AQUA', fontsize=16)
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)

# Subplot 3: PHYS on AB_M_feat_
axes[0, 2].scatter(AB_M_feat_true_labels_P, AB_M_feat_predictions_P, alpha=0.5)
axes[0, 2].plot([min(AB_M_feat_true_labels_P), max(AB_M_feat_true_labels_P)], [min(AB_M_feat_true_labels_P), max(AB_M_feat_true_labels_P)], '-', alpha=0.5, color='blue')
axes[0, 2].plot([min(AB_M_feat_true_labels_P), max(AB_M_feat_true_labels_P)], [min(AB_M_feat_true_labels_P) + AB_M_feat_P_rmse, max(AB_M_feat_true_labels_P) + AB_M_feat_P_rmse], '--', alpha=0.5, color='green')
axes[0, 2].plot([min(AB_M_feat_true_labels_P), max(AB_M_feat_true_labels_P)], [min(AB_M_feat_true_labels_P) - AB_M_feat_P_rmse, max(AB_M_feat_true_labels_P) - AB_M_feat_P_rmse], '--', alpha=0.5, color='green')
axes[0, 2].text(0.05, 0.95, f'R² = {AB_M_feat_P_r_squared:.4f}', transform=axes[0, 2].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 2].set_xlabel('True Values', fontsize=15)
axes[0, 2].set_ylabel('Predictions', fontsize=15)
axes[0, 2].set_title('PHYS', fontsize=16)
axes[0, 2].tick_params(axis='both', which='major', labelsize=12)

# Subplot 4: ALL-Data (EOAP) on AB_M_feat_
axes[0, 3].scatter(AB_M_feat_true_labels_EOAP, AB_M_feat_predictions_EOAP, alpha=0.5)
axes[0, 3].plot([min(AB_M_feat_true_labels_EOAP), max(AB_M_feat_true_labels_EOAP)], [min(AB_M_feat_true_labels_EOAP), max(AB_M_feat_true_labels_EOAP)], '-', alpha=0.5, color='blue')
axes[0, 3].plot([min(AB_M_feat_true_labels_EOAP), max(AB_M_feat_true_labels_EOAP)], [min(AB_M_feat_true_labels_EOAP) + AB_M_feat_EOAP_rmse, max(AB_M_feat_true_labels_EOAP) + AB_M_feat_EOAP_rmse], '--', alpha=0.5, color='green')
axes[0, 3].plot([min(AB_M_feat_true_labels_EOAP), max(AB_M_feat_true_labels_EOAP)], [min(AB_M_feat_true_labels_EOAP) - AB_M_feat_EOAP_rmse, max(AB_M_feat_true_labels_EOAP) - AB_M_feat_EOAP_rmse], '--', alpha=0.5, color='green')
axes[0, 3].text(0.05, 0.95, f'R² = {AB_M_feat_EOAP_r_squared:.4f}', transform=axes[0, 3].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 3].set_xlabel('True Values', fontsize=15)
axes[0, 3].set_ylabel('Predictions', fontsize=15)
axes[0, 3].set_title('All Data', fontsize=16)
axes[0, 3].tick_params(axis='both', which='major', labelsize=12)

# Subplot 1: PHYS on AB_M_feat_ with histogram of errors and KDE curve using Seaborn
sns.histplot(AB_M_feat_errors_E, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 0])
axes[1, 0].set_xlabel('Errors', fontsize=14)
axes[1, 0].set_ylabel('Density', fontsize=14)
axes[1, 0].set_title('', fontsize=16)
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
axes[1, 0].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 0].set_ylim(0, 1.00)  # Set y-axis limits

# Subplot 2: AQUA on AB_M_feat_ with histogram of errors and KDE curve using Seaborn
sns.histplot(AB_M_feat_errors_P, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 1])
axes[1, 1].set_xlabel('Errors', fontsize=14)
axes[1, 1].set_ylabel('Density', fontsize=14)
axes[1, 1].set_title('', fontsize=16)
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
axes[1, 1].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 1].set_ylim(0, 1.00)  # Set y-axis limits

# Subplot 5: AQUA on AB_M_feat_ with histogram of errors and KDE curve using Seaborn
sns.histplot(AB_M_feat_errors_A, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 2])
axes[1, 2].set_xlabel('Errors', fontsize=14)
axes[1, 2].set_ylabel('Density', fontsize=14)
axes[1, 2].set_title('', fontsize=16)
axes[1, 2].tick_params(axis='both', which='major', labelsize=12)
axes[1, 2].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 2].set_ylim(0, 1.00)  # Set y-axis limits

# Subplot 8: ALL Data (EOAP) on AB_M_feat_ with histogram of errors and KDE curve using Seaborn
sns.histplot(AB_M_feat_errors_EOAP, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 3])
axes[1, 3].set_xlabel('Errors', fontsize=14)
axes[1, 3].set_ylabel('Density', fontsize=14)
axes[1, 3].set_title('', fontsize=16)
axes[1, 3].tick_params(axis='both', which='major', labelsize=12)
axes[1, 3].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 3].set_ylim(0, 1.0)  # Set y-axis limits

file_path_3 = os.path.join(r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Plot results', 'error_dist_3.png')
plt.savefig(file_path_3)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 10), dpi = 300)
#fig.suptitle('Comparison of MAE and RMSE for GNN, Edge, and Feature-based (ESOL, AQUA, PHYS, ALL-Data)', fontsize=16)

# Subplot 1: ESOL on EdgeConv

# Define colors
color_GNN = rgb_to_hex((51, 90, 153))
color_Edge = rgb_to_hex((77, 148, 255))
color_AB_M = rgb_to_hex((200, 220, 255))

# Subplot 1: ESOL on EdgeConv
axes[0, 0].scatter(Edge_true_labels_E, Edge_predictions_E, alpha=0.5)
axes[0, 0].plot([min(Edge_true_labels_E), max(Edge_true_labels_E)], [min(Edge_true_labels_E), max(Edge_true_labels_E)], '-', alpha=0.5, color='blue')
axes[0, 0].plot([min(Edge_true_labels_E), max(Edge_true_labels_E)], [min(Edge_true_labels_E) + Edge_E_rmse, max(Edge_true_labels_E) + Edge_E_rmse], '--', alpha=0.5, color='green')
axes[0, 0].plot([min(Edge_true_labels_E), max(Edge_true_labels_E)], [min(Edge_true_labels_E) - Edge_E_rmse, max(Edge_true_labels_E) - Edge_E_rmse], '--', alpha=0.5, color='green')
axes[0, 0].text(0.05, 0.95, f'R² = {Edge_E_r_squared:.4f}', transform=axes[0, 0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 0].set_xlabel('True Values', fontsize=15)
axes[0, 0].set_ylabel('Predictions', fontsize=15)
axes[0, 0].set_title('ESOL', fontsize=16)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)

# Subplot 2: AQUA on EdgeConv
axes[0, 1].scatter(Edge_true_labels_A, Edge_predictions_A, alpha=0.5)
axes[0, 1].plot([min(Edge_true_labels_A), max(Edge_true_labels_A)], [min(Edge_true_labels_A), max(Edge_true_labels_A)], '-', alpha=0.5, color='blue')
axes[0, 1].plot([min(Edge_true_labels_A), max(Edge_true_labels_A)], [min(Edge_true_labels_A) + Edge_A_rmse, max(Edge_true_labels_A) + Edge_A_rmse], '--', alpha=0.5, color='green')
axes[0, 1].plot([min(Edge_true_labels_A), max(Edge_true_labels_A)], [min(Edge_true_labels_A) - Edge_A_rmse, max(Edge_true_labels_A) - Edge_A_rmse], '--', alpha=0.5, color='green')
axes[0, 1].text(0.05, 0.95, f'R² = {Edge_A_r_squared:.4f}', transform=axes[0, 1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 1].set_xlabel('True Values', fontsize=15)
axes[0, 1].set_ylabel('Predictions', fontsize=15)
axes[0, 1].set_title('AQUA', fontsize=16)
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)

# Subplot 3: PHYS on EdgeConv
axes[0, 2].scatter(Edge_true_labels_P, Edge_predictions_P, alpha=0.5)
axes[0, 2].plot([min(Edge_true_labels_P), max(Edge_true_labels_P)], [min(Edge_true_labels_P), max(Edge_true_labels_P)], '-', alpha=0.5, color='blue')
axes[0, 2].plot([min(Edge_true_labels_P), max(Edge_true_labels_P)], [min(Edge_true_labels_P) + Edge_P_rmse, max(Edge_true_labels_P) + Edge_P_rmse], '--', alpha=0.5, color='green')
axes[0, 2].plot([min(Edge_true_labels_P), max(Edge_true_labels_P)], [min(Edge_true_labels_P) - Edge_P_rmse, max(Edge_true_labels_P) - Edge_P_rmse], '--', alpha=0.5, color='green')
axes[0, 2].text(0.05, 0.95, f'R² = {Edge_P_r_squared:.4f}', transform=axes[0, 2].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 2].set_xlabel('True Values', fontsize=15)
axes[0, 2].set_ylabel('Predictions', fontsize=15)
axes[0, 2].set_title('PHYS', fontsize=16)
axes[0, 2].tick_params(axis='both', which='major', labelsize=12)

# Subplot 4: ALL-Data (EOAP) on EdgeConv
axes[0, 3].scatter(Edge_true_labels_EOAP, Edge_predictions_EOAP, alpha=0.5)
axes[0, 3].plot([min(Edge_true_labels_EOAP), max(Edge_true_labels_EOAP)], [min(Edge_true_labels_EOAP), max(Edge_true_labels_EOAP)], '-', alpha=0.5, color='blue')
axes[0, 3].plot([min(Edge_true_labels_EOAP), max(Edge_true_labels_EOAP)], [min(Edge_true_labels_EOAP) + Edge_EOAP_rmse, max(Edge_true_labels_EOAP) + Edge_EOAP_rmse], '--', alpha=0.5, color='green')
axes[0, 3].plot([min(Edge_true_labels_EOAP), max(Edge_true_labels_EOAP)], [min(Edge_true_labels_EOAP) - Edge_EOAP_rmse, max(Edge_true_labels_EOAP) - Edge_EOAP_rmse], '--', alpha=0.5, color='green')
axes[0, 3].text(0.05, 0.95, f'R² = {Edge_EOAP_r_squared:.4f}', transform=axes[0, 3].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
axes[0, 3].set_xlabel('True Values', fontsize=15)
axes[0, 3].set_ylabel('Predictions', fontsize=15)
axes[0, 3].set_title('All Data', fontsize=16)
axes[0, 3].tick_params(axis='both', which='major', labelsize=12)


# Subplot 1: PHYS on EdgeConv with histogram of errors and KDE curve using Seaborn
sns.histplot(Edge_errors_E, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 0])
axes[1, 0].set_xlabel('Errors', fontsize=14)
axes[1, 0].set_ylabel('Density', fontsize=14)
axes[1, 0].set_title('', fontsize=16)
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
axes[1, 0].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 0].set_ylim(0, 0.9)  # Set x-axis limits

# Subplot 2: AQUA on EdgeConv with histogram of errors and KDE curve using Seaborn
sns.histplot(Edge_errors_P, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 1])
axes[1, 1].set_xlabel('Errors', fontsize=14)
axes[1, 1].set_ylabel('Density', fontsize=14)
axes[1, 1].set_title('', fontsize=16)
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
axes[1, 1].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 1].set_ylim(0, 0.9)  # Set x-axis limits

# Subplot 5: AQUA on GNN with histogram of errors and KDE curve using Seaborn
sns.histplot(GNN_errors_A, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 2])
axes[1, 2].set_xlabel('Errors', fontsize=14)
axes[1, 2].set_ylabel('Density', fontsize=14)
axes[1, 2].set_title('', fontsize=16)
axes[1, 2].tick_params(axis='both', which='major', labelsize=12)
axes[1, 2].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 2].set_ylim(0, 0.9)  # Set x-axis limits

# Subplot 8: ALL Data (EOAP) on Feature-based with histogram of errors and KDE curve using Seaborn
sns.histplot(AB_M_feat_errors_EOAP, bins=50, alpha=0.4, color='green', kde=True, stat='density', ax=axes[1, 3])
axes[1, 3].set_xlabel('Errors', fontsize=14)
axes[1, 3].set_ylabel('Density', fontsize=14)
axes[1, 3].set_title('', fontsize=16)
axes[1, 3].tick_params(axis='both', which='major', labelsize=12)
axes[1, 3].set_xlim(-4, 4)  # Set x-axis limits
axes[1, 3].set_ylim(0, 0.9)  # Set x-axis limits

file_path_1 = os.path.join(r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Plot results', 'error_dist_1.png')
plt.savefig(file_path_1)
# Adjust layout and show the plot
plt.tight_layout()
plt.show()
