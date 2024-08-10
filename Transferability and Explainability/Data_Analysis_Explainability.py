# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:51:00 2024

@author: magha
"""

# %%
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from mordred import Calculator, descriptors
import deepchem as dc
import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse


from sklearn.model_selection import train_test_split
import shap
shap.initjs()

path_Numpy = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Revised_Processed_Saved_Data_solvent'
Labels_PHYS_NoI = pd.read_csv(f"{path_Numpy}/Labels_PHYS_NoI.csv")


""" Load Variables """
save_for_processing = r'C:\Users\magha\OneDrive - The University of Western Ontario\CODES\DL for Solubility Prediction\Manuscript_codes\Error_and_Chem_Analysis'
list_vars_names = ['combined_top_feature_names.pkl', 'EOAP_smiles.pkl']
numpy_vars_names = ['EOAP_X.npy', 'EOAP_Labels.npy',
                    'EOAP_Preds_Tabular.npy', 'Errors.npy']


#%%
list_vars = []
for name in list_vars_names:
    with open(f'{save_for_processing}\\{name}', 'rb') as file:
        list_vars.append(pickle.load(file))
combined_feature_names, EOAP_smiles = list_vars

numpy_vars = [np.load(f'{save_for_processing}\\{name}')
              for name in numpy_vars_names]
EOAP_X, EOAP_Labels, EOAP_Preds_Tabular, Errors = numpy_vars

model_path_save = f"{save_for_processing}/XGB_best_model_PHYS.joblib"
model = joblib.load(model_path_save)
EOAP_X_df = pd.DataFrame(EOAP_X, columns=combined_feature_names)


def count_functional_groups(smiles_list):
    # Define SMARTS patterns for hydrophilic functional groups
    hydrophilic_patterns = {
        'hydroxyl': Chem.MolFromSmarts('[OX2H]'),
        'amine': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
        'carboxylic acid': Chem.MolFromSmarts('C(=O)[OH]'),
        'sulfonic': Chem.MolFromSmarts('S(=O)(=O)[OH]'),
        'phosphate': Chem.MolFromSmarts('P(=O)(O)(O)O'),
        'amide': Chem.MolFromSmarts('C(=O)N'),
        'carbonyl': Chem.MolFromSmarts('C=O'),
        'ether': Chem.MolFromSmarts('COC'),
        'ester': Chem.MolFromSmarts('C(=O)O')
        
    }

    # Define SMARTS patterns for hydrophobic functional groups
    hydrophobic_patterns = {
        #'alkyl_chains': Chem.MolFromSmarts('[CX4]'),
        'aromatic': Chem.MolFromSmarts('c1ccccc1'),
        'halogen': Chem.MolFromSmarts('[F,Cl,Br,I]')
    }

    # Initialize results dictionary
    results = {name: [] for name in hydrophilic_patterns}
    results.update({name: [] for name in hydrophobic_patterns})
    results['hydrophilic_sum'] = []
    results['hydrophobic_sum'] = []

    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        hydrophilic_sum = 0
        hydrophobic_sum = 0

        if mol:
            # Count hydrophilic functional groups
            for name, pattern in hydrophilic_patterns.items():
                count = len(mol.GetSubstructMatches(pattern))
                results[name].append(count)
                hydrophilic_sum += count

            # Count hydrophobic functional groups
            for name, pattern in hydrophobic_patterns.items():
                count = len(mol.GetSubstructMatches(pattern))
                results[name].append(count)
                hydrophobic_sum += count
        else:
            for name in hydrophilic_patterns:
                results[name].append(0)
            for name in hydrophobic_patterns:
                results[name].append(0)

        results['hydrophilic_sum'].append(hydrophilic_sum)
        results['hydrophobic_sum'].append(hydrophobic_sum)

    return pd.DataFrame(results)

functional_groups_df = count_functional_groups(EOAP_smiles)
# functional_groups_df['logS'] = EOAP_Labels
functional_groups_df['SMILES'] = EOAP_smiles

# Convert Errors to DataFrame and categorize
errors_df = pd.DataFrame(Errors, columns=['error'])
quantiles = errors_df['error'].quantile([0.25, 0.75])
low_error_threshold = quantiles[0.25]
high_error_threshold = quantiles[0.75]
errors_df['error_category'] = pd.cut(errors_df['error'],
                                     bins=[-np.inf, low_error_threshold,
                                           high_error_threshold, np.inf],
                                     labels=['Low Error', 'Medium Error', 'High Error'])


# Convert Errors to DataFrame and categorize
logS_df = pd.DataFrame(EOAP_Labels, columns=['logS'])
quantiles = logS_df['logS'].quantile([0.25, 0.75])
low_S_threshold = quantiles[0.25]
high_S_threshold = quantiles[0.75]
logS_df['logS_category'] = pd.cut(logS_df['logS'],
                                  bins=[-np.inf, low_S_threshold,
                                        high_S_threshold, np.inf],
                                  labels=['Low logS', 'Medium logS', 'High logS'])
#%%
""" Explainability study """
explainer = shap.Explainer(model)
shap_values = explainer(EOAP_X_df)


EOAP_X_df = pd.DataFrame(EOAP_X, columns=combined_feature_names)
explainer = shap.Explainer(model)
shap_values = explainer(EOAP_X_df)
shap_df = pd.DataFrame(shap_values.values, columns=EOAP_X_df.columns)
mean_abs_shap_values = shap_df.abs().mean()
top_30_features = mean_abs_shap_values.sort_values(
    ascending=False).head(30).index


EOAP_key_features = EOAP_X_df[top_30_features]
EOAP_key_features['logS'] = logS_df['logS']
EOAP_key_features['logS_category'] = logS_df['logS_category']
# functional_groups_df, errors_df
# logS_df = logS_df['logS_category']
EOAP_key_features = pd.concat(
    [EOAP_X_df[top_30_features], functional_groups_df, logS_df, errors_df], axis=1)


Functional_Error_Feat_df = pd.concat(
    [EOAP_X_df[top_30_features], functional_groups_df, logS_df, errors_df], axis=1)



#%%
"""Top Features importance"""
shap.summary_plot(shap_values, EOAP_X_df, plot_type="bar")

#%%



""" Features correlation"""

correlation_matrix = EOAP_key_features[top_30_features[0:10]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Feature Correlation Matrix')
plt.show()


# Compute the sum of absolute correlation values for each feature with others (excluding itself)
sum_correlations = correlation_matrix.abs().sum() - 1
sum_correlations = sum_correlations.sort_values(ascending=False)

# Plot the summation of correlations
plt.figure(figsize=(12, 8))
sns.barplot(x=sum_correlations.index,
            y=sum_correlations.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Summation of Absolute Correlation Values for Each Feature')
plt.xlabel('Feature')
plt.ylabel('Sum of Absolute Correlation Values')
plt.show()


# %%

""" Functional Groups Features correlation"""

Functional_Error_df = pd.concat(
    [  logS_df, errors_df, functional_groups_df], axis=1)

correlation_matrix = functional_groups_df[:-1].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Feature Correlation Matrix')
plt.show()


# Compute the sum of absolute correlation values for each feature with others (excluding itself)
sum_correlations = correlation_matrix.abs().sum() - 1
sum_correlations = sum_correlations.sort_values(ascending=False)

# Plot the summation of correlations
plt.figure(figsize=(12, 8))
sns.barplot(x=sum_correlations.index,
            y=sum_correlations.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Summation of Absolute Correlation Values for Each Feature')
plt.xlabel('Feature')
plt.ylabel('Sum of Absolute Correlation Values')
plt.show()


""" Correlation with logS"""

# Concatenate logS with functional groups DataFrame
functional_groups_logS_df = pd.concat([logS_df['logS'], functional_groups_df], axis=1)

# Define columns to exclude
columns_to_exclude = ['hydrophilic_sum', 'hydrophobic_sum']

# Drop the specified columns from the DataFrame
filtered_df = functional_groups_logS_df.drop(columns=columns_to_exclude)

# Compute the correlation matrix
correlation_matrix = filtered_df.corr()

# Extract correlations with 'logS' and sort in descending order
logS_correlation = correlation_matrix['logS'].sort_values(ascending=False)[1:]

# Plot the correlations with 'logS'
# Plot the correlations with 'logS'
plt.figure(figsize=(12, 8))
sns.barplot(x=logS_correlation.index, y=logS_correlation.values, palette='viridis')
plt.xticks(rotation=45, fontsize=16)
plt.xlabel('No of Functional Groups', fontsize=16)
plt.ylabel('Correlation with logS', fontsize=16)
plt.title('Correlation of Functional Groups with logS', fontsize=16)
plt.show()
#%%
"================================="
""" functional groups distribution """

Functional_Error_df = pd.concat(
    [ functional_groups_df, logS_df, errors_df], axis=1)

Functional_Error_df.columns
functional_groups_df.columns
# List of functional group columns to check
functional_group_columns = Functional_Error_df.columns [:-7]


# Count the number of non-zero entries (1s) for each binary column
non_zero_counts = Functional_Error_df[functional_group_columns].sum()

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(non_zero_counts, labels=non_zero_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Distribution of Non-Zero Functional Groups')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


"================================="
total_count = non_zero_counts.sum()

threshold = 0.01 * total_count
large_groups = non_zero_counts[non_zero_counts >= threshold]
small_groups = non_zero_counts[non_zero_counts < threshold]

# Combine small groups into 'Others'
combined_counts = large_groups.copy()
combined_counts['Others'] = small_groups.sum()

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(combined_counts, labels=combined_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Distribution of Non-Zero Functional Groups')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


#%%
""" functional groups distribution """

Functional_Error_df = pd.concat(
    [ functional_groups_df, logS_df, errors_df], axis=1)
functional_groups_df.columns
Functional_Error_df.columns

# List of functional group columns to check
functional_group_columns = Functional_Error_df.columns [:-7]

# functional_group_columns = [
#     'hydroxyl_groups', 'amine_groups', 'carboxyl_groups', 'sulfonic_groups',
#     'phosphate_groups', 'amide_groups', 'carbonyl_groups', 'ether_groups',
#     'alkyl_chains', 'aromatic_rings', 'alkene_groups', 'alkyne_groups',
#     'halogen_groups'
# ]


# Create binary columns
for feature in functional_group_columns:
    Functional_Error_df[f'binary_{feature}'] = Functional_Error_df[feature].apply(lambda x: 1 if x != 0 else 0)

# List of binary columns
binary_columns = [f'binary_{feature}' for feature in functional_group_columns]

# Count the number of non-zero entries (1s) for each binary column
non_zero_counts = Functional_Error_df[binary_columns].sum()
len_dataset_molecules = len(Functional_Error_df)
non_zero_counts_with_total = non_zero_counts.copy()
non_zero_counts_with_total['total molecules'] = len_dataset_molecules
# bar plot

# Convert counts to a pandas Series
counts_series = pd.Series(non_zero_counts_with_total)

# Calculate proportions
proportions = (counts_series )

# Plot the bar plot
plt.figure(figsize=(12, 8))
plt.bar(proportions.index, proportions, color='skyblue')
plt.xlabel('Functional Groups')
plt.ylabel('Number of Molecules Containing Functional Groups')
plt.title('Distribution of Functional Groups in Molecules within Dataset')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(proportions) + 5)  # Adding some space at the top for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



renamed_proportions = proportions.rename(lambda x: x.replace('binary_', '') if x != 'total molecules' else x)
plt.figure(figsize=(12, 8))
plt.bar(renamed_proportions.index, renamed_proportions, color='skyblue')
# Add labels to each bar
for index, value in enumerate(proportions):
    plt.text(index, value + 50, f'{int(value)}', ha='center', va='bottom', fontsize=12)

# Add axis labels and title
plt.xlabel('Functional Groups')
plt.ylabel('Number of Molecules Containing Functional Groups')
plt.title('Distribution of Functional Groups in Molecules within Dataset')
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=14)

plt.ylim(0, len_dataset_molecules)  # Set max y to the total number of molecules

plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid lines for better readability

plt.tight_layout()  # Adjust layout to fit all labels
plt.show()
#%% 
"========== Bar plot and Venn circles plot ==========="

from matplotlib_venn import venn2, venn2_circles 
from matplotlib_venn import venn3, venn3_circles 
from itertools import combinations


Functional_Error_df.columns

pie_chart_columns = ['binary_hydroxyl', 'binary_carbonyl', 'binary_aromatic']

Pie_Df = Functional_Error_df [pie_chart_columns]
# Create new columns for each unique combination of binary columns
for i in range(1, len(pie_chart_columns) + 1):
    for combo in combinations(pie_chart_columns, i):
        col_name = '_'.join(combo)
        Pie_Df[col_name] = Pie_Df[list(combo)].all(axis=1).astype(int)

# Count occurrences of each combination
combination_counts = Pie_Df.iloc[:, len(pie_chart_columns):].sum()

# Filter out zero counts
combination_counts = combination_counts[combination_counts > 0]
combination_counts = pd.concat([non_zero_counts [pie_chart_columns], combination_counts])

from matplotlib_venn import venn3, venn3_circles 
venn_data = {
    '100': combination_counts[0],  # binary_hydroxyl
    '010': combination_counts[1],  # binary_carbonyl
    '001': combination_counts[2],  # binary_aromatic
    '110': combination_counts[3],   # binary_hydroxyl & binary_carbonyl
    '101': combination_counts[4],   # binary_hydroxyl & binary_aromatic
    '011': combination_counts[5],   # binary_carbonyl & binary_aromatic
    '111': combination_counts[6]    # binary_hydroxyl & binary_carbonyl & binary_aromatic
}
# Create the Venn diagram
plt.figure(figsize=(10, 8))
venn = venn3(subsets=venn_data, set_labels=('Hydroxyl', 'Carbonyl', 'Aromatic Ring'))
# Adjust font sizes for labels
for label in venn.set_labels:
    label.set_fontsize(14)
for label in venn.subset_labels:
    label.set_fontsize(14)
venn_circles = venn3_circles(subsets=venn_data, linestyle="dashed", linewidth=2)
plt.title("Venn Diagram of Key Functional Groups")
plt.show()





Functional_Error_df.columns
pie_chart_columns = ['binary_halogen', 'binary_ester', 'binary_aromatic']

Pie_Df = Functional_Error_df [pie_chart_columns]
# Create new columns for each unique combination of binary columns
for i in range(1, len(pie_chart_columns) + 1):
    for combo in combinations(pie_chart_columns, i):
        col_name = '_'.join(combo)
        Pie_Df[col_name] = Pie_Df[list(combo)].all(axis=1).astype(int)

# Count occurrences of each combination
combination_counts = Pie_Df.iloc[:, len(pie_chart_columns):].sum()

# Filter out zero counts
combination_counts = combination_counts[combination_counts > 0]
combination_counts = pd.concat([non_zero_counts [pie_chart_columns], combination_counts])

from matplotlib_venn import venn3, venn3_circles 
venn_data = {
    '100': combination_counts[0],  # binary_hydroxyl
    '010': combination_counts[1],  # binary_carbonyl
    '001': combination_counts[2],  # binary_aromatic
    '110': combination_counts[3],   # binary_hydroxyl & binary_carbonyl
    '101': combination_counts[4],   # binary_hydroxyl & binary_aromatic
    '011': combination_counts[5],   # binary_carbonyl & binary_aromatic
    '111': combination_counts[6]    # binary_hydroxyl & binary_carbonyl & binary_aromatic
}
# Create the Venn diagram
plt.figure(figsize=(10, 8))
venn = venn3(subsets=venn_data, set_labels=('Halogen', 'Ester', 'Aromatic Ring'))
# Adjust font sizes for labels
for label in venn.set_labels:
    label.set_fontsize(14)
for label in venn.subset_labels:
    label.set_fontsize(14)
venn_circles = venn3_circles(subsets=venn_data, linestyle="dashed", linewidth=2)
plt.title("Venn Diagram of Key Functional Groups")
plt.show()



"================================="

# Create binary columns for each functional group
for feature in functional_group_columns:
    Functional_Error_df[f'binary_{feature}'] = (Functional_Error_df[feature] > 0).astype(int)

# Count the number of non-zero entries across each binary column
non_zero_counts = Functional_Error_df[[f'binary_{feature}' for feature in functional_group_columns]].sum()

# Calculate the total count of non-zero entries
total_count = non_zero_counts.sum()
total_count = len(Functional_Error_df)

# Calculate the proportion of each functional group
proportions = non_zero_counts / total_count

# Prepare for pie chart
# Set a threshold for including 'Others' if necessary
threshold = 0.02  # Less than 1% of total
large_groups = proportions[proportions >= threshold]
small_groups = proportions[proportions < threshold]

# Combine small groups into 'Others'
combined_proportions = large_groups.copy()
combined_proportions['Others'] = small_groups.sum()

# Plotting
plt.figure(figsize=(10, 8))
plt.pie(combined_proportions, labels=combined_proportions.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Distribution of Functional Groups Across Columns')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

"======================="

"==================="





# Count the number of non-zero entries for each functional group
non_zero_counts = Functional_Error_df[functional_group_columns].astype(bool).sum()
types_sum = np.sum(non_zero_counts)
molecules_sum = len(functional_groups_df)

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(non_zero_counts, labels=non_zero_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Distribution of Non-Zero Functional Groups')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()





#%%
""" Significant functional groups having Correlation with logS"""

# Filter features with absolute correlation values larger than 0.15
significant_features = logS_correlation[logS_correlation.abs() > 0.15]
# Plot the significant correlations with logS
plt.figure(figsize=(10, 6))
sns.barplot(x=significant_features.index, y=significant_features.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Features with Significant Correlation with logS')
plt.xlabel('Feature')
plt.ylabel('Correlation with logS')
plt.show()

# Print the names of significant features
print("Features with absolute correlation values larger than 0.15 with logS:")
print(significant_features.index.tolist())

significant_functional_name = significant_features.index.tolist()

threshold_sig_fucn = np.floor(np.mean(functional_groups_df[significant_functional_name]))
functional_groups_df ['logS'] = Functional_Error_df['logS']


sum_type_functional_name = ['hydrophilic_sum', 'hydrophobic_sum']
significant_functional_name = [name for name in significant_features.index if name not in sum_type_functional_name]

for i, feature in enumerate( significant_functional_name):
    functional_groups_df[f'binary_{feature}'] = np.where(functional_groups_df[feature] > threshold_sig_fucn[i], 'Present', 'Absent')

for i, feature in enumerate( sum_type_functional_name):
    functional_groups_df[f'binary_{feature}'] = np.where(functional_groups_df[feature] > threshold_sig_fucn[i], 'High', 'Low')

functional_groups_df.columns

""" Plot violin logS for functional groups """
for feature in significant_functional_name:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=f'binary_{feature}', y='logS', data=functional_groups_df, hue = f'binary_{feature}', inner = 'quartile', split = True)
    plt.title(f'Distribution of {feature} by logS Category')
    plt.show()


functional_groups_df.columns

functional_groups_df.columns

# Filter the DataFrame based on 'binary_hydrophobic_sum'
filtered_df = functional_groups_df[functional_groups_df['binary_hydrophobic_sum'].isin(['High', 'Low'])]

# Plot violin plot with quartiles and split
plt.figure(figsize=(12, 8))
sns.violinplot(x='binary_hydrophilic_sum', y='logS', hue='binary_hydrophobic_sum', data=filtered_df, inner='quartile', split=True, palette='viridis')
plt.title('Distribution of logS by Hydrophilic and Hydrophobic Sum Categories')
plt.xlabel('Hydrophilic Sum Category')
plt.ylabel('logS')
plt.show()



""" logS distribution for halogen containing """
# Filter the DataFrame based on 'binary_hydrophobic_sum'
filtered_df = functional_groups_df[functional_groups_df['binary_halogen'].isin(['Present', 'Absent'])]

plt.figure(figsize=(12, 8))
sns.violinplot(x='binary_aromatic', y='logS', hue='binary_halogen', data=filtered_df, inner='quartile', split=True, palette='viridis')
plt.title('Distribution of logS by halogen and aromatic_rings Categories')
plt.xlabel('Hydrophilic Sum Category')
plt.ylabel('logS')
plt.show()



""" logS distribution for halogen containing """
# Filter the DataFrame based on 'binary_hydrophobic_sum'
filtered_df = functional_groups_df[functional_groups_df['binary_halogen'].isin(['Present', 'Absent'])]

plt.figure(figsize=(12, 8))
sns.violinplot(x='binary_hydrophilic_sum', y='logS', hue='binary_halogen', data=filtered_df, inner='quartile', split=True, palette='viridis')
plt.title('Distribution of logS by halogen and hydrophilic Categories')
plt.xlabel('hydrophilic Sum Category')
plt.ylabel('logS')
plt.show()



""" logS distribution for aromatic containing and hydrophobic_sum """
# Filter the DataFrame based on 'binary_hydrophobic_sum'
filtered_df = functional_groups_df[functional_groups_df['binary_aromatic'].isin(['Present', 'Absent'])]

plt.figure(figsize=(12, 8))
sns.violinplot(x='binary_hydrophobic_sum', y='logS', hue='binary_aromatic', data=filtered_df, inner='quartile', split=True, palette='viridis')
plt.title('Distribution of logS by binary_aromatic_rings and binary_hydrophobic_sum Categories')
plt.xlabel('hydrophobic Sum Category')
plt.ylabel('logS')
plt.show()

#%%

""" significant functional groups distribution """


# Define the features to plot
features_to_plot = ['hydrophilic_sum', 'hydrophobic_sum']

# Set up the figure and axis
plt.figure(figsize=(14, 6))

# Create subplots
for idx, feature in enumerate(features_to_plot, 1):
    plt.subplot(1, 2, idx)  # (rows, columns, panel number)
    sns.violinplot(x='logS_category', y=feature, data=Functional_Error_df, inner=None, palette='viridis')
    plt.title(f'Distribution of {feature} by logS Category')
    plt.xlabel('logS Category')
    plt.ylabel('Value')

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()





""" significant functional groups distribution with logS category """

for feature in significant_functional_name:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='logS_category', y=feature, data=Functional_Error_df, inner = 'quartile')
    plt.title(f'Distribution of {feature} by logS Category')
    plt.show()

""" significant functional groups distribution with error category"""

for feature in significant_functional_name:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='error_category', y=feature, data=Functional_Error_df, inner = 'quartile')
    plt.title(f'Distribution of {feature} by Error Category')
    plt.show()


# List of features to plot
features_to_plot = ['hydrophilic_sum', 'hydrophobic_sum']

# Plotting distribution of features by logS category with quartiles
plt.figure(figsize=(14, 6))

for idx, feature in enumerate(features_to_plot, 1):
    plt.subplot(1, 2, idx)  # (rows, columns, panel number)
    sns.violinplot(x='logS_category', y=feature, data=Functional_Error_df, inner='quartile', palette='viridis')
    plt.title(f'Distribution of {feature} by logS Category')
    plt.xlabel('logS Category')
    plt.ylabel('Value')

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

# Plotting distribution of features by error category with quartiles
plt.figure(figsize=(14, 6))

for idx, feature in enumerate(features_to_plot, 1):
    plt.subplot(1, 2, idx)  # (rows, columns, panel number)
    sns.violinplot(x='error_category', y=feature, data=Functional_Error_df, inner='quartile', palette='viridis')
    plt.title(f'Distribution of {feature} by Error Category')
    plt.xlabel('Error Category')
    plt.ylabel('Value')

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

#%%

""" Feature similarity vs LogS and their accordance"""

top_30_features = EOAP_key_features.columns[:30]
# Standardize features
scaler = StandardScaler()
EOAP_scaled = scaler.fit_transform(EOAP_key_features[top_30_features])

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
EOAP_pca = pca.fit_transform(EOAP_scaled)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(EOAP_pca)
EOAP_key_features['cluster'] = clusters

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=EOAP_pca, columns=['PC1', 'PC2'])
pca_df['logS'] = EOAP_key_features['logS']
pca_df['cluster'] = clusters

# Plot PCA clusters with logS as the hue
plt.figure(figsize=(14, 10))
scatter = sns.scatterplot(x='PC1', y='PC2', hue='logS',
                          palette='coolwarm', data=pca_df, s=100)


# # Plot decision boundaries
# x_min, x_max = pca_df['PC1'].min() - 1, pca_df['PC1'].max() + 1
# y_min, y_max = pca_df['PC2'].min() - 1, pca_df['PC2'].max() + 1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot decision boundaries with contour lines
# plt.contour(xx, yy, Z, levels=np.arange(-0.5, 3, 1), cmap='tab10', linewidths=1)
# Add color bar

norm = plt.Normalize(pca_df['logS'].min(), pca_df['logS'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])
plt.colorbar(sm)

plt.title('Clusters of Molecular Structures Based on Top Features with logS')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

#%%
""" Feature similarity-based clustering """
cluster_smiles_list = []

pca_df['SMILES'] = EOAP_key_features['SMILES']
for cluster_num in range(10):
    cluster_data = pca_df[pca_df['cluster'] == cluster_num]
    for _, row in cluster_data.iterrows():
        cluster_smiles_list.append({
            'Cluster': cluster_num,
            'SMILES': row['SMILES'],
            'logS': row['logS']
        })

# Convert list to DataFrame using pd.concat
cluster_smiles_df = pd.DataFrame(cluster_smiles_list)

# %%
""" Plot the relationship between features and logS categories """

for feature in top_30_features[:10]:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='logS_category', y=feature, data=EOAP_key_features, inner = 'quartile')
    plt.title(f'Distribution of {feature} by logS Category')
    plt.show()




# %%
""" Plot the relationship between features and error categories """

for feature in top_30_features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='error_category', y=feature, data=EOAP_key_features, inner = 'quartile')
    plt.title(f'Distribution of {feature} by Error Category')
    plt.show()


# Standardize features
scaler = StandardScaler()
EOAP_scaled = scaler.fit_transform(EOAP_key_features[top_30_features])

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
EOAP_pca = pca.fit_transform(EOAP_scaled)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(EOAP_pca)
EOAP_key_features['cluster'] = clusters

# Plot PCA clusters
plt.figure(figsize=(14, 10))
sns.scatterplot(x=EOAP_pca[:, 0], y=EOAP_pca[:, 1], hue='cluster',
                palette='tab10', data=EOAP_key_features, s=100)
plt.title('Clusters of Molecular Structures Based on Top Features')
plt.show()

#%%
""" ٍError analysis for functional groups  """

Functional_Error_df = pd.concat(
    [ functional_groups_df, logS_df, errors_df], axis=1)
functional_groups_df.columns
Functional_Error_df.columns

# List of functional group columns to check
functional_group_columns = Functional_Error_df.columns [:-7]

# Create binary columns
for feature in functional_group_columns:
    Functional_Error_df[f'binary_{feature}'] = Functional_Error_df[feature].apply(lambda x: 1 if x != 0 else 0)

# List of binary columns
binary_columns = [f'binary_{feature}' for feature in functional_group_columns]

# Calculate mean error for each functional group
mean_errors = Functional_Error_df.groupby('error_category').mean().reset_index()

# Select columns related to functional groups and error
functional_groups = [
    'hydroxyl', 'amine', 'carboxylic acid', 'sulfonic', 'phosphate',
    'amide', 'carbonyl', 'ether', 'ester', 'aromatic', 'halogen'
]

# Calculate the mean error for each functional group
mean_errors = {}
for group in functional_groups:
    mean_errors[group] = Functional_Error_df[Functional_Error_df[f'binary_{group}'] == 1]['error'].mean()

mean_errors_df = pd.DataFrame.from_dict(mean_errors, orient='index', columns=['mean_error'])

# Identify high and low error groups (e.g., top and bottom 3)
high_error_groups = mean_errors_df.nlargest(3, 'mean_error')
low_error_groups = mean_errors_df.nsmallest(3, 'mean_error')

# Plot mean errors for each functional group
plt.figure(figsize=(14, 8))
sns.barplot(x=mean_errors_df.index, y='mean_error', data=mean_errors_df, palette='viridis')
plt.xlabel('Functional Groups')
plt.ylabel('Mean Error')
plt.title('Mean Error for Each Functional Group')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, mean_errors_df['mean_error'].max() + 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
""" ======   fucntional groups patterns for solubility   ======="""

functional_groups_df = count_functional_groups(EOAP_smiles)
functional_groups_df['SMILES'] = EOAP_smiles

Functional_Error_df = pd.concat(
    [ functional_groups_df, logS_df, errors_df], axis=1)
functional_groups_df.columns
Functional_Error_df.columns
functional_group_columns = Functional_Error_df.columns [:-7]

# Create binary columns
for feature in functional_group_columns:
    Functional_Error_df[f'binary_{feature}'] = Functional_Error_df[feature].apply(lambda x: 1 if x != 0 else 0)

binary_columns = [f'binary_{feature}' for feature in functional_group_columns]

functional_EOAP_Error_df = pd.concat([
    Functional_Error_df, EOAP_X_df], axis=1)
functional_EOAP_Error_df.columns


def select_functional(functional_EOAP_Error_df, functional_list_names):
    # Add 'binary_' prefix to each functional group name
    binary_columns = [f'{name}' for name in functional_list_names]
    
    # Select rows where any of the binary columns have non-zero values
    condition = functional_EOAP_Error_df[binary_columns].any(axis=1)
    selected_df = functional_EOAP_Error_df[condition]
    
    return selected_df


def select_functional_type2(functional_EOAP_Error_df, functional_list_names, non_selected):
    # Add 'binary_' prefix to each functional group name
    selected_columns = [f'{name}' for name in functional_list_names]
    non_selected_columns = [f'{name}' for name in non_selected]
    
    # Select rows where any of the selected columns have non-zero values
    selected_condition = functional_EOAP_Error_df[selected_columns].any(axis=1)
    
    # Ensure all non-selected columns are zero
    non_selected_condition = (functional_EOAP_Error_df[non_selected_columns] == 0).all(axis=1)
    
    # Combine conditions
    condition = selected_condition & non_selected_condition
    selected_df = functional_EOAP_Error_df[condition]
    
    return selected_df


""" plot hydrophobic elements in molecules"""
functional_list_names = ['halogen', 'aromatic']
selected_df = select_functional(functional_EOAP_Error_df, functional_list_names)

# Extract the data for plotting
x1 = selected_df[functional_list_names[0]]
x2 = selected_df[functional_list_names[1]]
logS_selected = selected_df['logS']

# Create a DataFrame for aggregation
df = pd.DataFrame({'x1': x1, 'x2': x2, 'logS': logS_selected})

# Group by x1 and x2 and calculate the mean of logS
df_reduced = df.groupby(['x1', 'x2'], as_index=False).agg({'logS': 'mean'})

# Extract reduced data for plotting
x1_reduced = df_reduced['x1']
x2_reduced = df_reduced['x2']
logS_reduced = df_reduced['logS']
logS_halogen_aromatic = logS_reduced

# Create a custom colormap from red to blue
cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', ['red', 'yellow'])

plt.figure(figsize=(12, 8))

# Create the scatter plot with average logS values
scatter = plt.scatter(x2_reduced, x1_reduced, c=logS_reduced, cmap=cmap, alpha=0.7, edgecolor='k', linewidth=1, s=100)

# Add colorbar for scatter plot with custom colormap
cb = plt.colorbar(scatter, label='Average logS')

# Set plot labels and title
plt.xlabel(functional_list_names[1])
plt.ylabel(functional_list_names[0])
plt.title('Scatter Plot of Functional Groups with Average logS Values')
plt.show()


""" plot hydrophilic functional groups effect with respect to hydrophobic elements in molecules"""

functional_list_names = ['hydroxyl']
selected_df2 = select_functional(functional_EOAP_Error_df, functional_list_names)

# Extract the data for plotting
x1 = selected_df2[functional_list_names[0]]
x2_1 = selected_df2['halogen']
x2_2 = selected_df2['aromatic']
x2 = x2_1 + x2_2


logS_selected = selected_df2['logS']

# Create a DataFrame for aggregation
df = pd.DataFrame({'x1': x1, 'x2': x2, 'logS': logS_selected})

# Group by x1 and x2 and calculate the mean of logS
df_reduced = df.groupby(['x1', 'x2'], as_index=False).agg({'logS': 'mean'})

# Extract reduced data for plotting
x1_reduced = df_reduced['x1']
x2_reduced = df_reduced['x2']
logS_reduced = df_reduced['logS']
logS_hydroxyl = logS_reduced

# Create a custom colormap from red to blue
cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', ['red', 'yellow'])

plt.figure(figsize=(12, 8))

# Create the scatter plot with average logS values
scatter = plt.scatter(x2_reduced, x1_reduced, c=logS_reduced, cmap=cmap, alpha=0.7, edgecolor='k', linewidth=1, s=100)

# Add colorbar for scatter plot with custom colormap
cb = plt.colorbar(scatter, label='Average logS')

# Set plot labels and title
plt.xlabel('number of (halogen + aromatic rings)')
plt.ylabel('number of hydroxyl')
plt.title('Scatter Plot of Functional Groups with Average logS Values')
plt.show()




""" plot hydrophilic functional groups effect with respect to hydrophobic elements in molecules"""

functional_list_names = ['carboxylic acid']
selected_df2 = select_functional(functional_EOAP_Error_df, functional_list_names)

# Extract the data for plotting
x1 = selected_df2[functional_list_names[0]]
x2_1 = selected_df2['halogen']
x2_2 = selected_df2['aromatic']
x2 = x2_1 + x2_2


logS_selected = selected_df2['logS']

# Create a DataFrame for aggregation
df = pd.DataFrame({'x1': x1, 'x2': x2, 'logS': logS_selected})

# Group by x1 and x2 and calculate the mean of logS
df_reduced = df.groupby(['x1', 'x2'], as_index=False).agg({'logS': 'mean'})

# Extract reduced data for plotting
x1_reduced = df_reduced['x1']
x2_reduced = df_reduced['x2']
logS_reduced = df_reduced['logS']
logS_carboxylic = logS_reduced

# Create a custom colormap from red to blue
cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', ['red', 'yellow'])

plt.figure(figsize=(12, 8))

# Create the scatter plot with average logS values
scatter = plt.scatter(x2_reduced, x1_reduced, c=logS_reduced, cmap=cmap, alpha=0.7, edgecolor='k', linewidth=1, s=100)

# Add colorbar for scatter plot with custom colormap
cb = plt.colorbar(scatter, label='Average logS')

# Set plot labels and title
plt.xlabel('number of (halogen + aromatic rings)')
plt.ylabel('number of carboxylic acid')
plt.title('Scatter Plot of Functional Groups with Average logS Values')
plt.show()


""" plot hydrophilic functional groups effect with respect to hydrophobic elements in molecules"""

functional_list_names = ['sulfonic']
selected_df2 = select_functional(functional_EOAP_Error_df, functional_list_names)

# Extract the data for plotting
x1 = selected_df2[functional_list_names[0]]
x2_1 = selected_df2['halogen']
x2_2 = selected_df2['aromatic']
x2 = x2_1 + x2_2


logS_selected = selected_df2['logS']

# Create a DataFrame for aggregation
df = pd.DataFrame({'x1': x1, 'x2': x2, 'logS': logS_selected})

# Group by x1 and x2 and calculate the mean of logS
df_reduced = df.groupby(['x1', 'x2'], as_index=False).agg({'logS': 'mean'})

# Extract reduced data for plotting
x1_reduced = df_reduced['x1']
x2_reduced = df_reduced['x2']
logS_reduced = df_reduced['logS']
logS_sulfonic = logS_reduced
# Create a custom colormap from red to blue
cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', ['red', 'yellow'])

plt.figure(figsize=(12, 8))

# Create the scatter plot with average logS values
scatter = plt.scatter(x2_reduced, x1_reduced, c=logS_reduced, cmap=cmap, alpha=0.7, edgecolor='k', linewidth=1, s=100)

# Add colorbar for scatter plot with custom colormap
cb = plt.colorbar(scatter, label='Average logS')

# Set plot labels and title
plt.xlabel('number of (halogen + aromatic rings)')
plt.ylabel('number of sulfonic acid')
plt.title('Scatter Plot of Functional Groups with Average logS Values')
plt.show()



""" violin plots of corresponding logS"""

# Create a DataFrame for the violin plot
data = pd.DataFrame({
    'logS_sulfonic': logS_sulfonic,
    'logS_carboxylic': logS_carboxylic,
    'logS_hydroxyl': logS_hydroxyl,
    'logS_halogen_aromatic': logS_halogen_aromatic
})

# Melt the DataFrame to long-form
data_melted = data.melt(var_name='Functional Group', value_name='logS')

# Create the violin plot
plt.figure(figsize=(14, 8))
sns.violinplot(x='Functional Group', y='logS', data=data_melted, inner='quartile', palette='viridis')

# Set plot labels and title
plt.xlabel('Functional Group')
plt.ylabel('logS')
plt.title('Violin Plots of logS Values for Different Functional Groups')

# Show the plot
plt.grid(True)
plt.show()


""" ======   Explainability for Case Studies   ======="""

non_selected = [
     'amine', 'carboxylic acid', 'sulfonic', 'phosphate',
    'amide', 'carbonyl', 'ether', 'ester', 'halogen']

functional_list_names = ['halogen']

functional_list_names = ['sulfonic']
selected_df_sulfonic = select_functional(functional_EOAP_Error_df, functional_list_names)
functional_list_names = ['halogen', 'aromatic']
functional_list_names = ['aromatic', 'hydroxyl']

selected_df_halogen = select_functional_type2(functional_EOAP_Error_df, functional_list_names, non_selected)


""" ======   Explainability for Case Studies   ======="""

functional_groups_df = count_functional_groups(EOAP_smiles)
functional_groups_df['SMILES'] = EOAP_smiles

Functional_Error_df = pd.concat(
    [ functional_groups_df, logS_df, errors_df], axis=1)
functional_groups_df.columns
Functional_Error_df.columns
functional_group_columns = Functional_Error_df.columns [:-7]

# Create binary columns
for feature in functional_group_columns:
    Functional_Error_df[f'binary_{feature}'] = Functional_Error_df[feature].apply(lambda x: 1 if x != 0 else 0)

binary_columns = [f'binary_{feature}' for feature in functional_group_columns]

functional_EOAP_Error_df = pd.concat([
    Functional_Error_df, EOAP_X_df], axis=1)
functional_EOAP_Error_df.columns


a1_1_smiles = functional_EOAP_Error_df.iloc[597]['SMILES']
a1_2_smiles = functional_EOAP_Error_df.iloc[600]['SMILES']
a1_1_X = EOAP_X_df.iloc[597]
a1_2_X = EOAP_X_df.iloc[600]


a2_1_smiles = functional_EOAP_Error_df.iloc[613]['SMILES']
a2_2_smiles = functional_EOAP_Error_df.iloc[543]['SMILES']
a2_1_X = EOAP_X_df.iloc[613]
a2_2_X = EOAP_X_df.iloc[543]

a3_1_smiles = functional_EOAP_Error_df.iloc[231]['SMILES']
a3_2_smiles = functional_EOAP_Error_df.iloc[234]['SMILES']
a3_1_X = EOAP_X_df.iloc[231]
a3_2_X = EOAP_X_df.iloc[234]

a4_1_smiles = functional_EOAP_Error_df.iloc[577]['SMILES']
a4_2_smiles = functional_EOAP_Error_df.iloc[579]['SMILES']
a4_1_X = EOAP_X_df.iloc[577]
a4_2_X = EOAP_X_df.iloc[579]


a5_1_smiles = functional_EOAP_Error_df.iloc[508]['SMILES']
a5_2_smiles = functional_EOAP_Error_df.iloc[513]['SMILES']
a5_1_X = EOAP_X_df.iloc[508]
a5_2_X = EOAP_X_df.iloc[513]

a6_1_smiles = functional_EOAP_Error_df.iloc[1119]['SMILES']
a6_2_smiles = functional_EOAP_Error_df.iloc[1120]['SMILES']
a6_1_X = EOAP_X_df.iloc[1119]
a6_2_X = EOAP_X_df.iloc[1120]

a7_1_smiles = functional_EOAP_Error_df.iloc[1121]['SMILES']
a7_2_smiles = functional_EOAP_Error_df.iloc[1122]['SMILES']
a7_1_X = EOAP_X_df.iloc[1121]
a7_2_X = EOAP_X_df.iloc[1122]

a8_1_smiles = functional_EOAP_Error_df.iloc[1148]['SMILES']
a8_2_smiles = functional_EOAP_Error_df.iloc[1149]['SMILES']
a8_1_X = EOAP_X_df.iloc[1148]
a8_2_X = EOAP_X_df.iloc[1149]


a9_1_smiles = functional_EOAP_Error_df.iloc[1155]['SMILES']
a9_2_smiles = functional_EOAP_Error_df.iloc[1158]['SMILES']
a9_1_X = EOAP_X_df.iloc[1155]
a9_2_X = EOAP_X_df.iloc[1158]


# Create a list of tuples for easy iteration
smiles_pairs = [
    (a1_1_smiles, a1_1_X, 597),
    (a1_2_smiles, a1_2_X, 600),
    (a2_1_smiles, a2_1_X, 613),
    (a2_2_smiles, a2_2_X, 543),
    (a3_1_smiles, a3_1_X, 231),
    (a3_2_smiles, a3_2_X, 234),
    (a4_1_smiles, a4_1_X, 577),
    (a4_2_smiles, a4_2_X, 579),
    (a5_1_smiles, a5_1_X, 508),
    (a5_2_smiles, a5_2_X, 513),
    (a6_1_smiles, a6_1_X, 1119),
    (a6_2_smiles, a6_2_X, 1120),
    (a7_1_smiles, a7_1_X, 1121),
    (a7_2_smiles, a7_2_X, 1122),
    (a8_1_smiles, a8_1_X, 1148),
    (a8_2_smiles, a8_2_X, 1149),
    (a9_1_smiles, a9_1_X, 1155),
    (a9_2_smiles, a9_2_X, 1158),
]

def plot_molecule_with_shap(smiles, X_index):
    # Generate molecule image using RDKit
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    plt.figure(figsize=(10, 12))

    # Plot molecule
    # plt.subplot(9, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'SMILES: {smiles}\nSolubility: {functional_EOAP_Error_df.iloc[X_index]["logS"]}')
    
    # Plot SHAP waterfall plot
    # plt.subplot(2, 1, 2)
    plt.figure(figsize=(10, 12))
    shap.plots.waterfall(shap_values[X_index], max_display=20)
    
    plt.tight_layout()
    plt.show()
    print("""=================================""")

# Plot each pair
for smiles, X, index in smiles_pairs:
    print(f"""=============== smiles is : {smiles}  ===============""")
    plot_molecule_with_shap(smiles, index)


