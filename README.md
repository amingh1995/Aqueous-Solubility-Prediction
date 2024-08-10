# Aqueous-Solubility-Prediction
Three ML methods used for aqueous solubility prediction. The general description of python codes for these methods are as follows: 

EdgeConv method
The code loads and preprocesses data, trains an EdgeConv model with EdgeConv layers for feature extraction and regression on ESP maps, evaluates the model, and saves the results for further visualization.

Graph Convolutional Networks
"Graph_Featurizer.py" utilizes the MoleculeDataset class for molecular graph featurization, while "Functionalized_Train_datasets.py" incorporates functionalized training and testing. These functionalities are leveraged in "Train_GNN_datasets.py" to assess the performance of the Graph Neural Network (GNN) across four datasets.

Feature-extracted based 
After feature selection in "RF_Feature_selection.py" the selected feature indices are utilized in "Functionalized_Train_XGBoost_RF_Feat.py" to train XGBoost for solubility prediction based on feature extraction. The results are saved in a dataframe for subsequent visualization.

Plot Functions
"plot_data_visualization.py" generates visualizations corresponding to dataset analysis in the manuscript, while "Plot_models_prediction_results.py" creates plots based on the saved prediction results.

Transferability and Explainability
The corresponding codes are: “Data_Analysis_Explainability.py” and “Transferability_Datasets_Preds.py”.
