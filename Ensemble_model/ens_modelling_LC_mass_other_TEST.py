import numpy as np
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle 
from scipy import stats 
from sklearn.ensemble import RandomForestClassifier 
import joblib 
import xlrd
import openpyxl
import os
from os import listdir
from os.path import isfile, join
# Load data
file_path ="E:/project_new/LC_NonLC/DL_model_ENB3andIncV3/combined_conv_features.csv"
#"E:/project_new/LC_NonLC/Features_ML_model_inc/inc_V3_20d_8b_LC_mass_others_features_whole_combined.csv"
#file_path = "D:/clavicle_new_mes_reg_score3/Female_3pt_1pt_new_measurement _whole.xlsx"
data1  = pd.read_csv(file_path)
X = data1.iloc[:1,2:]   #independent columns
print(X) 
X1=X 

list_id = data1.iloc[:1, 0]
list_ID = data1.iloc[:1, 0]
import pickle
import os
print("\n rf_chi2")
def scale_datasets(X, scaler_path):
    #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler.transform(X)

# Define the path to the scaler file
scaler_path = r"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_chi2_BOTH__min_max_w_fec.pkl"
# Print the path to verify
X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))   
with open(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/selected_features_LC_mass_others_rf_chi2_BOTH_w_fec_200.txt", 'r') as file:
    selected_feature_indices = list(map(int, file.read().split(',')))
X_test_selected1 = X_test1.iloc[:, selected_feature_indices]
selected_feature_names = X1.columns[selected_feature_indices]
loaded_SVM_model = joblib.load(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/lbm_BOTH_rf_model_chi2_w_fec_200_train_acc1.0_test_acc0.914235294117647.pkl")
y_pred1=rf_chi2_LC_NR = loaded_SVM_model.predict(X_test_selected1)
#print('y_pred1',y_pred1)
unique_labels = np.unique(y_pred1) 
for i in range (0,len(y_pred1)):
    #print('y_pred1[i]',y_pred1[i])
    if y_pred1[i]==1:
        y_pred1[i]=1
    if y_pred1[i]==0:
        y_pred1[i]=0
rf_chi2_LC_NR=y_pred1
rf_chi2_LC_NR = rf_chi2_LC_NR.reshape(-1, 1)
rf_mi_5m1=[]
for i in range (0,len(rf_chi2_LC_NR)):
    rf_mi_5m1.append(rf_chi2_LC_NR[i,0])
rf_chi2_LC_NR=rf_mi_5m1
#############33
print("\n xgb_chi2")
def scale_datasets(X, scaler_path):
    #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler.transform(X)
# Define the path to the scaler file
scaler_path = r"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_2_LC_mass_other_xgb_chi2__min_max_K_{k}.pkl"
# Print the path to verify
X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))    
with open(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/selected_features_2_LC_mass_other_xgb_chi2_k150.txt", 'r') as file:
    selected_feature_indices = list(map(int, file.read().split(',')))
X_test_selected1 = X_test1.iloc[:, selected_feature_indices]
selected_feature_names = X1.columns[selected_feature_indices]
loaded_SVM_model = joblib.load(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/2_LC_mass_other_xgb_chi2_fec_150_acc1.0.pkl")
y_pred1=xgb_chi2_LC_NR = loaded_SVM_model.predict(X_test_selected1)
#print(y_pred1)
for i in range (0,len(y_pred1)):
    if y_pred1[i]==1:
        y_pred1[i]=1 
    if y_pred1[i]==0:
        y_pred1[i]=0
xgb_chi2_LC_NR=y_pred1  
xgb_chi2_LC_NR = xgb_chi2_LC_NR.reshape(-1, 1)
xgb_chi2_5m1=[]
for i in range (0,len(xgb_chi2_LC_NR)):
    xgb_chi2_5m1.append(xgb_chi2_LC_NR[i,0])
xgb_chi2_LC_NR=xgb_chi2_5m1

#################
print("\n xgb_annova")
def scale_datasets(X, scaler_path):
    #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler.transform(X)
# Define the path to the scaler file
scaler_path = r"E:/project_new/LC_NonLC/Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl"

# Print the path to verify
X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
with open(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/selected_features_LC_mass_others_rf_mutual_info_classif_BOTH_w_fec_150.txt", 'r') as file:
    selected_feature_indices = list(map(int, file.read().split(',')))
X_test_selected1 = X_test1.iloc[:, selected_feature_indices]
selected_feature_names = X1.columns[selected_feature_indices]
loaded_SVM_model = joblib.load(f"E:/project_new/LC_NonLC/Ensemble_model/selected_models/lbm_BOTH_rf_model_mutual_info_classif_w_fec_150_train_acc1.0_test_acc0.914235294117647.pkl")
y_pred1=rf_mi_LC_NR = loaded_SVM_model.predict(X_test_selected1)
#print(y_pred1)
for i in range (0,len(y_pred1)):
    if y_pred1[i]==1:
        y_pred1[i]=1
    if y_pred1[i]==0:
        y_pred1[i]=0
rf_mi_LC_NR=y_pred1 
rf_mi_LC_NR = rf_mi_LC_NR.reshape(-1, 1)
rf_mi_5m1=[]
for i in range (0,len(rf_mi_LC_NR)):
    rf_mi_5m1.append(rf_mi_LC_NR[i,0])
rf_mi_LC_NR=rf_mi_5m1
###############3 
avg_ens=[]
for i in range (0,len(rf_chi2_LC_NR)):
    if rf_chi2_LC_NR[i]+xgb_chi2_LC_NR[i]+rf_mi_LC_NR[i]>1:
        avg_ens.append(1)
    else:
        avg_ens.append(0)        
############### ens_STACK
### insert the name of the column as a string in brackets
list1 = rf_chi2_LC_NR
list2 = xgb_chi2_LC_NR
list3 = rf_mi_LC_NR 
list6 = data1.iloc[:1,1] 
#print('stacked_ML_ML_LCmass and others')
preds_model1=list1
preds_model2=list2
preds_model3=list3 

# Combine predictions into a feature matrix
X_stack = np.column_stack((preds_model1, preds_model2, preds_model3))

# Load the saved stacked ensemble model from the file
loaded_model = joblib.load('stacked_ensemble_model_ML_LCmass_others.pkl')
predicted_value = loaded_model.predict(X_stack)
print('st_predicted_value',predicted_value)
