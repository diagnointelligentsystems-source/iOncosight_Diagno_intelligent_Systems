def ens_ML_LC_NR():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    from sklearn.svm import SVC
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import xlrd
    import openpyxl
    import os
    from os import listdir
    from os.path import isfile, join
    from sklearn.linear_model import LogisticRegression
    # Load data
    file_path = "./yolov10_LC_NR_whole_features_test.csv"
    data1  = pd.read_csv(file_path)
    X = data1.iloc[:1,2:]   #independent columns
    print(X)
    X1=X

    list_id = data1.iloc[:1, 0]
    list_ID = data1.iloc[:1, 0]
    print("\n rf_annova")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path = r"./LC_NR_ML/Ensemble_model/selected_models/1_scaler_ALL_FEATURE_5m_SCORE_rf_f_classif_BOTH__min_max_w_fec.pkl"

    # Print the path to verify
    #print(f"Scaler path: {repr(scaler_path)}")
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
    with open(f"./LC_NR_ML/Ensemble_model/selected_models/1_selected_features_LC_Normal_rf_f_classif_BOTH_w_fec_361.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]

    loaded_SVM_model = joblib.load(f"./LC_NR_ML/Ensemble_model/selected_models/1_LC_Normal_rf_model_f_classif_fec_361_train_acc1.0_test_acc0.7714285714285715.pkl")


    y_pred1=rf_annova_LC_NR = loaded_SVM_model.predict(X_test_selected1)
    print('y_pred1',y_pred1)
    for i in range (0,len(y_pred1)):
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==0:
            y_pred1[i]=0
    rf_annova_LC_NR=y_pred1
    rf_annova_LC_NR = rf_annova_LC_NR.reshape(-1, 1)
    rf_mi_5m1=[]
    for i in range (0,len(rf_annova_LC_NR)):
        rf_mi_5m1.append(rf_annova_LC_NR[i,0])
    rf_annova_LC_NR=rf_mi_5m1
    #############33
    print("\n xgb_chi2")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path = r"./LC_NR_ML/Ensemble_model/selected_models/2_scaler_ALL_FEATURE_2_LC_nr_xgb_chi2__min_max_K_{k}.pkl"
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
    with open(f"./LC_NR_ML/Ensemble_model/selected_models/2_selected_features_2_LC_nr_xgb_chi2_k361.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]
    loaded_SVM_model = joblib.load(f"./LC_NR_ML/Ensemble_model/selected_models/2_LC_nr_xgb_chi2_fec_361_train_acc1.0_test_acc0.8285714285714286.pkl")

    y_pred1=xgb_chi2_LC_NR = loaded_SVM_model.predict(X_test_selected1)
    print(y_pred1)
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
    scaler_path = r"./LC_NR_ML/Ensemble_model/selected_models/3_scaler_ALL_FEATURE_2_LC_nr_xgb_chi2__min_max_K_{k}.pkl"

    # Print the path to verify
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
    with open(f"./LC_NR_ML/Ensemble_model/selected_models/3_selected_features_2_LC_nr_xgb_chi2_k191.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]

    loaded_SVM_model = joblib.load(f"./LC_NR_ML/Ensemble_model/selected_models/3_LC_nr_xgb_chi2_fec_191_train_acc1.0_test_acc0.8.pkl")

    y_pred1=xgb_mi_LC_NR = loaded_SVM_model.predict(X_test_selected1)
    print(y_pred1)
    for i in range (0,len(y_pred1)):
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==0:
            y_pred1[i]=0
    xgb_mi_LC_NR=y_pred1
    xgb_mi_LC_NR = xgb_mi_LC_NR.reshape(-1, 1)
    xgb_mi_5m1=[]
    for i in range (0,len(xgb_mi_LC_NR)):
        xgb_mi_5m1.append(xgb_mi_LC_NR[i,0])
    xgb_mi_LC_NR=xgb_mi_5m1
    ############### ens_STACK
    list1 = rf_annova_LC_NR
    list2 = xgb_chi2_LC_NR
    list3 = xgb_mi_LC_NR
    print('/n')
    print('stacked_ML_Lung_cancer_and_Normal')

    preds_model1=list1
    preds_model2=list2
    preds_model3=list3

    # Combine predictions into a feature matrix
    X_stack = np.column_stack((preds_model1, preds_model2, preds_model3))
    # Load the saved stacked ensemble model from the file
    loaded_model = joblib.load('./LC_NR_ML/Ensemble_model/stacked_ensemble_model_ML_5m_cl_F.pkl')
    # Use the loaded model for prediction on the single test value
    predicted_value= loaded_model.predict(X_stack)
    # Display the predicted value
    print("Predicted Value:", predicted_value)
    predicted_proba = loaded_model.predict_proba(X_stack)

    # Display

    print("Class Probabilities:", predicted_proba)
    return predicted_value,predicted_proba
