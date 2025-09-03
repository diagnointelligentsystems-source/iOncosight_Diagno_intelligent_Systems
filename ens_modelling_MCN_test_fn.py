

# Visualization
def ens_ML_MCN(sel_ens_M1,sel_ens_M2,sel_ens_M3,scaled_ens_M1,scaled_ens_M2,scaled_ens_M3,ens_MCN):
    # Data handling
    import numpy as np
    import pandas as pd
    # Utilities
    import joblib
    import pickle
    # Load data
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    file_path ="./yolov11_MCN_whole_features_test.csv" 
    data1  = pd.read_csv(file_path)
    X = data1.iloc[:1,2:]   #independent columns
    print(X) 
    X1=X 

    list_id = data1.iloc[:1, 0]
    list_ID = data1.iloc[:1, 0]

    print("\n  rf_annova")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path =scaled_ens_M1 # r"./selected_models/1_scaler_ALL_FEATURE_5m_SCORE_rf_f_classif_BOTH__min_max_w_fec.pkl"

    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path)) 
    with open(f"./selected_models/1_selected_features_MCN_rf_f_classif_BOTH_w_fec_51.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]
    #print("Selected features:", selected_feature_names)

    loaded_SVM_model = sel_ens_M1 #joblib.load(f"./selected_models/1_MCN_rf_model_f_classif_fec_51_train_acc1.0_test_acc1.0.pkl")
    y_pred1=rf_annova_MCN = loaded_SVM_model.predict(X_test_selected1)
    print('y_pred1',y_pred1)

    for i in range (0,len(y_pred1)):
        #print('y_pred1[i]',y_pred1[i])
        if y_pred1[i]==2:
            y_pred1[i]=2
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==0:
            y_pred1[i]=0
    rf_annova_MCN=y_pred1 
    rf_annova_MCN = rf_annova_MCN.reshape(-1, 1)
    rf_mi_5m1=[]
    for i in range (0,len(rf_annova_MCN)):
        rf_mi_5m1.append(rf_annova_MCN[i,0])
    rf_annova_MCN=rf_mi_5m1
    #############33
    print("\n  RF_MI")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path =scaled_ens_M2 # r"./selected_models/2_scaler_ALL_FEATURE_5m_SCORE_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl"
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path)) 
    with open(f"./selected_models/2_selected_features_MCN_rf_mutual_info_classif_BOTH_w_fec_51.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices] 

    loaded_SVM_model = sel_ens_M2 #joblib.load(f"./selected_models/2_MCN_rf_model_mutual_info_classif_fec_51_train_acc1.0_test_acc1.0.pkl")


    y_pred1=RF_MI_MCN = loaded_SVM_model.predict(X_test_selected1)
    print(y_pred1)
    for i in range (0,len(y_pred1)):
        if y_pred1[i]==2:
            y_pred1[i]=2
        if y_pred1[i]==1:
            y_pred1[i]=1 
        if y_pred1[i]==0:
            y_pred1[i]=0
    RF_MI_MCN=y_pred1 

    RF_MI_MCN = RF_MI_MCN.reshape(-1, 1)
    RF_MI_5m1=[]
    for i in range (0,len(RF_MI_MCN)):
        RF_MI_5m1.append(RF_MI_MCN[i,0])
    RF_MI_MCN=RF_MI_5m1

    #################

    print("\n  xgb_annova")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path =scaled_ens_M3#  r"./selected_models/3_scaler_ALL_FEATURE_3_MCN_xgb_mutual_info_classif__min_max_K_{k}.pkl"

    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path)) 
    with open(f"./selected_models/3_selected_features_3_MCN_xgb_mutual_info_classif_k51.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices] 
    loaded_SVM_model = sel_ens_M3# joblib.load(f"./selected_models/3_MCN_xgb_mutual_info_classif_fec_51_train_acc1.0_test_acc1.0.pkl")


    y_pred1=xgb_mi_MCN = loaded_SVM_model.predict(X_test_selected1)
    print(y_pred1)
    for i in range (0,len(y_pred1)):
        if y_pred1[i]==2:
            y_pred1[i]=2
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==0:
            y_pred1[i]=0
    xgb_mi_MCN=y_pred1

    xgb_mi_MCN = xgb_mi_MCN.reshape(-1, 1)
    xgb_mi_5m1=[]
    for i in range (0,len(xgb_mi_MCN)):
        xgb_mi_5m1.append(xgb_mi_MCN[i,0])
    xgb_mi_MCN=xgb_mi_5m1



    ###############3
    ############### ens_STACK
    ### insert the name of the column as a string in brackets
    list1 = rf_annova_MCN
    list2 = RF_MI_MCN
    list3 = xgb_mi_MCN  
    print('/n')
    print('stacked_MCN')


    preds_model1=list1
    preds_model2=list2
    preds_model3=list3 

    # Combine predictions into a feature matrix
    X_stack = np.column_stack((preds_model1, preds_model2, preds_model3)) 



    # Use the loaded model for prediction on the single test value 
    loaded_model = ens_MCN #joblib.load('stacked_ensemble_model_ML_MCN.pkl')
    predicted_value = (loaded_model.predict(X_stack))[0]
    print('st_predicted_value',predicted_value)
    predicted_proba = loaded_model.predict_proba(X_stack)

    # Display

    print("Class Probabilities:", predicted_proba)
    return predicted_value,predicted_proba




