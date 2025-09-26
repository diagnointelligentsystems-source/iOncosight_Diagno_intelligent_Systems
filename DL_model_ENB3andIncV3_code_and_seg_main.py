def full_code(image_path,eff_model,inc_model,rf_chi2_ens,xgb_chi2_ens,rf_mi_ens,ens_scaler_rf_chi2,ens_scaler_xgb_chi2,ens_scaler_rf_mi,
             st_ens_LC_NR,sel_ens_M1,sel_ens_M2,sel_ens_M3,scaled_ens_M1,scaled_ens_M2,scaled_ens_M3,ens_MCN,yolov11):
    import streamlit as st
    import cv2
    import os
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import pickle
    import warnings
    import tensorflow as tf
    import keras
    from tensorflow import keras
    import shutil
    import matplotlib.pyplot as plt
    import gc
    import traceback
    import psutil
    def print_free_memory():
        mem = psutil.virtual_memory()
        st.write(f"💾 Total Memory: {mem.total / (1024**3):.2f} GB")
        st.write(f"💾 Available Memory: {mem.available / (1024**3):.2f} GB")
        st.write(f"💾 Used Memory: {mem.used / (1024**3):.2f} GB")
        st.write(f"💾 Memory Usage: {mem.percent}%")
        print(f"💾 Total Memory: {mem.total / (1024**3):.2f} GB")
        print(f"💾 Available Memory: {mem.available / (1024**3):.2f} GB")
        print(f"💾 Used Memory: {mem.used / (1024**3):.2f} GB")
        print(f"💾 Memory Usage: {mem.percent}%")
    def log_memory(note=""):
      process = psutil.Process()
      print(f"[{note}] RSS memory: {process.memory_info().rss / (1024**2):.2f} MB")

    
    # Example usage
    print_free_memory()
    def log_memory_usage(note=""):
      process = psutil.Process(os.getpid())
      mem = process.memory_info().rss / (1024 * 1024)  # in MB
      st.write(f"🧠 Memory at {note}: {mem:.2f} MB")
               
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    #new_dir = "E:/project_new/Project_MCN_code"  # replace with your desired folder
    #os.chdir(new_dir)
    ################  getting input
    img_path = image_path
    imp_result=[]
    predicted_value=[]
    region_rows1 = []

    # ==================================
    # ================================
    # Load models
    # ================================
    # ================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = "output_poly_feret/region_stats_with_class.csv"
    file_path = os.path.join(current_dir, file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {os.path.exists(file_path)}", flush=True)
    #else:
     #   print("File does not exist.")
    # Preprocessing
    # ================================
    def preprocess_image(img_path, img_size):
        """Resize, normalize, add batch dim."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"❌ Could not read image: {img_path}")
        resized_img = cv2.resize(img, (600, 600))
        img = cv2.resize(resized_img, (img_size, img_size))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # ================================
    # Feature extraction
    # ================================
    def extract_features_dual(img_path, eff_model, inc_model,
                              dummy_label="Unknown",
                              eff_layer_index=-2, inc_layer_index=-2):
        """
        Extract features from EfficientNetB3 + InceptionV3
        using specified layer indices (default: -2 = second last).
        """

        # Preprocess for each
        img_eff = preprocess_image(img_path, img_size=300)
        img_inc = preprocess_image(img_path, img_size=299)
        ### model predction alone
        eff_pred = eff_model.predict(img_eff, verbose=0)[0]  # EfficientNet prediction
        inc_pred = inc_model.predict(img_inc, verbose=0)[0]

        # EfficientNet features
        eff_layer = eff_model.layers[eff_layer_index].name
        eff_intermediate = keras.Model(inputs=eff_model.input,
                                       outputs=eff_model.layers[eff_layer_index].output)
        eff_features = eff_intermediate.predict(img_eff).flatten()

        # InceptionV3 features
        inc_layer = inc_model.layers[inc_layer_index].name
        inc_intermediate = keras.Model(inputs=inc_model.input,
                                       outputs=inc_model.layers[inc_layer_index].output)
        inc_features = inc_intermediate.predict(img_inc).flatten()

        # Combine
        combined = np.concatenate([eff_features, inc_features])

        # Columns
        col_eff = [f"A{i}" for i in range(len(eff_features))]
        col_inc = [f"{i}" for i in range(len(inc_features))]

        # DataFrame
        df = pd.DataFrame([combined], columns=col_eff + col_inc)
        df.insert(0, "filename", img_path)
        df.insert(1, "label", dummy_label)

        #print(f"✅ Extracted EfficientNet layer: {eff_layer}, Inception layer: {inc_layer}")
        return df,img_eff,eff_pred,inc_pred

    # ================================
    # Example
    # ================================
    features_df,img_eff,eff_pred,inc_pred = extract_features_dual(
        img_path, eff_model, inc_model,
        dummy_label="TestImage",
        eff_layer_index=-2,   # adjust if needed
        inc_layer_index=-2    # adjust if needed
    )
    print ('#****************************************', flush=True)
    print('efficientNetB3', eff_pred, 'InceptionV3', inc_pred, flush=True)
    print( '# ****************************************', flush=True)

    
    #print(features_df.head())
    features_df.to_csv("./DL_model_ENB3andIncV3/combined_conv_features.csv", index=False)
    #print('saved')
    ### grade cam
    import grad_cam_img
    from grad_cam_img import grad_Cam_1
    img_p=grad_Cam_1(image_path,img_path,eff_model)
        #############    ML model

    # Load data
    file_path ="./DL_model_ENB3andIncV3/combined_conv_features.csv" 
    data1  = pd.read_csv(file_path)
    X = data1.iloc[:1,2:]   #independent columns
    #print(X)
    X1=X

    list_id = data1.iloc[:1, 0]
    list_ID = data1.iloc[:1, 0]
    ##print("\n rf_chi2")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path = ens_scaler_rf_chi2#r"./Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_chi2_BOTH__min_max_w_fec.pkl"
    # Print the path to verify
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
    with open(f"./Ensemble_model/selected_models/selected_features_LC_mass_others_rf_chi2_BOTH_w_fec_200.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))
    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]
    selected_feature_names = X1.columns[selected_feature_indices]
    # Load model
    loaded_SVM_model = rf_chi2_ens#joblib.load(rf_chi2_ens)
    st.success("Model loaded successfully!")
    #loaded_SVM_model = joblib.load(f"./Ensemble_model/selected_models/lbm_BOTH_rf_model_chi2_w_fec_200_train_acc1.0_test_acc0.914235294117647.pkl")
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
    ##print("\n xgb_chi2")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)
    # Define the path to the scaler file
    scaler_path = ens_scaler_xgb_chi2
    #r"./Ensemble_model/selected_models/scaler_ALL_FEATURE_2_LC_mass_other_xgb_chi2__min_max_K_{k}.pkl"
    # Print the path to verify
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
    with open(f"./Ensemble_model/selected_models/selected_features_2_LC_mass_other_xgb_chi2_k150.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))
    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]
    selected_feature_names = X1.columns[selected_feature_indices]
    loaded_SVM_model = xgb_chi2_ens #joblib.load(f"./Ensemble_model/selected_models/2_LC_mass_other_xgb_chi2_fec_150_acc1.0.pkl")
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
    ##print("\n xgb_annova")
    def scale_datasets(X, scaler_path):
        #print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)
    # Define the path to the scaler file
    scaler_path =ens_scaler_rf_mi# r"./Ensemble_model/selected_models/scaler_ALL_FEATURE_LC_mass_other_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl"

    # Print the path to verify
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))
    with open(f"./Ensemble_model/selected_models/selected_features_LC_mass_others_rf_mutual_info_classif_BOTH_w_fec_150.txt", 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))
    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]
    selected_feature_names = X1.columns[selected_feature_indices]
    # Google Drive file ID for the new model
    # Load model
    loaded_SVM_model = rf_mi_ens #joblib.load()
    st.success("New model loaded successfully!")

    #loaded_SVM_model = joblib.load(f"./Ensemble_model/selected_models/lbm_BOTH_rf_model_mutual_info_classif_w_fec_150_train_acc1.0_test_acc0.914235294117647.pkl")
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
    print('X_stack ',X_stack , flush=True)
    # Load the saved stacked ensemble model from the file
    loaded_model = st_ens_LC_NR #joblib.load()#'./Ensemble_model/stacked_ensemble_model_ML_LCmass_others.pkl')
    predicted_value = loaded_model.predict(X_stack)

    print('st_predicted_value',predicted_value, flush=True)
    # Get probability scores
    proba_scores = loaded_model.predict_proba(X_stack)

    # Example: if it's binary classification
    # proba_scores[:, 1] gives probability of the positive class
    max_confidence =  np.max(proba_scores, axis=1)  # take highest probability
    predicted_proba_DL = (np.round(max_confidence * 100, 2))[0]

    print("Probability scores:", predicted_proba_DL, flush=True)
    max_confidence_ML=predicted_proba_DL
    if (X_stack[0])[0]==0 and (X_stack[0])[2]==0:
        predicted_value[0]=0
    if (X_stack[0])[0] == 0 and (X_stack[0])[1] == 0 and eff_pred[0] >= 0.5 and inc_pred[0] >= 0.5:
        predicted_value[0] = 0
    if ((X_stack[0])[0] == 0 or (X_stack[0])[1] == 0 or (X_stack[0])[2] == 0 ) and eff_pred[0] >= 0.5 and inc_pred[0] >= 0.5:
        predicted_value[0]=0
    print('predicted_value[0]',predicted_value[0], flush=True)
    plt.close('all')
    print('ex 1', flush=True)
    #### delecting un used data
    # Segmentation code
    import seg_code_v11
    from seg_code_v11 import seg_code
    imp_result,max_confidence_ML=seg_code(current_dir,img_p,yolov11,image_path,predicted_proba_DL,predicted_value,sel_ens_M1, sel_ens_M2, sel_ens_M3, scaled_ens_M1, scaled_ens_M2, scaled_ens_M3, ens_MCN)
    ##del results
    #del result
    #delmodel
    #del df
    #delimg, img1
    gc.collect()
    print('imp_result',imp_result,'max_confidence_ML',max_confidence_ML, flush=True)
    print('ex 9','Analysis completed', flush=True)
    plt.close('all')
    ################3
    return imp_result,max_confidence_ML




