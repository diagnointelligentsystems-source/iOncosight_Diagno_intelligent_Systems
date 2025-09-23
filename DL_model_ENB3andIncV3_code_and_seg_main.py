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
        print(f"Deleted: {os.path.exists(file_path)}")
    else:
        print("File does not exist.")
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
        return df,img_eff

    # ================================
    # Example
    # ================================
    features_df,img_eff = extract_features_dual(
        img_path, eff_model, inc_model,
        dummy_label="TestImage",
        eff_layer_index=-2,   # adjust if needed
        inc_layer_index=-2    # adjust if needed
    )

    #print(features_df.head())
    features_df.to_csv("./DL_model_ENB3andIncV3/combined_conv_features.csv", index=False)
    #print('saved')
    ### grade cam
    if 1==1:
        img = img_p = cv2.imread(image_path)
        ###
        # ---- Grad-CAM Function ----
        def get_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()


        # ---- Preprocess Input Image ----
            # ---- Overlay Grad-CAM on Original Image ----
        def overlay_gradcam(img, heatmap, alpha=0.4, cmap=cv2.COLORMAP_JET):
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cmap)
            img = np.array(img)
            superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
            return superimposed_img


        # ---- Run Grad-CAM ----
        #img_path = "E:/project_new/Project_MCN_code/sample_images/test_image.png"  # change this
        def preprocess_image_g(img_path, target_size=(300, 300)):
            img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # normalize
            return img, img_array


        img1, img_array = preprocess_image_g(img_path, target_size=(300, 300))
        # Find last conv layer name (EfficientNetB3 usually: "top_conv")
        last_conv_layer_name = "top_conv"

        heatmap = get_gradcam(eff_model, img_array, last_conv_layer_name)
        superimposed_img = overlay_gradcam(img1, heatmap)
        cv2.imwrite("./output_YOLOV11/Grad_cam_PRED.png", superimposed_img)
        #################
        # Copy image
    shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
        #############    ML model

    # Load data
    file_path ="./DL_model_ENB3andIncV3/combined_conv_features.csv"
    #"E:/project_new/Project_MCN_code/Features_ML_model_inc/inc_V3_20d_8b_LC_mass_others_features_whole_combined.csv"
    #file_path = "D:/clavicle_new_mes_reg_score3/Female_3pt_1pt_new_measurement _whole.xlsx"
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
    print('X_stack ',X_stack )
    # Load the saved stacked ensemble model from the file
    loaded_model = st_ens_LC_NR #joblib.load()#'./Ensemble_model/stacked_ensemble_model_ML_LCmass_others.pkl')
    predicted_value = loaded_model.predict(X_stack)

    print('st_predicted_value',predicted_value)
    # Get probability scores
    proba_scores = loaded_model.predict_proba(X_stack)

    # Example: if it's binary classification
    # proba_scores[:, 1] gives probability of the positive class
    max_confidence =  np.max(proba_scores, axis=1)  # take highest probability
    predicted_proba_DL = (np.round(max_confidence * 100, 2))[0]

    print("Probability scores:", predicted_proba_DL)
    max_confidence_ML=predicted_proba_DL
    if (X_stack[0])[0]==0 and (X_stack[0])[2]==0:
        predicted_value[0]=0
    print('predicted_value[0]',predicted_value[0])
    plt.close('all')
    print('ex 1')
    ########################## segmentation model
    output_path = "./images_YOLOV11/V11_input.png"
    try: #if 1==1:#predicted_value[0]!=1:
        print('ex 1_1')
        from PIL import Image
        # Load best model
        # Class mapping
        class_names = {0: "Mass", 1: "COPD", 2: "Normal"}
        # Fixed colors (BGR for OpenCV)
        class_colors = {
            "Mass": (0, 0, 255),       # Red
            "COPD": (0, 165, 255),     # Orange
            "Normal": (0, 255, 0)      # Green
        }
        
        # Transparency factor
        alpha = 0.4
        results=[]
        # Load best model
        cwd = os.getcwd()
        print("Current working directory:", cwd)
        # Set directory
        #from ultralytics import YOLO
        model = yolov11 #YOLO(yolov11)#"./yolov11_seg_MCN_best.pt")
        # Input/output
        ############################
        output_path = "./images_YOLOV11/V11_input.png"

        # Make sure the output folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Copy image
        shutil.copy(image_path, output_path)
        print('ex 1_2')
        output_path = "./output_YOLOV11/V11_SEG_PRED.png"
        print('ex 1_2_1')
        # Run inference
        img_samp = cv2.imread(image_path)
        print('image_path :',image_path)
        print("img_samp shape:", img_samp.shape) 
        results = model(image_path, conf=0.5, iou=0.5, imgsz=512, device="cpu")
        print('ex 1_2_3')
        result = results[0]
        print('ex 1_3')
        # Read original image
        img=img_p = cv2.imread(image_path)
        #img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        #print('*0')
        # Run inference
        print('ex 1_4')
        if result.masks is None:   # ✅ check before using
            #print('*1')
            results = model(image_path, conf=0.3, iou=0.5, imgsz=512, device="cpu")
            result = results[0]
            # if result.masks is not None:
            #     print('*11')
        if result.masks is None:   # ✅ check before using
            results = model(image_path, conf=0.05, iou=0.5, imgsz=512, device="cpu")
            result = results[0]
            #print('*22')
            if result.masks is None:
                print(" No segmentation detected even after multiple inference attempts.")
                #print('*2###############################################')
        #print(sfdsfsgag)
        ###### feature extraction
        print('ex 1_5')
        import torch
        from tqdm import tqdm

        # === Load trained YOLOv10 model ===
        model.eval()

        # === Hook to extract backbone features ===
        features_dict = {}

        def hook_fn(module, input, output):
            pooled = torch.mean(output[0], dim=(1, 2))  # Global Average Pooling
            features_dict['feat'] = pooled.detach().cpu().numpy()

        # You might need to adjust this index based on your model structure
        #hook = model.model.model[10].register_forward_hook(hook_fn)
        try:
            hook = model.model.model[10].register_forward_hook(hook_fn)
        except Exception as e:
            print(f"❌ Failed to register hook: {e}")
        print('ex 1_6')
        def extract_features_from_txt(image_folder, save_csv_path):
            data = []
            all_images = sorted(os.listdir(image_folder))
            print('ex 1_7')
            for filename in tqdm(all_images, desc=f"Extracting from {os.path.basename(image_folder)}"):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

                img_path = os.path.join(image_folder, filename)
                #label_path = os.path.join(label_folder, filename.replace('.png', '.txt').replace('.jpg', '.txt'))
                main_class=10
                print('ex 1_8')
                try:
                    _ = model(img_path,imgsz=512, device="cpu")
                    feat = features_dict.get('feat')
                    if feat is None:
                        print(f"⚠️ Feature not extracted for {filename}")
                        continue
                    if feat is not None:
                        row = [filename, main_class] + feat.tolist()
                        data.append(row)
                    print('ex 1_9')
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

            # Save CSV
            if data:
                print('ex 1_10')
                columns = ['filename', 'label'] + [f'feat_{i}' for i in range(len(data[0]) - 2)]
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(save_csv_path, index=False)
                #print(f"✅ Saved features to {save_csv_path}")
                print('ex 1_11')
            else:
                print('ex 1_12')
                print("⚠️ No data was extracted.")
        print('ex 1_13')
        extract_features_from_txt(
            image_folder='./images_YOLOV11',
            save_csv_path='./yolov11_MCN_whole_features_test.csv'
        )
        print('ex 2')
        ######### ML results
        import ens_modelling_MCN_test_fn

        ens_ML_MCN_output,predicted_proba = ens_modelling_MCN_test_fn.ens_ML_MCN(sel_ens_M1,sel_ens_M2,sel_ens_M3,scaled_ens_M1,scaled_ens_M2,scaled_ens_M3,ens_MCN)
        #print("Ens ML results:", ens_ML_MCN_output)
        predicted_proba=predicted_proba[0]
        conf_ML=predicted_proba[ens_ML_MCN_output]*100


        #print('conf_ML',conf_ML)
        cv2.imwrite(output_path, img_p)
        print('ex 3')
        ################3 changing label confidence score
        # -------- Step 1: Apply segmentation masks (without darkening background) --------
        if result.masks is not None and ens_ML_MCN_output<2:   # ✅ check before using (include both Mass and COPD)
            for mask, cls_id in zip(result.masks.data, result.boxes.cls):
                cls_id = int(cls_id)
                # if cls_id>0:
                #     continue
                cls_name = class_names[cls_id]
                color = class_colors[cls_name]

                mask = mask.cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                # Extract the region of interest (ROI) where mask==1
                roi = img[mask == 1]

                # Create same-shape color array
                color_arr = np.full_like(roi, color, dtype=np.uint8)

                # Blend only masked region
                blended = cv2.addWeighted(roi, 1 - alpha, color_arr, alpha, 0)

                # Put back blended pixels
                img[mask == 1] = blended

            copd_p=0
            # -------- Step 2: Draw bounding boxes + labels (with white background) --------
            for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                cls_id = int(cls_id)
                # if cls_id>0:
                #     continue
                cls_name = class_names[cls_id]
                color = class_colors[cls_name]

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Build label with confidence
                conf=conf_ML
                label = f"{cls_name} {conf:.0f}%"
                label_w=cls_name

                # Get text size
                (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # White background rectangle
                cv2.rectangle(img,
                              (x1, y1 - font_h - baseline),
                              (x1 + font_w, y1),
                              (255, 255, 255), -1)

                # Text on top of white background
                if ens_ML_MCN_output==2:
                    color1=(0,0,0)
                    cv2.putText(img, label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                color1, 2)
                else:
                    #print('label+++++++++++++++++++++++++++++++',label)
                    if 'COPD'==label_w:
                        copd_p = 1
                    cv2.putText(img, label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                color, 2)

        # -------- Step 3: Save and show result --------
        if ens_ML_MCN_output == 0 or ens_ML_MCN_output == 1:  #
            cv2.imwrite(output_path, img)
        else:
            cv2.imwrite(output_path, img_p)
        #print(f"Saved to {output_path}")

        # Show with matplotlib (correct colors)
##        plt.figure(figsize=(7, 7))
##        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
##        plt.title("Detected regions")
##        plt.axis("off")
##        plt.tight_layout()
        #plt.show()

        ############## segmented region
        print('ex 4')
        if 1==1:#ens_ML_MCN_output==0: #
            # ----------------------------
            # CLASS NAMES
            # ----------------------------
            class_names = {0: "Mass", 1: "COPD", 2:"Normal"}


            # ==================== helpers ====================

            def feret_diameters_from_contour(cnt):
                if cnt.ndim == 3 and cnt.shape[1] == 1:
                    cnt = cnt[:, 0, :]
                cnt = cnt.astype(np.float32)

                area = float(cv2.contourArea(cnt))
                if len(cnt) < 3:
                    return dict(area=area, major_len=0.0, minor_len=0.0,
                                major_p1=(0, 0), major_p2=(0, 0), major_angle_deg=0.0,
                                minor_p1=(0, 0), minor_p2=(0, 0), minor_angle_deg=0.0)

                hull = cv2.convexHull(cnt)
                if hull.ndim == 3:
                    hull = hull[:, 0, :]
                P = hull.astype(np.float32)
                M = len(P)

                # --- Max Feret ---
                if M > 600:
                    step = int(M / 600) + 1
                    P_major = P[::step]
                else:
                    P_major = P

                A = P_major[:, None, :]
                B = P_major[None, :, :]
                diff = A - B
                D2 = (diff ** 2).sum(-1)
                i, j = np.unravel_index(np.argmax(D2), D2.shape)
                p1 = tuple(P_major[i].astype(float))
                p2 = tuple(P_major[j].astype(float))
                major_len = float(np.sqrt(D2[i, j]))
                major_angle_deg = float((np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) + 180) % 180)

                # --- Min Feret (rotating calipers) ---
                rect = cv2.minAreaRect(P)
                (cx, cy), (w, h), angle = rect
                if w < h:
                    min_len = w
                    min_angle_deg = angle
                else:
                    min_len = h
                    min_angle_deg = angle + 90
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                dists = [np.linalg.norm(box[(k + 1) % 4] - box[k]) for k in range(4)]
                kmin = int(np.argmin(dists))
                q1 = tuple(box[kmin])
                q2 = tuple(box[(kmin + 1) % 4])

                return dict(
                    area=area,
                    major_len=major_len, major_p1=p1, major_p2=p2, major_angle_deg=major_angle_deg,
                    minor_len=float(min_len), minor_p1=q1, minor_p2=q2, minor_angle_deg=min_angle_deg
                )

            print('ex 5')
            # ==================== main ====================

            def process_segmentation(image_path, results,predicted_value):
                copd_p=0
                orig_img = np.array(Image.open(image_path).convert("RGB"))
                H, W = orig_img.shape[:2]

                if results[0].masks is None:
                    print("No segmentation detected.")
                    return copd_p

                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                region_rows = []
                overlay = orig_img.copy()
                red = np.array([255, 0, 0], dtype=np.uint8)
                alpha = 0.35

                os.makedirs("./output_poly_feret", exist_ok=True)

                for idx, m in enumerate(masks, start=0):  # keep index aligned with boxes
                    cls_id233 = class_ids[idx]
                    label ='none'
                    conf=0
                    # 🚨 skip anything that is not class 0
                    #print('cls_id233',cls_id233)
                    if cls_id233 > 1:
                        continue
                    m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)  ########### resize to original
                    mask_bin = (m_resized > 0.5).astype(np.uint8)
                    if mask_bin.sum() == 0:
                        continue

                    m_idx = mask_bin.astype(bool)
                    if cls_id233 == 0 and predicted_value[0]!=1:
                        overlay[m_idx] = (alpha * np.array([255, 0, 0]) + (1 - alpha) * overlay[m_idx]).astype(np.uint8)
                    cnts, _ = cv2.findContours((mask_bin * 255).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # ✅ Case 2: bounding box with label (class 1)
                    if cls_id233 == 1:
                        # get box coords and clamp to image
                        box = boxes.xyxy[idx].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(x1));
                        y1 = max(0, int(y1))
                        x2 = min(W - 1, int(x2));
                        y2 = min(H - 1, int(y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # percentage confidence (use your confidences array)
                        #conf_ML = confidences[idx] * 100  # convert to percent if conf is 0..1
                        label_p = f"COPD ({conf_ML:.0f})%"
                        label = f"COPD"
                        if label == f"COPD":
                            if copd_p==0:
                                copd_p=1
                        # colors (BGR)
                        sandal = (255, 204, 102)  # sandal yellow (RGB)
                        dark_orange = (255, 255, 255)  # dark orange (RGB)
                        bbox_border_color = (255, 140, 0)

                        # ---------- 1) Fill the bbox with semi-transparent sandal ----------
                        alpha_fill = 0.35
                        sub = overlay[y1:y2, x1:x2].copy()
                        if sub.size != 0:
                            sandal_rect = np.full(sub.shape, sandal, dtype=np.uint8)
                            cv2.addWeighted(sandal_rect, alpha_fill, sub, 1 - alpha_fill, 0, sub)
                            overlay[y1:y2, x1:x2] = sub

                        # optional: draw bbox border (keeps a visible outline)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), bbox_border_color, 2)

                        # ---------- 2) Draw an opaque label bar on top INSIDE the bbox ----------
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        padding = 6

                        # auto-shrink font if label is wider than bbox
                        (text_w, text_h), baseline = cv2.getTextSize(label_p, font, font_scale, text_thickness)
                        box_width = x2 - x1
                        while (text_w + 2 * padding) > box_width and font_scale > 0.25:
                            font_scale -= 0.05
                            (text_w, text_h), baseline = cv2.getTextSize(label_p, font, font_scale, text_thickness)

                        # label rectangle coordinates (inside top of bbox)
                        label_x1 = x1
                        label_x2 = x1 + text_w + 2 * padding
                        if label_x2 > x2:
                            label_x2 = x2
                        label_y1 = y1
                        label_y2 = y1 + text_h + 2 * padding + baseline

                        # clamp vertical coords
                        if label_y2 > H:
                            label_y2 = H
                            label_y1 = max(0, label_y2 - (text_h + 2 * padding + baseline))

                        # draw opaque sandal label bar
                        cv2.rectangle(overlay,
                                      (int(label_x1), int(label_y1)),
                                      (int(label_x2), int(label_y2)),
                                      sandal,
                                      thickness=-1)

                        # ---------- 3) Put the text inside the label bar (dark orange) ----------
                        text_org = (int(label_x1 + padding), int(label_y2 - baseline - padding))
                        cv2.putText(overlay, label_p, text_org, font, font_scale, dark_orange, text_thickness, cv2.LINE_AA)

                    if not cnts:
                        continue
                    cnt = max(cnts, key=cv2.contourArea)

                    # # Approx polygon
                    # epsilon = 0.01 * cv2.arcLength(cnt, True)
                    # approx = cv2.approxPolyDP(cnt, epsilon, True)
                    # print('len(approx)',len(approx))
                    # if len(approx) < 10:
                    #     print(f"Skipping region {idx + 1}: not a polygon")
                    #     continue
                    if cls_id233 == 0 and predicted_value[0]!=1:
                        stats = feret_diameters_from_contour(cnt)

                        # --- Class + Confidence ---
                        cls_id = class_ids[idx]
                        conf = confidences[idx]
                        label = class_names.get(cls_id, str(cls_id))

                        # --- Draw major axis line ---
                        p1 = tuple(map(int, stats['major_p1']))
                        p2 = tuple(map(int, stats['major_p2']))
                        cv2.line(overlay, p1, p2, (0, 255, 0), 2)

                    # --- Label ---
                    def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                                                  font_scale=0.6, text_color=(255, 0, 0),  # Red text
                                                  bg_color=(255, 0, 255), thickness=2, padding=4):  # Yellow background

                        # Get text size
                        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                        # Coordinates for background rectangle
                        x, y = org
                        cv2.rectangle(img, (x, y - h - baseline - padding),
                                      (x + w + padding * 2, y + baseline + padding),
                                      bg_color, -1)

                        # Put text over the background
                        cv2.putText(img, text, (x + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                        return img
                    if cls_id233 <2:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # --- Place text near region instead of top-left ---
                        # Conversion factors from your resized DICOM (640x640)
                        px_to_cm = 0.034365    # 0.0531  # cm per pixel
                        px_to_cm2 = 0.034365*0.033193  #0.00114 #0.00292  # cm² per pixel
                        if cls_id233 ==0 and predicted_value[0]!=1:
                            txt = [
                                f"{label} ({conf_ML:.0f}%)",
                                f"Area: {stats['area'] * px_to_cm2:.2f} cm2",
                                f"Length: {stats['major_len'] * px_to_cm:.2f} cm",
                                # f"MinFeret: {stats['minor_len'] * px_to_cm:.2f} cm"
                            ]
                            # Start y above the bounding box (or inside if too close to top)
                            y0 = max(y - 10, 20)
                            dy = 30
                            print("overlay Image shape:", overlay.shape)
                            for i, t in enumerate(txt):
                                yy = y0 + i * dy
                                if i == 0:  # label + confidence → red text on yellow background
                                    draw_text_with_background(
                                        overlay, t, (x + w, yy),
                                        text_color=(255, 0, 0),  # red
                                        bg_color=(255, 255, 0)  # yellow
                                    )
                                else:  # white plain text for stats
                                    cv2.putText(
                                        overlay, t, (x + w, yy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 2, cv2.LINE_AA
                                    )
                    # --- Save row ---
                    # Conversion factors (for your resized 640x640 case)
                    #0.034365×0.033193 cm 1024*1024
                    px_to_mm_x, px_to_mm_y = 0.34365 ,0.33193       #0.550, 0.531  (640*640)
                    px_to_cm_x, px_to_cm_y = px_to_mm_x / 10, px_to_mm_y / 10
                    px_to_mm2 = px_to_mm_x * px_to_mm_y
                    px_to_cm2 = px_to_mm2 / 100

                    region_rows.append({
                        "Region": idx + 1,
                        "Class": label,
                        #"Confidence": float(conf),
                        #"Area_px2": stats['area'],
                        #"Area_cm2": stats['area'] * px_to_cm2,
                        #"MaxFeret_px": stats['major_len'],
                        #"MaxFeret_cm": stats['major_len'] * ((px_to_cm_x + px_to_cm_y) / 2),  # avg if spacing not square
                        #"MinFeret_px": stats['minor_len'],
                        #"MinFeret_cm": stats['minor_len'] * ((px_to_cm_x + px_to_cm_y) / 2),
                        #"MaxFeret_angle_deg": stats['major_angle_deg'],
                        #"MinFeret_angle_deg": stats['minor_angle_deg'],
                    })
                plt.figure(figsize=(7, 7))
                plt.imshow(overlay)
                plt.title("Overlay + Polygon Feret Diameters + Class")
                plt.axis("off")
                plt.tight_layout()
                #plt.show()

                #cv2.imwrite("./output_poly_feret/overlay_with_class.png",
                            #cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./output_YOLOV11/V11_SEG_PRED.png",  cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                df = pd.DataFrame(region_rows)
                df.to_csv(os.path.join(current_dir, "output_poly_feret", "region_stats_with_class.csv"), index=False)
                #print(df)
                return copd_p

            copd_p=process_segmentation(image_path, results,predicted_value)
            print('ex 6')
    #else:
     #   print('predicted output: Non-Lung Cancer')
    #print('classfier_output',predicted_value[0])
    except Exception as e:
        print("⚠️ An error occurred in process_segmentation:")
        traceback.print_exc()   # prints full traceback with line number
        raise  # re-raise so the real error propagates
    #############3
    # Extract all class values
    #df = pd.DataFrame(region_rows)
    #df.to_csv("./output_poly_feret/region_stats_with_class.csv", index=False)
    #print(df)
    has_mass = has_copd = False
    if os.path.exists(os.path.join(current_dir, "output_poly_feret", "region_stats_with_class.csv")):
        from pandas.errors import EmptyDataError

        csv_path = os.path.join(current_dir, "output_poly_feret", "region_stats_with_class.csv")

        df = None  # initialize
        has_mass = has_copd = False
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print("CSV loaded successfully:", csv_path)
            except EmptyDataError:
                print("CSV file exists but is empty:", csv_path)
        else:
            print("CSV file does not exist:", csv_path)

        # ✅ Only run this block if df was successfully loaded
        if df is not None and not df.empty:
            # Check if Class column exists before accessing
            if "Class" in df.columns:
                has_mass = (df["Class"] == "Mass").any()
                has_copd = (df["Class"] == "COPD").any()
            else:
                has_mass = has_copd = False
                print("⚠️ 'Class' column not found in CSV")

            if predicted_value[0] == 0:
                max_confidence_ML = predicted_proba_DL
                imp_result = "Lung Cancer"
                if has_mass and (has_copd or copd_p == 1):
                    imp_result = "Lung Cancer + Mass + COPD"
                    max_confidence_ML = conf_ML
                elif has_mass:
                    imp_result = "Lung Cancer + Mass"
                    max_confidence_ML = conf_ML
                elif has_copd or (copd_p == 1):
                    imp_result = "Lung Cancer + COPD"
                    max_confidence_ML = conf_ML
            else:
                imp_result = "Non-Lung Cancer"
                max_confidence_ML = predicted_proba_DL
                if has_copd:
                    imp_result = "COPD (High Risk for Lung Cancer)"
                    max_confidence_ML = conf_ML
        else:
            # fallback if df missing/empty
            if predicted_value[0] == 0:
                imp_result = 'Lung Cancer'
                max_confidence_ML = predicted_proba_DL
                shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
                imp_image_out = "./result.jpg"
            else:
                imp_result = 'Non-Lung Cancer'
                max_confidence_ML = predicted_proba_DL
                shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
                imp_image_out = "./result.jpg"
        print('ex 7')
        if imp_result != 'Non-Lung Cancer':
            imp_image_out2 = "./output_YOLOV11/Grad_cam_PRED.png"
            imp_image_out1 = "./output_YOLOV11/V11_SEG_PRED.png"
            imp_image_out = "./result.jpg"  # output_YOLOV11/concat_PRED.png"
            ### concat image
            # Read images
            img1 = cv2.imread(imp_image_out1)
            img2 = cv2.imread(imp_image_out2)

            # Make sure both images have the same height
            if img1.shape[0] != img2.shape[0]:
                # Resize second image height same as first
                img2 = cv2.resize(img2, (int(img2.shape[1] * img1.shape[0] / img2.shape[0]), img1.shape[0]))

            # Resize final image to width = 1024, keep aspect ratio
            # h, w = img2.shape[:2]
            # new_w = 1024
            # new_h = int(h * (new_w / w))
            # img1 = cv2.resize(img1,  (new_w, new_h))
            # Concatenate horizontally
            concat_img_resized = np.vstack((img2, img1))

            # Save output
            cv2.imwrite(imp_image_out, concat_img_resized)

    else:
        if predicted_value[0]==0:
            imp_result='Lung Cancer'
            max_confidence_ML = predicted_proba_DL
            imp_image_out = "./result.jpg"#"./output_YOLOV11/Grad_cam_PRED.png"
        else:
            if copd_p==1:
                max_confidence_ML = conf_ML
                imp_result = " COPD (High Risk for Lung Cancer)"
                imp_image_out2 = "./output_YOLOV11/Grad_cam_PRED.png"
                imp_image_out1 = "./output_YOLOV11/V11_SEG_PRED.png"
                imp_image_out = "./result.jpg"#output_YOLOV11/concat_PRED.png"
                ### concat image
                # Read images
                img1 = cv2.imread(imp_image_out1)
                img2 = cv2.imread(imp_image_out2)

                # Make sure both images have the same height
                if img1.shape[0] != img2.shape[0]:
                    # Resize second image height same as first
                    img2 = cv2.resize(img2, (int(img2.shape[1] * img1.shape[0] / img2.shape[0]), img1.shape[0]))

                # Resize final image to width = 1024, keep aspect ratio
                #h, w = img2.shape[:2]
                #new_w = 1024
                #new_h = int(h * (new_w / w))
                #img1 = cv2.resize(img1,  (new_w, new_h))
                # Concatenate horizontally
                concat_img_resized= np.vstack((img2, img1))

                # Save output
                cv2.imwrite(imp_image_out, concat_img_resized)
                #imp_result=='COPD'

            else:
                imp_result = 'Non-Lung Cancer'
                max_confidence_ML = predicted_proba_DL
                imp_image_out="./result.jpg"#"./output_YOLOV11/Grad_cam_PRED.png"
    print('ex 8')
    if imp_result=='Lung Cancer':
      shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
    plt.close('all')
    print('ex 9')
    ################3

    return imp_result,max_confidence_ML























