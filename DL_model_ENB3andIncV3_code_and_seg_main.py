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
    ###  DL model
    import DL_code_full
    from DL_code_full import DL_code
    predicted_proba_DL,predicted_value,img_path,image_path=DL_code(image_path,eff_model,inc_model,rf_chi2_ens,xgb_chi2_ens,rf_mi_ens,ens_scaler_rf_chi2,ens_scaler_xgb_chi2,ens_scaler_rf_mi,st_ens_LC_NR)
    #############   Segmentation
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

##    #import streamlit as st
##    import concurrent.futures
##    import seg_code_v11
##    from seg_code_v11 import seg_code
##    #import shutil  # <- needed for copy
##    
##    TIMEOUT = 15  # seconds
##    
##    inputs = (
##        current_dir, img_p, yolov11, image_path,
##        predicted_proba_DL, predicted_value,
##        sel_ens_M1, sel_ens_M2, sel_ens_M3,
##        scaled_ens_M1, scaled_ens_M2, scaled_ens_M3,
##        ens_MCN
##    )
##    
##    with st.spinner("🔄 Running segmentation..."):
##        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
##            future = executor.submit(seg_code, *inputs)
##            try:
##                imp_result, max_confidence_ML = future.result(timeout=TIMEOUT)
##                st.success("✅ Segmentation completed successfully!")
##                st.write("Result:", imp_result)
##                st.write("Max confidence (ML):", max_confidence_ML)
##    
##            except concurrent.futures.TimeoutError:
##                st.error(f"❌ Seg_code took longer than {TIMEOUT} seconds!")
##                if predicted_value[0] == 1:
##                    imp_result = 'Non-Lung Cancer'
##                    max_confidence_ML = predicted_proba_DL
##                    shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
##                elif predicted_value[0] == 0:
##                    imp_result = 'Lung Cancer'
##                    max_confidence_ML = predicted_proba_DL
##                    shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
##    
##            except Exception as e:
##                st.error(f"❌ Seg_code failed: {e}")
##                if predicted_value[0] == 1:
##                    imp_result = 'Non-Lung Cancer'
##                    max_confidence_ML = predicted_proba_DL
##                    shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
##                elif predicted_value[0] == 0:
##                    imp_result = 'Lung Cancer'
##                    max_confidence_ML = predicted_proba_DL
##                    shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
               

