import os
os.environ["STREAMLIT_WATCHDOG"] = "false"
import streamlit as st
 
# Clear any old states at the very start
if "initialized" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.processed_result = None
    st.session_state.report_data = None
    st.session_state.show_report = False
    st.session_state.completed = False
    st.session_state.initialized = True
 
import time
from datetime import datetime
from PIL import Image
import io
import base64
import json
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from huggingface_hub import hf_hub_download, whoami
import joblib
from ultralytics import YOLO
from datetime import datetime
now = datetime.now()
print(now)                           # full date & time
import ultralytics
print(ultralytics.__version__ , flush=True)
import gc
import psutil
import sys
import threading
# ----------------------------
# Set Hugging Face token safely
# ----------------------------
hf_token = st.secrets["HF_TOKEN"]
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
##############
import streamlit as st
import psutil, os

MEMORY_LIMIT_MB = 3072  # ~0.8 GB for Streamlit Cloud

def check_app_memory():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    st.sidebar.metric("App Memory Usage (MB)", f"{mem_mb:.2f}")

    if mem_mb > MEMORY_LIMIT_MB:
        st.warning(f"⚠️ App memory too high ({mem_mb:.2f} MB). Resetting session...")
        # Clear relevant session state
        for key in ["uploaded_file", "processed_result", "report_data", 
                    "show_report", "completed"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()  # safe rerun in Streamlit

# Call this at the top of your script
check_app_memory()

 
# ----------------------------
# Deep Learning Models
# ----------------------------
@st.cache_resource
def load_yolo():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="yolov11_seg_MCN_best.pt"
    )
    return YOLO(model_path) #if you need to use YOLO class


@st.cache_resource
def load_eff_model():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="model_LCm_others_B3_20d_8b_m300_ly1024_ly512.keras"
    )
    return keras.models.load_model(model_path, compile=False)


@st.cache_resource
def load_inc_model():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="model_LCm_others_V3_20d_64b_m299_ly1024_ly512.keras"
    )
    return keras.models.load_model(model_path, compile=False)


# ----------------------------
# Machine Learning Models
# ----------------------------
@st.cache_resource
def load_rf_chi2():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="lbm_BOTH_rf_model_chi2_w_fec_200_train_acc1.0_test_acc0.914235294117647.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_xgb_chi2():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="2_LC_mass_other_xgb_chi2_fec_150_acc1.0.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_rf_mi():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="lbm_BOTH_rf_model_mutual_info_classif_w_fec_150_train_acc1.0_test_acc0.914235294117647.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_stacked_LC_NR():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="stacked_ensemble_model_ML_LCmass_others.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_sel_ens_M1():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="1_MCN_rf_model_f_classif_fec_51_train_acc1.0_test_acc1.0.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_sel_ens_M2():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="2_MCN_rf_model_mutual_info_classif_fec_51_train_acc1.0_test_acc1.0.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_sel_ens_M3():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="3_MCN_xgb_mutual_info_classif_fec_51_train_acc1.0_test_acc1.0.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_ens_MCN():
    model_path = hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename="stacked_ensemble_model_ML_MCN.pkl"
    )
    return joblib.load(model_path)


# ----------------------------
# Scalers (use cache_data since they are small)
# ----------------------------
@st.cache_data
def load_scaler(filename: str):
    return hf_hub_download(
        repo_id="DiagnoIntelligentSytem/lung-xray-models",
        filename=filename
    )


# ----------------------------
# Initialize all models once
# ----------------------------
yolov11 = load_yolo()
eff_model = load_eff_model()
inc_model = load_inc_model()

rf_chi2_ens = load_rf_chi2()
xgb_chi2_ens = load_xgb_chi2()
rf_mi_ens = load_rf_mi()
st_ens_LC_NR = load_stacked_LC_NR()

sel_ens_M1 = load_sel_ens_M1()
sel_ens_M2 = load_sel_ens_M2()
sel_ens_M3 = load_sel_ens_M3()
ens_MCN = load_ens_MCN()

ens_scaler_rf_chi2 = load_scaler("scaler_ALL_FEATURE_LC_mass_other_rf_chi2_BOTH__min_max_w_fec.pkl")
ens_scaler_xgb_chi2 = load_scaler("scaler_ALL_FEATURE_2_LC_mass_other_xgb_chi2__min_max_K_{k}.pkl")
ens_scaler_rf_mi = load_scaler("scaler_ALL_FEATURE_LC_mass_other_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl")
scaled_ens_M1 = load_scaler("1_scaler_ALL_FEATURE_5m_SCORE_rf_f_classif_BOTH__min_max_w_fec.pkl")
scaled_ens_M2 = load_scaler("2_scaler_ALL_FEATURE_5m_SCORE_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl")
scaled_ens_M3 = load_scaler("3_scaler_ALL_FEATURE_3_MCN_xgb_mutual_info_classif__min_max_K_{k}.pkl")

# ----------------------------
# Optional: Check token
# ----------------------------
try:
    user_info = whoami()  # This should succeed now
    st.write("Logged in as:", user_info["name"])
except Exception as e:
    st.error(f"Token is invalid or missing: {e}")
# Load models (cached, won’t redownload on each rerun)

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

st.set_page_config(
    page_title="iOncoSight",
    layout="wide",
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    background: #fff !important;
    color: #4e342e;
    min-height: 100vh;
    }

    /* Global text colors */
    /* Make ALL headings white; deeper shadow for contrast */
    /* Black headings! */
    h1, h2, h3, h4, h5, h6 {
    color: #111 !important;
    font-weight: 900;
    letter-spacing: 1px;
    text-shadow: none;
    }


    p, div, span, label {
        color: #4e342e !important; /* dark brownish for body text */
    }

    /* Sidebar styling */
    .sidebar .element-container {
        background: rgba(255, 245, 251, 0.95);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        color: #b71c1c !important;
        font-weight: 600;
    }

    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: #b71c1c !important;
    }

    .sidebar p, .sidebar div {
        color: #4e342e !important;
    }

    /* Main header white background, black text, red border */
    .main-header {
    background: #fff !important;
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    color: #111 !important;
    border: 2px solid #B71C1C;
    box-shadow: 0 8px 30px rgba(229,57,53,0.07);
    position: relative;
    }
    
    .header-flex {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.8rem;
    }
    .header-logo-img {
        height: 4.5rem;
        width: auto;
        margin-right: 0.6rem;
    }
    .header-title-group {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }

    .main-header h1 {
    font-size: 3.1rem;
    color: #111 !important;
    font-weight: 900;
    letter-spacing: 2px;
    text-shadow: none;
    }
    .main-header p {
    font-size: 1.1rem;
    font-style: italic;
    color: #B71C1C !important;
    opacity: 1;
    }

    /* Section backgrounds */
    .upload-section, .image-container {
        background: #ffe6ea; /* light pink */
        border-radius: 20px;
        border: 2px solid #e53935;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(229, 57, 53, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #b71c1c;
    }

    .upload-section:hover, .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 14px 40px rgba(229, 57, 53, 0.15);
    }

    /* Result Section */
    .result-section h1, .result-section h2, .result-section h3, .result-section h4 {
        color: #b71c1c !important;
    }

    .result-section p, .result-section div, .result-section span {
        color: #4e342e !important;
    }

    .result-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(230, 74, 111, 0.18);
    }

    /* Feedback Section */
    .feedback-section {
        background: linear-gradient(145deg, #fefce8, #fef3c7);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid #f59e0b;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.1);
        transition: transform 0.3s ease;
        color: #b71c1c;
    }

    .feedback-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(245, 158, 11, 0.15);
    }

    /* Status Messages */
    .status-success {
        background: linear-gradient(135deg, #fadadf, #ffe6ea);
        color: #b71c1c !important;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #e53935;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(229, 57, 53, 0.18);
        animation: slideIn 0.5s ease-out;
    }

    .status-error {
        background: linear-gradient(135deg, #fecaca, #fca5a5);
        color: #7f1d1d !important;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #dc2626;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.2);
    }

    .status-processing {
        background: linear-gradient(135deg, #ffe6ea, #ffd1dc);
        color: #b71c1c !important;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #e53935;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(229, 57, 53, 0.17);
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #fff0f5, #ffe6ea);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #f8bbd0;
        box-shadow: 0 4px 20px rgba(229,57,53,0.07);
        text-align: center;
        margin: 1rem 0;
        color: #b71c1c;
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e53935, #f06292, #f48fb1);
    }

    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 30px rgba(229,57,53,0.15);
    }

    .metric-card h3 {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #b71c1c !important;
    }
    .metric-card p {+
        color: #4e342e !important;
        font-weight: 500;
        margin: 0.5rem 0;
    }

    .metric-card strong {
        color: #b71c1c !important;
    }

    /* Buttons */
    .stButton > button[data-baseweb="button"] {
    background: #ff7f7f !important; /* light red */
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: background 0.3s ease, box-shadow 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(255, 127, 127, 0.4) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-size: 0.9rem !important;
    }

    .stButton > button[data-baseweb="button"]:hover {
    background: color: white !important;
    box-shadow: 0 6px 16px rgba(255, 76, 76, 0.6) !important;
    transform: translateY(-2px);
    }

    .stButton > button[data-baseweb="button"]:disabled {
    background: #ffcaca !important;
    color: color: white !important;
    cursor: not-allowed;
    box-shadow: none !important;
    transform: none !important;
    }
    /* Selectbox option text color (General Comments, etc.) */
    [data-baseweb="select"] div[role="listbox"] div,
    [data-baseweb="select"] > div > div > div {
    color: white !important;
    }

    /* Buttons text color to white (Submit Feedback, Clear Form) */
    .stButton > button {
    color: white !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #e53935, #f06292);
        border-radius: 10px !important;
    }

    /* File Uploader */
    .stFileUploader > div {
        border: 3px dashed #e53935 !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        background: linear-gradient(145deg, #fff0f5, #ffffff);
        transition: all 0.3s ease;
        color: white !important;
    }

    .stFileUploader > div:hover {
        border-color: #ad1457 !important;
        background: linear-gradient(145deg, #ffe6ea, #fff);
    }

    .stFileUploader label, 
    .stFileUploader > div > div > p {
    color: white !important;
    }

    /* Text inputs and areas */
    .stTextArea > div > div > textarea {
        border: 2px solid #f8bbd0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        color: #b71c1c !important;
        background: white !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #e53935 !important;
        box-shadow: 0 0 0 3px rgba(229, 57, 53, 0.10) !important;
    }

    .stTextInput > div > div > input {
        color: #b71c1c !important;
        background: white !important;
    }

    .stTextInput label {
        color: #b71c1c !important;
    }

    /* Tables */
    .stTable, .stDataFrame, table, td, th {
        color: #b71c1c !important;
    }

    /* Info/Success/Warning/Error boxes */
    .stAlert {
        color: #b71c1c !important;
    }

    /* Tab content */
    .stTabs [data-baseweb="tab-list"] {
        color: #b71c1c !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #b71c1c !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #b71c1c !important;
    }

    .streamlit-expanderContent {
        color: #4e342e !important;
    }

    /* Footer */
    .footer {
        background: linear-gradient(135deg, #d81b60, #ad1457);
        color: white !important;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 -10px 30px rgba(229,57,53,0.07);
    }

    .footer p, .footer h1, .footer h2, .footer h3, .footer h4, .footer span {
        color: white !important;
    }

    /* Loading Spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #e53935;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .upload-section, .result-section, .feedback-section {
            padding: 1.5rem;
        }
        .metric-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Your remaining Python Streamlit app code follows here unchanged...
# (Include the rest of your Streamlit app Python code logic here.)

# Enhanced session state management
class SessionManager:
    @staticmethod
    def initialize():
        defaults = {
            'processed_result': None,
            'report_data': None,
            'uploaded_file': None,
            'processing': False,
            'show_report': False,
            'feedback_history': [],
            'session_id': str(uuid.uuid4()),
            'processed_count': 0,
            'last_processing_time': None
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def update_stats():
        st.session_state.processed_count += 1
        st.session_state.last_processing_time = datetime.now()


# Enhanced AI Analysis Engine
class AIAnalysisEngine:
    @staticmethod
    def generate_realistic_report(Patient_ID, Predicted_class_ML,  impression, max_confidence_ML, Risk_level,
                                  processing_time):
        import random

        findings_options = [
            "No acute cardiopulmonary abnormality detected. Lung fields are clear with normal cardiac silhouette.",
            "Mild linear opacity in the right lower lobe, likely representing minor atelectasis. Heart size within normal limits.",
            "Clear lung fields bilaterally. Normal cardiac silhouette and mediastinal contours.",
            "Subtle increased opacity in left mid-zone, clinical correlation recommended. Otherwise unremarkable.",
            "Normal chest radiograph with clear lung fields and appropriate cardiac size.",
            "No evidence of pneumothorax or pleural effusion. Cardiac and mediastinal structures appear normal.",
            "Lungs are well-expanded and clear. No focal consolidation or pneumothorax identified."
        ]

        impression_options = [
            "Normal chest radiograph.",
            "No acute cardiopulmonary process.",
            "Essentially normal study.",
            "No significant abnormalities detected.",
            "Clear chest X-ray examination.",
            "Normal appearing chest radiograph.",
            "No acute findings identified."
        ]

        # confidence_score = round(random.uniform(94.2, 99.8), 1)
        # processing_time = round(random.uniform(1.8, 4.2), 1)
        Risk_level='High'
        if impression=='Non-Lung Cancer':# or impression=='':
            Risk_level = 'Low'
        max_confidence_ML=y1 = round(max_confidence_ML)
        return {
            'patient_id': f'PT-{Patient_ID}',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'findings': Predicted_class_ML,
            'impression': impression,
            'confidence': f'{max_confidence_ML}',
            'processing_time': f'{processing_time} seconds',
            'ai_model': 'iOncoSight AI v1.0',
            'risk_score': Risk_level,
            'recommendations': 'Continue routine monitoring as clinically indicated.'
        }


# Enhanced DICOM and Image Processing
class ImageProcessor:
    @staticmethod
    def load_image(uploaded_file):
        """Load and process both DICOM and standard image files"""
        try:
            # Check if it's a DICOM file
            if uploaded_file.name.lower().endswith('.dcm'):# or uploaded_file.type == 'application/octet-stream':
                return ImageProcessor.load_dicom_image(uploaded_file)
            else:
                # Handle standard image formats
                return ImageProcessor.load_standard_image(uploaded_file)

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            return None, None

    @staticmethod
    def load_dicom_image(uploaded_file):
        """Load and process DICOM files"""
        if not DICOM_AVAILABLE:
            st.error("❌ DICOM support not available. Please install pydicom: pip install pydicom")
            return None, None

        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Read DICOM file
            dicom_data = pydicom.dcmread(uploaded_file)
            # Extract pixel array
            pixel_array = dicom_data.pixel_array

            # Apply VOI LUT if available
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                image = apply_voi_lut(pixel_array, dicom_data)
            else:
                image = pixel_array

            # Handle photometric interpretation
            if hasattr(dicom_data, 'PhotometricInterpretation'):
                if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    image = np.amax(image) - image
                if dicom_data.PhotometricInterpretation == "MONOCHROME2":
                    image = np.amax(image) - image

            # Normalize to 0-255 range
            image = image - np.min(image)
            if np.max(image) > 0:
                image = (image / np.max(image) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Convert to RGB if grayscale
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            return pil_image, dicom_data

        except Exception as e:
            st.error(f"❌ Error processing DICOM file: {str(e)}")
            return None, None

    @staticmethod
    def load_standard_image(uploaded_file):
        """Load standard image formats"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Open with PIL
            image = Image.open(uploaded_file)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image, None

        except Exception as e:
            st.error(f"❌ Error loading standard image: {str(e)}")
            return None, None

    @staticmethod
    def get_image_info(uploaded_file, dicom_data=None):
        """Extract image information"""
        info = {}

        if dicom_data is not None:
            # DICOM specific information
            info.update({
                'modality': getattr(dicom_data, 'Modality', 'Unknown'),
                'body_part': getattr(dicom_data, 'BodyPartExamined', 'Unknown'),
                'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
                'institution': getattr(dicom_data, 'InstitutionName', 'Unknown'),
                'manufacturer': getattr(dicom_data, 'Manufacturer', 'Unknown'),
                'rows': getattr(dicom_data, 'Rows', 'Unknown'),
                'columns': getattr(dicom_data, 'Columns', 'Unknown'),
                'pixel_spacing': getattr(dicom_data, 'PixelSpacing', 'Unknown')
            })
        else:
            # Standard image information
            try:
                uploaded_file.seek(0)
                temp_image = Image.open(uploaded_file)
                info.update({
                    'dimensions': f"{temp_image.size[0]}x{temp_image.size[1]}",
                    'mode': temp_image.mode,
                    'format': temp_image.format
                })
            except:
                pass

        return info


# Initialize session manager
SessionManager.initialize()

# Enhanced main header with animations

import streamlit as st
import base64
import os

# Function to encode image as base64
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your logo (adjust if needed)
import os

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print('current_dir', flush=True)
# Build the logo path relative to current directory
logo_path = os.path.join(current_dir, "LOGO.jpg")
#logo_path = "./LOGO.jpg"

if os.path.exists(logo_path):
    img_base64 = get_base64_image(logo_path)

    st.markdown(
        f"""
        <style>
            .main-header {{
                display: flex;
                align-items: center;
                justify-content: flex-start;
                padding: 10px;
                flex-wrap: wrap; /* makes it responsive */
            }}
            .header-flex {{
                display: flex;
                align-items: center;
                gap: 25px; /* spacing between logo and text */
                flex-wrap: wrap;
            }}
            .header-logo-img {{
                height: 200px;          /* controls height */
                width: 400px;          /* increase initial width */
                object-fit: contain;   /* keeps aspect ratio */
                flex-shrink: 0;        /* prevents shrinking */
                max-width: 100%;       /* ensures responsiveness */
            }}
            .header-title-group {{
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                line-height: 1.2; /* reduce line spacing */
            }}
            .header-title-group h1 {{
                font-size: 40px; /* title font size */
                margin: 0;
            }}
            .header-title-group p {{
                font-size: 30px; /* subtitle font size */
                margin: 0;
                color: gray;  
            }}
        </style>

        <div class="main-header">
            <div class="header-flex">
                <img src="data:image/png;base64,{img_base64}" class="header-logo-img" alt="Diagno Intelligent Systems Logo" />
                <div class="header-title-group">
                    <h1>🫁 iOncoSight</h1>
                    <p><b><i>AI vision that spots lung cancer in every chest X-ray</i></b></p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error(f"❌ Logo not found at {logo_path}")

st.markdown("""
<style>
    .how-title {
        color: #e53935; /* Bright red */
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-size: 2.75rem;
        font-weight: 700;
        margin-bottom: 2.5rem;
        user-select: none;
    }
    .how-card {
        background: #ffe6ea; /* Light pink background */
        border-radius: 1.5rem;
        box-shadow: 0 4px 20px rgba(25, 118, 210, 0.12);
        padding: 2.5rem 2rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: default;
        user-select: none;
    }
    .how-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 36px rgba(25, 118, 210, 0.25);
    }
    .how-icon {
        display: block;
        margin: 0 auto 1.75rem auto;
        width: 4.5rem;
        height: 4.5rem;
        color: #1976d2; /* Medical blue */
        transition: color 0.3s ease;
    }
    .how-card:hover .how-icon {
        color: #0d47a1; /* Darker blue on hover */
    }
    .how-heading {
        color: #1976d2; /* Medical blue */
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .how-desc {
        color: #1565c0; /* Slightly darker blue for text */
        font-size: 1.05rem;
        line-height: 1.4;
        max-width: 280px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.how-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 2.5rem;
}
.how-title {
    color: #1976d2;
    font-family: 'Inter', sans-serif;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 2.2rem;
    user-select: none;
}
.timeline {
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 2rem;
    width: 100%;
}
.timeline-step {
    position: relative;
    background: #fff;
    border: 2px solid #e53935;
    border-radius: 1.25rem;
    box-shadow: 0 4px 16px rgba(25, 118, 210, 0.10);
    padding: 2rem 1.2rem 1.5rem 1.2rem;
    text-align: center;
    min-width: 220px;
    max-width: 270px;
    transition: box-shadow 0.3s, border-color 0.3s;
}
.timeline-step:hover {
    border-color: #0d47a1;
    box-shadow: 0 8px 28px rgba(25, 118, 210, 0.18);
}
.step-badge {
    position: absolute;
    top: -1.2rem;
    left: 50%;
    transform: translateX(-50%);
    background: #1976d2;
    color: #fff;
    font-weight: bold;
    font-size: 1.1rem;
    border-radius: 50%;
    width: 2.2rem;
    height: 2.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(25, 118, 210, 0.12);
    border: 3px solid #fff;
}
.timeline-dot {
    width: 0.7rem;
    height: 0.7rem;
    background: #1976d2;
    border-radius: 50%;
    position: absolute;
    left: 50%;
    top: 0;
    transform: translate(-50%, -50%);
    z-index: 2;
}
.timeline-line {
    width: 4px;
    height: 100%;
    background: linear-gradient(135deg, #ffe6ea 0%, #f8bbd0 100%);
    position: absolute;
    left: 50%;
    top: 0.7rem;
    transform: translateX(-50%);
    z-index: 1;
}
.timeline-step .how-icon {
    margin: 2.2rem auto 1.1rem auto;
    width: 2.8rem;
    height: 2.8rem;
    color: #1976d2;
    display: block;
}
.timeline-step .how-heading {
    color: #1976d2;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.timeline-step .how-desc {
    color: #1565c0;
    font-size: 1rem;
    line-height: 1.5;
}
@media (max-width: 950px) {
    .timeline { flex-direction: column; align-items: center; gap: 3.5rem;}
    .timeline-step { min-width: 0; width: 90vw; max-width: 340px;}
}
</style>
<div class="how-section">
    <div class="how-title">How It Works</div>
    <div class="timeline">
        <div class="timeline-step">
            <div class="step-badge">1</div>
            <svg class="how-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 64 64">
                <rect x="12" y="16" width="40" height="32" rx="6" stroke="currentColor" stroke-width="3"/>
                <path d="M32 24v16M24 32h16" stroke="currentColor" stroke-width="3"/>
            </svg>
            <div class="how-heading">Upload</div>
            <div class="how-desc">Chest X-ray PA view (DICOM)</div>
        </div>
        <div class="timeline-step">
            <div class="step-badge">2</div>
            <svg class="how-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 64 64">
                <circle cx="32" cy="32" r="24" stroke="currentColor" stroke-width="3"/>
                <path d="M32 16v16l12 6" stroke="currentColor" stroke-width="3"/>
            </svg>
            <div class="how-heading">Analyze</div>
            <div class="how-desc">Automated AI powered analysis</div>
            </div>
        <div class="timeline-step">
            <div class="step-badge">3</div>
            <svg class="how-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 64 64">
                <rect x="16" y="16" width="32" height="32" rx="6" stroke="currentColor" stroke-width="3"/>
                <path d="M24 32h16M32 24v16" stroke="currentColor" stroke-width="3"/>
            </svg>
            <div class="how-heading">Report</div>
            <div class="how-desc">AI generated diagnostic report</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

if "processing" not in st.session_state:
    st.session_state.processing = False
if "processed_result" not in st.session_state:
    st.session_state.processed_result = None
if "completed" not in st.session_state:
    st.session_state.completed = False

with col1:
    st.subheader("📤 Upload X-ray Image")

    uploaded_file = st.file_uploader(
        "Choose a DICOM file",
        type=['dcm', 'dicom', 'png', 'jpg', 'jpeg', 'tif', 'bmp'],
        help="📁 Supported formats: DICOM (.dcm,.dicom)",
        disabled=st.session_state.completed or st.session_state.processing
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        try:
            image, dicom_data = ImageProcessor.load_image(uploaded_file)
            if image:
                st.image(image, caption="📷 Uploaded Medical Image", width="stretch")
            else:
                st.error("❌ Failed to load the uploaded image.")
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

    if st.button("🔍 Analyze Image",
                 disabled=(uploaded_file is None) or st.session_state.completed,
                 use_container_width=True):
        st.session_state.processing = True
        st.rerun()
with col2:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Results")

    if st.session_state.processing:
        # Enhanced processing animation with better UX
        st.markdown("""
            <div class="status-processing">
                <div class="loading-spinner"></div> AI Analysis in Progress...
            </div>
            """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text = st.empty()
        ##############################################################################
        import os

        # os.chdir(r'F:/project')
        # Get directory where current script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Change working directory to that path
        os.chdir(current_dir)
        folder_path = './output_YOLOV11'  # Change this to your target folder
        folder_path_full = './output_YOLOV11/V11_SEG_PRED.png'
        # Delete all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path_full)#folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}", flush=True)
        ######
        import random
        # import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        import os
        import io
        # import cv2
        from PIL import Image

        from pydicom.uid import ExplicitVRLittleEndian
        import time

        start_time = time.time()

        # dicom = pydicom.dcmread(BytesIO(uploaded_file.read()))
        # pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
        image, dicom_data = ImageProcessor.load_image(uploaded_file)
        pixel_array = np.array(image)
        img = pixel_array
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        from PIL import Image, ImageDraw, ImageFont

        TARGET_SIZE = 300 #1024

        # Convert NumPy array to PIL Image
        if len(img.shape) == 2:  # grayscale
            img_pil = Image.fromarray(img).convert("RGB")
        elif img.shape[2] == 3:
            img_pil = Image.fromarray(img)
        elif img.shape[2] == 4:  # RGBA
            img_pil = Image.fromarray(img[:, :, :3])
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")


        # Resize
        img_resized = img_pil.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # Get original size
        w0, h0 = img_pil.size

        # Save as PNG
        output_path = os.path.join("./images/input.png")
        img_resized.save(output_path)
        import DL_model_ENB3andIncV3_code_and_seg_main
        from DL_model_ENB3andIncV3_code_and_seg_main import full_code

        imp_result,max_confidence_ML = full_code(output_path, eff_model, inc_model,rf_chi2_ens,xgb_chi2_ens,rf_mi_ens,ens_scaler_rf_chi2,ens_scaler_xgb_chi2,ens_scaler_rf_mi,
                                                st_ens_LC_NR,sel_ens_M1,sel_ens_M2,sel_ens_M3,scaled_ens_M1,scaled_ens_M2,scaled_ens_M3,ens_MCN,yolov11)

        print('final_impression', imp_result, flush=True)
        #print('output image path :', imp_image_out)
        imp=imp_result
        max_confidence_ML=max_confidence_ML
        Risk_level =[]# Risk_level  # rish level
        Predicted_class_ML=impression = imp  ### impression text
        """ Patient information"""
        Patient_ID = uploaded_file.name
        Patient_Name = "NA"
        Patient_Age = "NA"
        Patient_Sex = "NA"

        end_time = time.time()
        print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds", flush=True)
        processing_time = f"{end_time - start_time:.2f} "
        ##############################################################################

        # Simulate processing completion with enhanced results
        st.session_state.processing = False
        st.session_state.completed = True
        st.session_state.processed_result = "result.jpg"
        st.session_state.report_data = AIAnalysisEngine.generate_realistic_report(
            Patient_ID, Predicted_class_ML, impression, max_confidence_ML,
            Risk_level, processing_time
        )
        SessionManager.update_stats()
        st.rerun()

    elif st.session_state.processed_result:
        st.success("✅ Analysis completed successfully!")

        # Display result image
        st.image("result.jpg", caption="🤖 AI Analysis Visualization", width="stretch")

        # Display metrics
        if st.session_state.report_data:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    "🎯 Confidence",
                    st.session_state.report_data['confidence'],
                    delta="High" if float(
                        st.session_state.report_data['confidence'].rstrip('%')
                    ) > 95 else "Good"
                )
            with col_m2:
                st.metric(
                    "⚡ Processing Time",
                    st.session_state.report_data['processing_time'],
                    delta="Fast"
                )

        # Action buttons
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            if st.button("📋 View Report", use_container_width=True):
                st.session_state.show_report = True
        with col_b:
            if st.button("💾 Download", use_container_width=True):
                report_text = f"""
iOncoSight DIAGNOSTIC REPORT
============================

PATIENT INFORMATION
Patient ID: {st.session_state.report_data['patient_id']}
Analysis Date: {st.session_state.report_data['date']}
AI Model: {st.session_state.report_data['ai_model']}

ANALYSIS RESULTS
Finding : {st.session_state.report_data['findings']}
Confidence Score (%): {st.session_state.report_data['confidence']}
Processing Time: {st.session_state.report_data['processing_time']}  

#CLINICAL IMPRESSION  
Clinical Impression: {st.session_state.report_data['impression']}

IMPORTANT DISCLAIMERS
- This AI analysis is intended to assist healthcare professionals
- Clinical correlation and professional medical interpretation required
- Not intended as a substitute for professional medical diagnosis
- For research and educational purposes

Generated by iOncoSight v1.0, Diagno Intelligent Systems Private Limited
Report ID: {st.session_state.report_data['patient_id']}-{datetime.now().strftime('%H%M%S')}
                    """
                st.download_button(
                    label="📄 Download Report",
                    data=report_text,
                    file_name=f"iOncoSight_report_{st.session_state.report_data['patient_id']}.txt",
                    mime="text/plain",
                     width="stretch"
                )

        with col_c:
            if st.button("🔄 New Analysis",  width="stretch"):
                st.session_state.uploaded_file = None
                st.session_state.processed_result = None
                st.session_state.report_data = None
                st.session_state.show_report = False
                st.session_state.completed = False
                st.session_state.clear() 
                for key in ["uploaded_file", "processed_result", "report_data", 
                        "show_report", "completed"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            else:
                st.info("🎯 Upload an image and click 'Analyze Image' to begin new AI analysis")

        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced detailed report section
if st.session_state.show_report and st.session_state.report_data:
    st.markdown("---")
    # Enhanced metrics cards
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown(f"""
            <div class="metric-card" >
                <h3>👤 Patient Information</h3>
                <p><strong>ID:</strong> {st.session_state.report_data['patient_id']}</p>
                <p><strong>Date:</strong> {st.session_state.report_data['date']}</p>
                <p><strong>Model:</strong> {st.session_state.report_data['ai_model']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_r2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Analysis Metrics</h3>
                <p><strong>Findings:</strong> {st.session_state.report_data['findings']}</p>
                <p><strong>Confidence Score (%):</strong> {st.session_state.report_data['confidence']}</p>  
                <p><strong>Processing Time:</strong> {st.session_state.report_data['processing_time']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Enhanced findings and impression
    # col_find1 = st.columns(1)

    # with col_find1:
    st.markdown("### 🔍 Clinical Impression")
    st.success(st.session_state.report_data['impression'])

# Enhanced Doctor's Feedback Section
st.markdown("---")
# st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
st.subheader("👨‍⚕️ Doctor's Feedback")

# Feedback type selection
feedback_type = st.selectbox(
    "🏷️ Feedback Type",
    ["General Comments", "Diagnostic Correction", "Technical Issue", "Improvement Suggestion",
     "Accuracy Assessment"]
)

# Rating system
col_rating1, col_rating2 = st.columns(2)
with col_rating1:
    accuracy_rating = st.slider("🎯 Accuracy Rating", 1, 5, 4, help="Rate the AI analysis accuracy")
with col_rating2:
    usefulness_rating = st.slider("💡 Usefulness Rating", 1, 5, 4, help="Rate how useful this analysis is")

# Enhanced feedback text area
feedback_text = st.text_area(
    "📋 Detailed Feedback:",
    placeholder="Enter your professional feedback, corrections, or suggestions here...\n\nConsider including:\n• Diagnostic accuracy assessment\n• Missing findings\n• Technical improvements\n• Clinical correlation notes",
    height=120,
    help="Your feedback helps improve our AI model"
)

# Enhanced submit section
col_submit1, col_submit2, col_submit3 = st.columns([1, 1, 1])

with col_submit1:
    if st.button("📤 Submit Feedback",  width="stretch"):
        if feedback_text.strip():
            # Save feedback to session history
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': feedback_type,
                'accuracy_rating': accuracy_rating,
                'usefulness_rating': usefulness_rating,
                'text': feedback_text,
                'patient_id': st.session_state.report_data[
                    'patient_id'] if st.session_state.report_data else 'N/A'
            }

        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
            st.session_state.feedback_history.append(feedback_entry)

            st.success("✅ Feedback submitted successfully! Thank you for helping improve our AI model.")

            # Show submission details
            st.info(
                f"📊 Submission ID: {feedback_entry['timestamp'][-8:]}\n🎯 Accuracy: {accuracy_rating}/5 ⭐\n💡 Usefulness: {usefulness_rating}/5 ⭐")
        else:
            st.warning("⚠️ Please enter your feedback before submitting.")

with col_submit2:
    if st.button("🔄 Clear Form",  width="stretch"):
        st.rerun()

with col_submit3:
    if len(st.session_state.feedback_history) > 0:
        st.metric("📈 Total", len(st.session_state.feedback_history))
    else:
        st.info("📤 Provide professional feedback on the AI results before submit.")

# st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
            <style>
            .tiny-footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background: #e3f2fd;
                color: #1976d2;
                text-align: center;
                padding: 0.18rem 0 0.18rem 0;
                font-family: 'Inter', sans-serif;
                font-size: 0.88rem;
                z-index: 100;
                box-shadow: 0 -1px 8px rgba(25, 118, 210, 0.06);
                line-height: 1.5;
            }
            </style>
            <div class="tiny-footer">
                🫁 iOncoSight • © 2025 @Diagno Intelligent Systems Private Limited
            </div>
 """, unsafe_allow_html=True)
