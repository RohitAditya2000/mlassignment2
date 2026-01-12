"""
Bank Marketing Prediction - Streamlit Application
Author: ML Assignment 2
Description: Interactive web app for predicting term deposit subscriptions using 6 ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score, 
                             precision_score, recall_score, f1_score, 
                             matthews_corrcoef, classification_report)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main .block-container {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e3a8a !important;
        font-weight: 600;
        font-size: 1.8rem !important;
    }
    
    h3 {
        color: #4338ca !important;
        font-weight: 600;
        font-size: 1.3rem !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed #667eea !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #1e3a8a !important;
        font-weight: 600 !important;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        border-radius: 12px !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #1e3a8a !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Model names and file paths
MODEL_NAMES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

MODELS_DIR = 'model/trained_models'

# Load preprocessing objects
@st.cache_resource
def load_preprocessing():
    """Load scaler and label encoders"""
    try:
        scaler = joblib.load(f'{MODELS_DIR}/scaler.pkl')
        label_encoders = joblib.load(f'{MODELS_DIR}/label_encoders.pkl')
        feature_names = joblib.load(f'{MODELS_DIR}/feature_names.pkl')
        return scaler, label_encoders, feature_names
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def load_model(model_name):
    """Load a trained model"""
    try:
        model_path = f'{MODELS_DIR}/{MODEL_NAMES[model_name]}'
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

def preprocess_data(df, label_encoders, feature_names):
    """Preprocess the input data"""
    df_copy = df.copy()
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df_copy.columns:
            df_copy[col] = encoder.transform(df_copy[col])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in df_copy.columns:
            df_copy[feature] = 0
    
    # Select only the features used in training
    df_copy = df_copy[feature_names]
    
    return df_copy

def create_confusion_matrix_plot(cm, model_name):
    """Create an interactive confusion matrix using Plotly"""
    
    # Normalize for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1f}%)",
                    x=j,
                    y=i,
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black", 
                             size=16)
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No', 'Predicted: Yes'],
        y=['Actual: No', 'Actual: Yes'],
        colorscale='RdPu',
        showscale=True,
        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'Confusion Matrix - {model_name}', 
                  font=dict(size=20)),
        annotations=annotations,
        xaxis=dict(side='bottom'),
        width=500,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_metrics_chart(metrics_dict):
    """Create a radar chart for model metrics"""
    
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgb(102, 126, 234)', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                showline=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            )
        ),
        showlegend=False,
        title=dict(text='Model Performance Metrics', font=dict(size=20)),
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.markdown("## 🏦 Bank Marketing")
    st.markdown("### ML Prediction System")
    st.markdown("---")
    
    # Model selection
    st.markdown("### 🤖 Select Model")
    selected_model = st.selectbox(
        "Choose a model",
        list(MODEL_NAMES.keys()),
        help="Choose one of the 6 trained classification models",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Information
    st.markdown("### 📊 Dataset Info")
    st.info("""
    **Features:** 17  
    **Instances:** 45,211  
    **Target:** Term Deposit (Yes/No)  
    **Source:** UCI ML Repository
    """)
    
    st.markdown("---")
    
    # Sample data download
    if os.path.exists(f'{MODELS_DIR}/test_data.csv'):
        with open(f'{MODELS_DIR}/test_data.csv', 'rb') as f:
            st.download_button(
                label="📥 Download Sample Data",
                data=f,
                file_name="sample_test_data.csv",
                mime="text/csv",
                help="Download sample test data to try the app"
            )
    
    st.markdown("---")
    st.markdown("### 👨‍💻 About")
    st.markdown("""
    This app predicts whether a bank client will subscribe to a term deposit 
    based on marketing campaign data.
    
    **Models Available:**
    - Logistic Regression
    - Decision Tree
    - kNN
    - Naive Bayes
    - Random Forest
    - XGBoost
    """)

# Main content
st.title("🏦 Bank Marketing Prediction System")
st.markdown("### Predict Term Deposit Subscriptions with Machine Learning")

# Check if models are trained
scaler, label_encoders, feature_names = load_preprocessing()

if scaler is None:
    st.error("⚠️ Models not found! Please train the models first.")
    st.code("python model/train_models.py", language="bash")
    st.stop()

# Load model results
if os.path.exists(f'{MODELS_DIR}/model_results.csv'):
    results_df = pd.read_csv(f'{MODELS_DIR}/model_results.csv')
    
    st.markdown("---")
    st.markdown("## 📈 Model Performance Overview")
    
    # Display metrics in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    model_row = results_df[results_df['Model'] == selected_model].iloc[0]
    
    with col1:
        st.metric("Accuracy", f"{model_row['Accuracy']:.3f}")
    with col2:
        st.metric("AUC", f"{model_row['AUC']:.3f}")
    with col3:
        st.metric("Precision", f"{model_row['Precision']:.3f}")
    with col4:
        st.metric("Recall", f"{model_row['Recall']:.3f}")
    with col5:
        st.metric("F1 Score", f"{model_row['F1']:.3f}")
    with col6:
        st.metric("MCC", f"{model_row['MCC']:.3f}")
    
    # Show comparison table
    st.markdown("### 🔍 All Models Comparison")
    st.dataframe(
        results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], 
                                       color='lightgreen'),
        use_container_width=True,
        height=280
    )

st.markdown("---")
st.markdown("## 🎯 Make Predictions")

# File upload
uploaded_file = st.file_uploader(
    "Upload your test CSV file (must contain all 17 features)",
    type=['csv'],
    help="Upload a CSV file with the same structure as the training data"
)

if uploaded_file is not None:
    try:
        # Read the uploaded file - auto-detect delimiter (comma or semicolon)
        test_data = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        st.success(f"✅ File uploaded successfully! Found {len(test_data)} rows and {len(test_data.columns)} columns.")
        
        # Show data preview
        with st.expander("📋 View Uploaded Data Preview (first 10 rows)"):
            st.dataframe(test_data.head(10), use_container_width=True)
        
        # Check if target column exists
        has_target = 'y' in test_data.columns
        if has_target:
            y_true = test_data['y'].map({'no': 0, 'yes': 1}) if test_data['y'].dtype == 'object' else test_data['y']
            test_data_features = test_data.drop('y', axis=1)
        else:
            test_data_features = test_data
            st.warning("⚠️ No target column 'y' found. Predictions will be made without evaluation metrics.")
        
        # Preprocess
        test_data_processed = preprocess_data(test_data_features, label_encoders, feature_names)
        
        # Load selected model
        model = load_model(selected_model)
        
        if model is not None:
            # Make predictions
            if selected_model in ['Logistic Regression', 'kNN', 'Naive Bayes']:
                test_data_scaled = scaler.transform(test_data_processed)
                predictions = model.predict(test_data_scaled)
                predictions_proba = model.predict_proba(test_data_scaled)[:, 1]
            else:
                predictions = model.predict(test_data_processed)
                predictions_proba = model.predict_proba(test_data_processed)[:, 1]
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎊 Prediction Results")
            
            # Prediction distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Prediction Distribution")
                pred_counts = pd.Series(predictions).value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['No Subscription', 'Subscription'],
                    values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
                    marker=dict(colors=['#667eea', '#764ba2']),
                    textinfo='label+percent+value',
                    hole=0.4
                )])
                fig_pie.update_layout(
                    title="Predicted Outcomes",
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Prediction Confidence")
                fig_hist = go.Figure(data=[go.Histogram(
                    x=predictions_proba,
                    nbinsx=30,
                    marker=dict(
                        color=predictions_proba,
                        colorscale='Purples',
                        showscale=True,
                        colorbar=dict(title="Probability")
                    )
                )])
                fig_hist.update_layout(
                    title="Distribution of Prediction Probabilities",
                    xaxis_title="Probability of Subscription",
                    yaxis_title="Count",
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # If ground truth is available, show evaluation metrics
            if has_target:
                st.markdown("---")
                st.markdown("## 🎯 Model Evaluation")
                
                # Calculate metrics
                cm = confusion_matrix(y_true, predictions)
                accuracy = accuracy_score(y_true, predictions)
                auc = roc_auc_score(y_true, predictions_proba)
                precision = precision_score(y_true, predictions, zero_division=0)
                recall = recall_score(y_true, predictions, zero_division=0)
                f1 = f1_score(y_true, predictions, zero_division=0)
                mcc = matthews_corrcoef(y_true, predictions)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Confusion matrix
                    fig_cm = create_confusion_matrix_plot(cm, selected_model)
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    # Metrics radar chart
                    metrics_dict = {
                        'Accuracy': accuracy,
                        'AUC': auc,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'MCC': (mcc + 1) / 2  # Normalize MCC to 0-1 range for visualization
                    }
                    fig_radar = create_metrics_chart(metrics_dict)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Detailed classification report
                with st.expander("📑 Detailed Classification Report"):
                    report = classification_report(y_true, predictions, 
                                                  target_names=['No Subscription', 'Subscription'],
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'),
                               use_container_width=True)
            
            # Download predictions
            st.markdown("---")
            predictions_df = test_data_features.copy()
            predictions_df['Predicted_Subscription'] = ['Yes' if p == 1 else 'No' for p in predictions]
            predictions_df['Prediction_Probability'] = predictions_proba
            if has_target:
                predictions_df['Actual_Subscription'] = ['Yes' if y == 1 else 'No' for y in y_true]
            
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )
            
        else:
            st.error(f"❌ Could not load {selected_model} model. Please ensure it's trained.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format with all required features.")

else:
    st.info("👆 Please upload a CSV file to begin making predictions")
    
    # Show example of expected format
    with st.expander("ℹ️ Expected CSV Format"):
        st.markdown("""
        Your CSV should contain the following columns:
        - age
        - job
        - marital
        - education
        - default
        - balance
        - housing
        - loan
        - contact
        - day
        - month
        - duration
        - campaign
        - pdays
        - previous
        - poutcome
        - y (optional - for evaluation)
        """)
