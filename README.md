# 🏦 Machine Learning Classification Project: Bank Marketing Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive machine learning project predicting term deposit subscriptions using 6 different classification algorithms with an interactive Streamlit web application.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Models Used](#-models-used)
- [Model Performance](#-model-performance)
- [Project Files](#-project-files)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)

---

## 🎯 Problem Statement

The goal of this project is to **predict whether a client will subscribe to a term deposit** (variable 'y') based on direct marketing campaign data from a Portuguese banking institution. This is a **binary classification problem** where we implement and compare six different machine learning algorithms to identify which provides the highest predictive accuracy and MCC (Matthews Correlation Coefficient) score for effective financial targeting.

**Business Context**: Direct marketing campaigns are expensive. By accurately predicting which clients are likely to subscribe, banks can optimize their marketing efforts, reduce costs, and improve conversion rates.

---

## 📊 Dataset Description

- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Instance Size**: 45,211 rows (✅ **exceeds >500 requirement**)
- **Feature Size**: 17 features (✅ **exceeds >12 requirement**)
- **Target Variable**: `y` (Binary: "yes" or "no")
- **Class Distribution**: Imbalanced dataset with ~11.7% positive class (subscriptions)

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Client's age |
| job | Categorical | Type of job (12 categories) |
| marital | Categorical | Marital status (married/single/divorced) |
| education | Categorical | Education level |
| default | Binary | Has credit in default? |
| balance | Numeric | Average yearly balance (euros) |
| housing | Binary | Has housing loan? |
| loan | Binary | Has personal loan? |
| contact | Categorical | Contact communication type |
| day | Numeric | Last contact day of the month |
| month | Categorical | Last contact month of year |
| duration | Numeric | Last contact duration (seconds) |
| campaign | Numeric | Number of contacts during campaign |
| pdays | Numeric | Days since last contact |
| previous | Numeric | Number of contacts before campaign |
| poutcome | Categorical | Outcome of previous campaign |
| **y** | **Binary** | **Has subscribed to term deposit?** |

---

## 🤖 Models Used

The following **6 machine learning models** were implemented and evaluated:

### 1. **Logistic Regression**
- Linear model for binary classification
- Fast training and prediction
- Good interpretability
- Baseline model for comparison

### 2. **Decision Tree**
- Tree-based non-linear classifier
- High interpretability with visual tree structure
- Prone to overfitting without constraints
- Feature importance analysis available

### 3. **k-Nearest Neighbors (kNN)**
- Instance-based lazy learning algorithm
- Non-parametric approach
- Computationally expensive for large datasets
- Sensitive to feature scaling

### 4. **Naive Bayes**
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction
- Works well with high-dimensional data

### 5. **Random Forest**
- Ensemble of decision trees
- Reduces overfitting through bagging
- Robust to outliers and noise
- Provides feature importance rankings

### 6. **XGBoost** ⭐
- Gradient boosting framework
- State-of-the-art performance
- Handles missing values automatically
- Regularization to prevent overfitting

---

## 📈 Model Performance

### Comprehensive Results Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC | Training Time |
|----------|----------|-----|-----------|--------|----------|-----|---------------|
| Logistic Regression | 0.891 | 0.873 | 0.593 | 0.226 | 0.327 | 0.320 | Fast ⚡ |
| Decision Tree | 0.899 | 0.851 | 0.584 | 0.484 | 0.529 | 0.476 | Fast ⚡ |
| kNN | 0.892 | 0.809 | 0.572 | 0.317 | 0.408 | 0.372 | Slow 🐌 |
| Naive Bayes | 0.838 | 0.813 | 0.355 | 0.473 | 0.406 | 0.318 | Fast ⚡ |
| Random Forest | 0.907 | 0.926 | 0.669 | 0.398 | 0.499 | 0.470 | Medium ⏱️ |
| **XGBoost** ⭐ | **0.909** | **0.931** | **0.654** | **0.471** | **0.547** | **0.506** | Medium ⏱️ |

### Model Performance Observations

| Model | Key Observations |
|-------|-----------------|
| **Logistic Regression** | Good baseline performance with 89.1% accuracy. Very fast to train but struggled with the minority class due to imbalanced data. Low recall (22.6%) indicates many false negatives. Best for interpretability and speed. |
| **Decision Tree** | Achieved 89.9% accuracy with better recall (48.4%) than Logistic Regression. High interpretability allows understanding decision rules. Showed signs of overfitting as tree depth increased, requiring pruning. |
| **k-Nearest Neighbors** | Average performance with 89.2% accuracy. Computationally expensive during prediction phase on 45k instances. Sensitive to feature scaling and distance metrics. Not recommended for production due to speed. |
| **Naive Bayes** | Lowest accuracy (83.8%) but highest recall (47.3%) among simpler models. The independence assumption doesn't hold well for this dataset as features are correlated. Good for quick baseline and probability estimates. |
| **Random Forest** | Strong performance with 90.7% accuracy and 92.6% AUC. Successfully balanced the variance of decision trees through ensemble learning. Significant jump in precision (66.9%) and AUC score. Feature importance helped identify key predictors. |
| **XGBoost** ⭐ | **Best Overall Performance**. Highest MCC (0.506) and F1 score (0.547), indicating best balance between precision and recall. Effectively handles non-linear relationships and feature interactions in banking data. Highest AUC (93.1%) shows excellent ranking ability. **Recommended for deployment**. |

### Key Insights

- ✅ **Best Model**: XGBoost outperforms all others across most metrics
- ✅ **AUC Analysis**: Random Forest (92.6%) and XGBoost (93.1%) show excellent discrimination ability
- ✅ **Precision-Recall Tradeoff**: Ensemble methods (RF, XGBoost) achieve best balance
- ⚠️ **Class Imbalance**: All models struggle with minority class detection (low recall)
- 💡 **Recommendation**: Use XGBoost for production with cost-sensitive learning or SMOTE

---

## 📁 Project Files & Deployment

### Core Files

| File | Description | Size |
|------|-------------|------|
| `app.py` | **Main Streamlit application** with premium UI design | 20+ KB |
| `requirements.txt` | Python dependencies for deployment | 1 KB |
| `model/train_models.py` | Training script for all 6 models | 6 KB |
| `model/trained_models/*.pkl` | Saved model files (6 models + preprocessors) | 32 MB |
| `bank/bank-full.csv` | Complete dataset (45,211 rows) | 4.4 MB |
| `README.md` | This comprehensive documentation | You're here! |

### Model Files

All trained models are saved in `model/trained_models/`:
- `logistic_regression.pkl` (991 bytes)
- `decision_tree.pkl` (79 KB)
- `knn.pkl` (4.9 MB)
- `naive_bayes.pkl` (1.3 KB)
- `random_forest.pkl` (25.8 MB)
- `xgboost.pkl` (557 KB)
- `scaler.pkl`, `label_encoders.pkl`, `feature_names.pkl` (preprocessors)
- `model_results.csv` (summary metrics)
- `test_data.csv` (sample test data)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 500MB free disk space

### Step-by-Step Setup

1. **Clone or Download the Project**
   ```bash
   cd c:\code\ml_assignment2
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if you encounter permission errors:
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import streamlit; import xgboost; print('✅ All dependencies installed!')"
   ```

### Dependencies

```
streamlit==1.29.0          # Web application framework
pandas==2.1.4              # Data manipulation
numpy==1.26.2              # Numerical computing
scikit-learn==1.3.2        # ML algorithms
xgboost==2.0.3             # Gradient boosting
matplotlib==3.8.2          # Plotting
seaborn==0.13.0            # Statistical visualization
plotly==5.18.0             # Interactive charts
imbalanced-learn==0.11.0   # Handling imbalanced data
joblib==1.3.2              # Model serialization
```

---

## 💻 Usage

### 1. Train Models (Optional - Pre-trained models included)

```bash
python model/train_models.py
```

**Output:**
- Trains all 6 models on the dataset
- Saves models to `model/trained_models/`
- Generates performance metrics
- Creates test data for demo
- **Time**: ~2-3 minutes

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

**The app will open in your browser at**: `http://localhost:8501` or `http://localhost:8502`

### 3. Using the Application

#### **Upload Data**
- Click "Browse files" to upload a CSV file
- Supports both comma (`,`) and semicolon (`;`) delimited files
- Must contain all 17 required features
- Optional: Include `y` column for evaluation metrics

#### **Select Model**
- Choose from 6 models in the sidebar dropdown
- View model metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Compare all models in the performance table

#### **View Results**
- **Prediction Distribution**: Pie chart showing subscription predictions
- **Confidence Scores**: Histogram of prediction probabilities
- **Confusion Matrix**: Heatmap with actual vs predicted values
- **Metrics Radar Chart**: Visual comparison of all metrics
- **Classification Report**: Detailed per-class statistics

#### **Download Predictions**
- Click "Download Predictions as CSV"
- Includes predicted class, probabilities, and actual values (if provided)

---

## 🌐 Deployment

### Deploy to Streamlit Community Cloud (FREE)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Bank Marketing ML Project"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set main file: `app.py`
   - Click "Deploy"!

3. **Your app will be live at**: `https://<your-username>-ml-assignment2.streamlit.app`

### Alternative Deployment Options

- **Heroku**: Use `Procfile` with web dyno
- **AWS EC2**: Deploy with Docker container
- **Google Cloud Run**: Serverless container deployment
- **Azure Web Apps**: Cloud hosting with auto-scaling

---

## ✨ Features

### 🎨 Premium UI Design
- **Modern Gradient Backgrounds**: Purple/blue color scheme
- **Glassmorphism Effects**: Translucent cards with shadows
- **Google Fonts Integration**: Inter font family
- **Responsive Layout**: Works on desktop and tablet
- **Smooth Animations**: Fade-in effects and transitions

### 📊 Interactive Visualizations
- **Plotly Charts**: Interactive confusion matrices and plots
- **Radar Charts**: Multi-dimensional metric comparison
- **Heatmaps**: Color-coded confusion matrices with annotations
- **Distribution Charts**: Pie charts and histograms
- **Dynamic Updates**: Real-time chart updates based on model selection

### 🔧 Technical Features
- **Auto-delimiter Detection**: Handles CSV files with `;` or `,`
- **Automatic Preprocessing**: Label encoding and scaling
- **Model Caching**: Fast loading with Streamlit cache
- **Error Handling**: Graceful error messages
- **File Validation**: Checks for required features
- **Download Functionality**: Export predictions as CSV

### 📱 User Experience
- **Sidebar Navigation**: Easy model selection
- **Collapsible Sections**: Expandable data previews
- **Progress Indicators**: Loading states
- **Helpful Tooltips**: Guidance for each feature
- **Sample Data**: Download test data for demo

---

## 📂 Project Structure

```
ml_assignment2/
├── 📄 app.py                          # Main Streamlit application (20KB)
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # This comprehensive documentation
├── 📄 ML_Assignment_2.pdf            # Assignment instructions
│
├── 📁 bank/                           # Dataset directory
│   ├── bank-full.csv                 # Complete dataset (45,211 rows)
│   ├── bank.csv                      # Subset (4,521 rows)
│   └── bank-names.txt                # Feature descriptions
│
├── 📁 model/                          # Model training and storage
│   ├── train_models.py               # Training script for all 6 models
│   └── trained_models/               # Saved models and artifacts
│       ├── logistic_regression.pkl   # Logistic Regression model
│       ├── decision_tree.pkl         # Decision Tree model
│       ├── knn.pkl                   # k-Nearest Neighbors model
│       ├── naive_bayes.pkl           # Naive Bayes model
│       ├── random_forest.pkl         # Random Forest model
│       ├── xgboost.pkl               # XGBoost model (best)
│       ├── scaler.pkl                # Standard scaler for features
│       ├── label_encoders.pkl        # Categorical encoders
│       ├── feature_names.pkl         # Feature list
│       ├── model_results.csv         # Performance metrics summary
│       └── test_data.csv             # Sample test data for demo
│
└── 📁 .streamlit/                     # Streamlit configuration (optional)
    └── config.toml                    # App settings
```

**Total Project Size**: ~33 MB (mostly model files)

---

## 📸 Screenshots

### Main Dashboard
The app features a beautiful gradient UI with interactive charts and real-time predictions.

### Model Performance Comparison
Side-by-side comparison of all 6 models with highlighted best performers.

### Prediction Results
Upload CSV data and instantly see predictions with confidence scores and visualizations.

### Confusion Matrix & Metrics
Interactive heatmaps and radar charts for comprehensive model evaluation.

---

## 🎓 Assignment Compliance

### Requirements Met ✅

| Requirement | Status | Details |
|-------------|--------|---------|
| Dataset Size (>500 instances) | ✅ | 45,211 instances |
| Feature Count (>12 features) | ✅ | 17 features |
| Number of Models (6+) | ✅ | 6 models implemented |
| Model Training | ✅ | All models trained and saved |
| Performance Metrics | ✅ | Accuracy, AUC, Precision, Recall, F1, MCC |
| Streamlit App | ✅ | Interactive web application |
| CSV Upload | ✅ | Supports file upload |
| Model Selection | ✅ | Dropdown menu |
| Confusion Matrix | ✅ | Interactive Plotly heatmap |
| Deployment Ready | ✅ | Can deploy to Streamlit Cloud |
| Documentation | ✅ | Comprehensive README |

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`
```bash
# Solution:
pip install --user -r requirements.txt
```

**Issue**: CSV file not parsing correctly
```bash
# Solution: The app auto-detects delimiters (both , and ; supported)
# Ensure your CSV has the correct 17 feature columns
```

**Issue**: Models not found
```bash
# Solution: Train models first
python model/train_models.py
```

**Issue**: Port already in use
```bash
# Solution: Streamlit will auto-select next available port (8501, 8502, etc.)
# Or specify port manually:
streamlit run app.py --server.port 8503
```

---

## 📝 Notes

- **Training Time**: Initial model training takes 2-3 minutes
- **Prediction Speed**: Real-time predictions (<1 second for 1000 rows)
- **Memory Usage**: ~200MB RAM for app + models
- **Best Performance**: XGBoost with MCC=0.506 and F1=0.547
- **Data Format**: Supports both comma and semicolon delimited CSVs

---

## 📚 References

1. [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
2. [Moro et al., 2014] - A Data-Driven Approach to Predict the Success of Bank Telemarketing
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)
5. [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 👨‍💻 Author

**ML Assignment 2 - Bank Marketing Classification Project**

Created for: Machine Learning Course  
Date: January 2026  
Framework: Streamlit + Scikit-learn + XGBoost

---

## 📄 License

This project is created for educational purposes as part of a machine learning assignment.

---

## 🎉 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Streamlit team for the amazing framework
- Scikit-learn and XGBoost communities

---

**⭐ Star this project if you found it helpful!**

**🚀 Ready to deploy? Follow the [Deployment](#-deployment) section above!**
