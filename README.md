# ANNClassification-Churn-Analysis

# Customer Churn Prediction Using Artificial Neural Networks (ANN) + Web Deployment

This project is a **complete machine learning pipeline** built to predict customer churn using Artificial Neural Networks. It includes everything from raw data ingestion and exploratory analysis to model training, evaluation, and **deployment using Streamlit**. The goal is to simulate a real-world machine learning solution for a SaaS/telecom/banking business use case.

---

## Executive Summary

**Business Context:**  
Customer churn represents a significant revenue leak for subscription-based businesses. Predicting churn enables targeted interventions to improve retention.

**Goal:**  
To build a scalable, interpretable, and deployable churn prediction model trained on structured customer data, and expose it through a user-friendly web application.

**Target Users:**  
- Business analysts looking for insights on customer behavior  
- Retention teams needing a real-time prediction tool  
- Hiring managers evaluating full-stack ML skills

---

## Dataset Overview

- **Source:** Simulated telco/banking customer dataset
- **File:** `Churn_Modelling.csv`
- **Samples:** ~10,000 rows
- **Target:** `Exited` (1 = churned, 0 = retained)
- **Type:** Tabular structured data

### Key Features

| Feature          | Description                             |
|------------------|-----------------------------------------|
| CreditScore      | Numerical value indicating credit worthiness |
| Geography        | Country of residence                    |
| Gender           | Male/Female                             |
| Age              | Customer age                            |
| Tenure           | Years of association with company       |
| Balance          | Account balance                         |
| NumOfProducts    | Number of financial products used       |
| HasCrCard        | Possesses credit card (0 or 1)          |
| IsActiveMember   | Active account flag                     |
| EstimatedSalary  | Predicted annual income                 |

---

## Tech Stack

| Layer               | Tools/Libraries                        |
|---------------------|----------------------------------------|
| Data Handling       | `Pandas`, `NumPy`                      |
| Visualization       | `Seaborn`, `Matplotlib`                |
| Preprocessing       | `scikit-learn`                         |
| Modeling            | `TensorFlow`, `Keras`                  |
| Model Evaluation    | `ConfusionMatrix`, `ROC-AUC`, `F1-Score` |
| Web Deployment      | `Streamlit`                            |
| Environment         | `Python 3.8+`, `Jupyter Notebook`      |
| Project Management  | `Git`, `requirements.txt`, `README.md` |

---

## Project Structure

```text

ANNClassification-Churn-Analysis/
├── data/                 # Source data: Churn_Modelling.csv
├── notebooks/            # EDA & experimentation: churn_analysis.ipynb
├── scripts/              # Modular Python code
│   ├── preprocess.py     # Data cleaning & encoding
│   └── model.py          # ANN training + evaluation
├── models/               # Saved trained model: ann_model.h5
├── outputs/              # Plotted outputs: confusion_matrix.png, roc_curve.png
├── app.py                # Streamlit application for inference
├── requirements.txt      # Python dependencies
└── README.md             # Project overview (this file)

```

##  Setup Instructions  
*(Recommended: macOS/Linux/WSL; adapt powershell for native Windows)*

### 1. Clone repository  

git clone https://github.com/Pooja-Arumugam/ANNClassification-Churn-Analysis.git
cd ANNClassification-Churn-Analysis

### 2.Create virtual environment

python -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt
###### How To Run
A. Explore EDA & Model Training
- jupyter notebook notebooks/churn_analysis.ipynb
B. Train the ANN model
- python scripts/model.py
- Produces the trained model at models/ann_model.h5.
C. Launch the Streamlit app
- streamlit run app.py
- Open link on your browser

## ANN Model Architecture
- Layer	Units	Activation
- Input	11	—
  - Hidden Layer 1	6	ReLU
  - Hidden Layer 2	6	ReLU
  - Output	1	Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs / Batch: 100 / 32

## Evaluation Metrics
Metric	Score
Accuracy	~87%
Precision	~79%
Recall	~76%
F1 Score	~78%
ROC-AUC	~0.88

Check the /outputs/ folder for confusion matrix and ROC visuals.

## Key Insights
- Top predictors: Geography, Age, Active status
- Users with low engagement and fewer products show higher churn risk
- Salary alone is not a strong differentiator

## Project Highlights
✅ End-to-end pipeline from EDA to live deployment

✅ Modular, reusable Python scripts

✅ Interactive web app for real-time prediction

✅ Clear business framing and documentation

✅ Environment reproducibility via virtualenv & requirements.txt

## Future Enhancements
- Hyperparameter tuning (Optuna/GridSearch)
- Handle class imbalance (SMOTE, class weights)
- Explainability (SHAP/LIME)
- Containerize using Docker
- Add CSV batch inference in Streamlit
- Deploy to Cloud or Hugging Face Spaces


### Acknowledgments
- Kaggle Telco/Banking datasets
- TensorFlow & Streamlit open-source communities
- ML tutorials and practice programs
