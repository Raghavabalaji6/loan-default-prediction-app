Loan Default Risk Prediction (Machine Learning App)

This project predicts whether a loan applicant is likely to default using a Machine Learning model trained on historical financial data. The application is built end-to-end — from data preprocessing and model training to a deployed Streamlit web app.

⚠️ Disclaimer: This project is for educational and portfolio purposes only, not for real financial decision-making.

🚀 Live Demo :https://loan-default-prediction-app-p5zoqxo9h4zhebkmmcfzjc.streamlit.app/

🔗 Deployed App: (Add your Streamlit Cloud URL here after deployment)

📌 Problem Statement

Banks face significant losses when loans are approved for high-risk customers. The goal of this project is to identify risky borrowers early, prioritizing high recall so that most defaulters are correctly flagged.

🧠 Solution Overview

Binary classification problem:

0 → Safe customer

1 → Risky (likely to default)

Focused on reducing False Negatives (risky customers predicted as safe)

Used threshold tuning instead of relying on default 0.5 probability

📊 Dataset

Large real-world styled loan dataset (~255,000 records)

Features include:

Demographics (Age, Education, Marital Status)

Financial data (Income, Loan Amount, Credit Score)

Credit behavior (DTI Ratio, Credit Lines)

Employment & loan details

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

Joblib

⚙️ ML Pipeline

Data cleaning & preprocessing

Label Encoding for categorical variables

Feature scaling using StandardScaler

Handling class imbalance using scale_pos_weight

Model training with XGBoost Classifier

Threshold tuning (0.30) to improve recall

Model evaluation using confusion matrix & classification report

📈 Model Performance (Test Set)

Recall (Risky Customers): ~88%

Optimized to catch maximum defaulters

Trade-off: lower precision accepted intentionally

🖥️ Web Application

The Streamlit app allows users to:

Enter loan applicant details

View predicted default probability

Get LOW RISK / HIGH RISK classification

See simple rule-based risk indicators

📁 Project Structure loan-default-prediction-app/ │ ├── app.py ├── requirements.txt ├── xgb_default_model.pkl ├── scaler.pkl ├── label_encoders.pkl ├── model_columns.pkl ├── threshold.pkl └── README.md

▶️ How to Run Locally pip install -r requirements.txt streamlit run app.py

🎯 Key Learnings

Importance of handling imbalanced datasets

Why recall matters more than accuracy in finance

Correct use of encoders & scalers during inference

End-to-end ML deployment challenges

📌 Future Improvements

Add SHAP for model explainability

Improve UI with feature contribution insights

Add authentication and logging

Try LightGBM comparison

👤 Author

Raghava Balaji

BCA Graduate

Aspiring Machine Learning Engineer
