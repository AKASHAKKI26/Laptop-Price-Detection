Working Link -- https://price-detection.streamlit.app/

🛒 Amazon Laptop Price Prediction App

A Machine Learning web application built using Streamlit that predicts laptop prices based on brand, processor, RAM, storage, OS, GPU, and rating.

📌 Project Overview

This project uses:

Linear Regression (from Scikit-Learn)

Pandas for data preprocessing

Streamlit for web UI

One-Hot Encoding for categorical variables

The app allows users to:

View dataset preview

Train the model

Check R² Score

Input laptop specifications

Predict laptop price in ₹ (INR)

📂 Project Structure
Amazon-Price-Prediction/
│
├── detect.py
├── Price_detection.xls
├── README.md
└── requirements.txt
⚙️ Features

✔ Data loading and preprocessing
✔ Currency conversion (USD → INR)
✔ One-hot encoding for categorical features
✔ Train-test split
✔ Linear Regression model training
✔ R² Score evaluation
✔ Interactive Streamlit UI
✔ Real-time price prediction

🧠 Machine Learning Workflow

Load dataset

Convert USD price to INR

Clean numeric columns (RAM, Storage)

Apply one-hot encoding

Split into training and testing data

Train Linear Regression model

Evaluate using R² Score

Predict based on user input

📊 Model Evaluation

The model uses:

R² Score for performance evaluation

R² Score ranges from:

1 → Perfect prediction

0 → No predictive power

<0 → Poor model

📥 User Input Features

Brand

Processor

RAM (GB)

Storage (GB)

Operating System

GPU

Rating (0–5)

📈 Output

R² Score

Predicted Laptop Price in ₹

🛠️ Technologies Used

Python 3.13

Streamlit

Pandas

Scikit-Learn

NumPy

🚀 Future Improvements

Add RMSE & MAE metrics

Deploy on Streamlit Cloud

Add interactive graphs

Use advanced models (Random Forest, XGBoost)

Save trained model using Pickle
