# UPI Fraud Detection System

A machine learning-based web application that predicts whether a UPI transaction may be fraudulent.

## 🚀 Live Demo

Try the application here: [UPI Fraud Detection](https://upi-fraud-detection-1tog.onrender.com)

## 📌 Project Overview

Digital payment fraud is a growing concern. This project demonstrates how machine learning can help identify potentially fraudulent UPI transactions based on transaction information.

## ✨ Features

- Predicts potentially fraudulent transactions.
- Uses a trained machine learning model.
- Provides a web interface for transaction analysis.
- Uses saved encoders to process categorical inputs, where required.

## 🛠️ Technologies Used

- Python
- Flask (if used by the application)
- Pandas
- Machine Learning
- HTML and CSS

## 🧠 How It Works

1. The user enters the required transaction details.
2. The application prepares the input features.
3. Saved encoders transform categorical values when required.
4. The trained model processes the features and predicts the transaction class.
5. The application displays the prediction.

The model's reliability depends on the quality of the training data and its evaluation results.

## 📂 Project Structure

- `app.py` — Main web application
- `fraud_model.pkl` — Saved trained model
- `encoders.pkl` — Saved encoders
- `train.py` — Training script
- `train_model.py` — Model training script
- `templates/` — HTML templates
- `static/` — CSS and other static assets
- `sampledata.csv` — Sample dataset
- `requirements.txt` — Python dependencies

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/DEEPTHI-SRINIVASA/UPI-FRAUD-DETECTION.git
cd UPI-FRAUD-DETECTION
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local address shown in the terminal. The application must have its required model files and configuration available.

## 🎯 Objective

To demonstrate the use of machine learning for identifying potentially fraudulent UPI transactions.

## ⚠️ Disclaimer

This project is for educational purposes only. Predictions may be incorrect and should not be used as the sole basis for real financial decisions.
