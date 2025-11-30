# 📈 PLFD European Stock Predictor

## 📝 Description
This project was created for the course **Predictive Learning from Data**  
at **National Taiwan Normal University**, Winter Semester **2025**.

It aims to explore, engineer, and evaluate machine-learning models for financial forecasting.

---

## 🎯 Goal of the Project
The goal is to evaluate multiple prediction models and identify one that can **reliably predict the opening price of the STOXX Europe 600**, using several **Asian stock indices** as input features.

---

## 🚀 How to Use

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/niMtAndoX/plfd-european-stock-predictor.git
cd plfd-european-stock-predictor
```

---

### 2️⃣ Create a Virtual Environment

#### Windows
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the ETL Pipeline
This script downloads the raw data, cleans and processes it, generates engineered features, and produces visualizations.

```bash
python src/data_processing/etl_pipeline.py
```

---
