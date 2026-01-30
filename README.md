# Sampling Techniques Analysis on Imbalanced Dataset

## 📌 Project Overview
This project analyzes the effect of different **sampling techniques** on the performance of various **machine learning models** using a **highly imbalanced credit card dataset**.  
The dataset is first balanced using random undersampling, after which multiple sampling strategies are applied and evaluated.

---

## 📂 Dataset
- **Name:** Credit Card Fraud Detection Dataset  
- **Target Column:** `Class`  
  - `0` → Non-Fraud  
  - `1` → Fraud  

The original dataset is highly imbalanced, making it unsuitable for direct model training.

---

## ⚙️ Libraries Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 🔄 Data Preprocessing

### 1️⃣ Balancing the Dataset
The dataset is balanced using **random undersampling**:
- Majority class samples are reduced
- Minority class samples are retained
- Final dataset contains equal instances of both classes

This prevents model bias towards the majority class.

---

## 🔁 Sampling Techniques Implemented

Five different sampling techniques are applied to the balanced dataset:

| Sampling ID | Technique |
|------------|----------|
| Sampling1 | Simple Random Sampling |
| Sampling2 | Stratified Sampling |
| Sampling3 | Bootstrap Sampling |
| Sampling4 | Systematic Sampling |
| Sampling5 | Cluster Sampling |

Each sampling technique follows a distinct strategy, allowing comparative analysis.

---

## 🤖 Machine Learning Models Used

| Model ID | Algorithm |
|--------|----------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors |
| M5 | Support Vector Machine |

---

## 📊 Model Training & Evaluation
- Each sampled dataset is split into **70% training** and **30% testing**
- Models are trained independently on each sample
- **Accuracy (%)** is used as the evaluation metric
- Results are stored in a comparison table

---

## 📈 Visual Analysis

The following plots were generated to analyze model performance:

- **Accuracy Comparison Line Plot**  
  `sampling_vs_models_accuracy.png`

- **Average Accuracy per Sampling Technique**  
  `average_accuracy_sampling.png`

- **Best Sampling Technique per Model**  
  `best_sampling_per_model.png`

- **Accuracy Heatmap (Models vs Sampling)**  
  `accuracy_heatmap.png`

- **Sampling Technique Stability (Variance Plot)**  
  `sampling_variance.png`

All plots are saved as PNG files in the repository.

---

## 🏆 Results Summary
The analysis shows that:
- No single sampling technique performs best for all models
- Bootstrap and Cluster sampling perform well across multiple models
- Model performance is highly dependent on the chosen sampling strategy

---

## 📁 Project Structure
