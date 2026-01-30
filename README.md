# Sampling Techniques Analysis on an Imbalanced Credit Card Dataset

## Project Overview
In this project, an experimental study is carried out to understand how **different sampling techniques** affect the performance of **machine learning classification models** when working with a **highly imbalanced dataset**.  

The dataset used is a credit card transaction dataset where fraudulent transactions are very rare compared to non-fraudulent ones. Such imbalance can cause machine learning models to become biased toward the majority class. To address this issue, the dataset is first balanced and then multiple sampling strategies are applied and evaluated.

The main goal of this project is to compare sampling techniques and analyze which sampling method works best with different machine learning models.

---

## Dataset Description
- **Dataset Name:** Credit Card Fraud Detection Dataset  
- **Target Attribute:** `Class`  
  - `0` → Legitimate (Non-Fraud) transaction  
  - `1` → Fraudulent transaction  

Initially, the dataset contains a very large number of non-fraud transactions and a very small number of fraud transactions. This imbalance makes it unsuitable for direct model training without preprocessing.

---

## Libraries and Tools Used
The following Python libraries were used during implementation:

- **Pandas** – data loading and manipulation  
- **NumPy** – numerical operations  
- **Scikit-learn** – sampling methods, model training, and evaluation  
- **Matplotlib** – visualization of results  

---

## Data Preprocessing

### Balancing the Dataset
Before applying sampling techniques, the dataset is converted into a balanced dataset using **random undersampling**.  

In this step:
- Samples from the majority class (`Class = 0`) are reduced
- All samples from the minority class (`Class = 1`) are retained
- The final dataset contains an equal number of fraud and non-fraud records

Balancing the dataset helps reduce bias and allows machine learning models to learn meaningful patterns from both classes.

---

## Sampling Techniques Applied
After balancing, five different sampling techniques are applied to create multiple samples of the data. Each technique follows a different selection strategy.

| Sampling ID | Sampling Technique |
|------------|-------------------|
| Sampling1 | Simple Random Sampling |
| Sampling2 | Stratified Random Sampling |
| Sampling3 | Bootstrap Sampling |
| Sampling4 | Systematic Sampling |
| Sampling5 | Cluster Sampling |

These techniques were chosen to represent a wide variety of sampling approaches, including random selection, proportion-preserving sampling, sampling with replacement, interval-based sampling, and group-based sampling.

---

## Machine Learning Models Used
Each sampled dataset is used to train the following machine learning models:

| Model ID | Algorithm |
|--------|----------|
| M1 | Logistic Regression |
| M2 | Decision Tree Classifier |
| M3 | Random Forest Classifier |
| M4 | K-Nearest Neighbors (KNN) |
| M5 | Support Vector Machine (SVM) |

This combination allows analysis of how sampling methods interact with both linear and non-linear models.

---

## Model Training and Evaluation
For every sampling technique:
- The sampled dataset is split into **70% training data** and **30% testing data**
- Each model is trained independently on the training set
- Predictions are made on the test set
- **Accuracy (%)** is calculated as the evaluation metric

All accuracy values are stored in a comparison table for further analysis.

---

## Visual Analysis and Plots

- **Accuracy Comparison Line Plot**  
<img width="2557" height="1638" alt="sampling_vs_models_accuracy" src="https://github.com/user-attachments/assets/0faa01e8-458d-4d2b-a97e-9dacd9f40640" />

---

- **Average Accuracy per Sampling Technique**  
<img width="2060" height="1407" alt="average_accuracy_sampling" src="https://github.com/user-attachments/assets/dd6458ab-a523-4039-bc8d-afa7ae4668be" />

---

- **Best Sampling Technique per Model**
  
<img width="2087" height="1407" alt="best_sampling_per_model" src="https://github.com/user-attachments/assets/d77f9676-5cb1-43f8-91bd-645c99fde440" />

---
- **Sampling Ranking (Graph)**  
 <img width="2060" height="1407" alt="sampling_ranking" src="https://github.com/user-attachments/assets/9388ae6a-dcee-4998-8d76-74c76fb54409" />

---

- **Sampling Technique Stability Plot (Variance)**  
<img width="2087" height="1407" alt="sampling_variance" src="https://github.com/user-attachments/assets/98f3bbd1-024a-4ed0-9669-25780056b575" />

  ---

  - **Accuracy of Several Sampling Techniques**  
 <img width="2552" height="1659" alt="sampling_bar_plot" src="https://github.com/user-attachments/assets/f63b747e-83dd-43a0-bb60-4d8cf53e0259" />



---

## Results and Observations
From the experimental results, the following observations were made:

- Bootstrap Sampling and Cluster Sampling perform well for multiple models
- Some models are highly sensitive to the choice of sampling technique
- Sampling strategy plays a crucial role in model performance on imbalanced data

---

## Conclusion
The study demonstrates that careful selection of sampling techniques is essential, since different models react differently and significantly affect prediction reliability.

---

## Author
**Krish Mahajan**

## License 
This project is open-source and available under the MIT License.
