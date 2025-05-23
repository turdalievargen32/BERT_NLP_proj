# BERT-based Sentiment Analysis Project

This project demonstrates a complete pipeline for **sentiment analysis** using a fine-tuned BERT model. It includes training, evaluation, SHAP-based explainability, and attention visualization.

## 📌 Project Features

- **Model:** BERT (`bert-base-uncased`)
- **Task:** Binary sentiment classification (`Positive` vs `Negative`)
- **Explainability:** SHAP values for token-level interpretation
- **Visualization:** Confusion matrix, classification report, attention heatmap

---

## ✅ Evaluation Results

### Confusion Matrix

![Confusion Matrix](./confusion_matrix.png)

- True Negatives (TN): 10,952
- False Positives (FP): 1,548
- False Negatives (FN): 1,307
- True Positives (TP): 11,193

---

### Classification Report

![Classification Report](./classification_report_precise.png)

- **Accuracy:** 89%
- **Precision (Negative):** 90%  
- **Precision (Positive):** 88%  
- **Recall (Negative):** 88%  
- **Recall (Positive):** 90%  
- **F1-score (both classes):** 0.89

---

## 🔍 SHAP Explanation Example

![SHAP Analysis](./Screenshot\ from\ 2025-05-23\ 17-56-35.png)

The word **"disappointing"** had the most negative impact on the sentiment classification, contributing `-0.962` to the negative class.

---

## 🧠 BERT Attention Heatmap

![Attention Heatmap](./Attention_heatmap.png)

This heatmap shows how the BERT model allocates attention across the sentence tokens in the **last layer** by averaging all heads.

---

## 💻 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/turdalievargen32/BERT_NLP_proj.git
   cd BERT_NLP_proj
