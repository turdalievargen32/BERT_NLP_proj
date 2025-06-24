# 🧠 Sentiment Analysis with BERT

This project implements a **Sentiment Analysis** pipeline using a fine-tuned BERT model (`bert-base-uncased`) for classifying text data into **positive**, **negative**, or **neutral** sentiments. It combines powerful **transformer-based modeling** with modern tools for **training**, **evaluation**, **visualization**, and **interpretability**.

## 🚀 Features

- ✅ Fine-tuning BERT for sequence classification using Hugging Face Transformers  
- ✅ Clean, configurable training and evaluation workflow  
- ✅ Sentiment prediction on custom texts  
- ✅ Performance metrics (accuracy, precision, recall, F1)  
- ✅ Visualization tools (ROC curve, confusion matrix, SHAP, LIME)  
- ✅ Easy error analysis and explainability  

## 📁 Project Structure
```
train.py # Script to fine-tune BERT model
├── evaluate.py # Script to evaluate model performance
├── predict.py # Predict sentiment for new input
├── dataset.py # Dataset loading and preprocessing
├── error_analysis.py # Tools for misclassification analysis
├── shap_explain.py # SHAP explainability
├── lime_explain.py # LIME interpretability
├── utils.py # Helper functions
├── config.json # Hyperparameter and config file
├── requirements.txt # Python dependencies
├── report.pdf # 📄 Final project report (see below)
└── README.md # Project documentation

```


## 🛠️ Quickstart

### 🔽 Clone and Run

```bash
git clone https://github.com/turdalievargen32/BERT_NLP_proj.git
cd BERT_NLP_proj
pip install -r requirements.txt
python train.py

python evaluate.py

python predict.py --text "This movie was amazing!"
```
 ## Explainability Tools

    SHAP (shap_explain.py): Visualize feature importance

    LIME (lime_explain.py): Understand individual predictions
    
