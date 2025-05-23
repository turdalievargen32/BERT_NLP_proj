import pandas as pd
import torch 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load model and tokenizer
model_path = "/content/sentiment-bert/src/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Load local test set
df = pd.read_csv("/content/sentiment-bert/data/test.csv")  # assumes 'text' and 'label' columns
texts = df["text"].tolist()
true_labels = df["label"].tolist()

# 3. Predict
def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1).numpy()
    return preds, probs[:, 1].numpy()

preds, scores = predict(texts)

# 4. Classification report
print("Classification Report:")
print(classification_report(true_labels, preds))

# 5. Confusion Matrix
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# 6. ROC Curve
fpr, tpr, _ = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()
