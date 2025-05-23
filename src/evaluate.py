from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load model and tokenizer
model_path = "/content/sentiment-bert/src/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Load dataset
dataset = load_dataset("imdb", split="test[:5000]")

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. Inference
all_preds = []
all_labels = []
all_logits = []

for batch in torch.utils.data.DataLoader(dataset, batch_size=16):
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(batch["label"].numpy())
        all_logits.extend(logits.numpy())

# 5. Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

# 6. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as 'confusion_matrix.png'")
