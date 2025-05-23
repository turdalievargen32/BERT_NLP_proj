import sys
sys.path.append("/content/sentiment-bert/src")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from data.dataset import IMDbDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели и токенизатора
model_path = "/content/sentiment-bert/src/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Датасет
raw_dataset = load_dataset("imdb")
test_dataset = raw_dataset["test"]

def preprocess(example):
    example["text"] = example["text"]
    return example

test_dataset = test_dataset.map(preprocess)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_test = test_dataset.map(tokenize, batched=True)

test_data = IMDbDataset(tokenized_test)
test_loader = DataLoader(test_data, batch_size=16)

# Предсказания
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
labels = ['Negative', 'Positive']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Примеры ошибок
false_positives = []
false_negatives = []
for i in range(len(all_labels)):
    if all_labels[i] == 0 and all_preds[i] == 1:
        false_positives.append(test_dataset[i]["text"])
    elif all_labels[i] == 1 and all_preds[i] == 0:
        false_negatives.append(test_dataset[i]["text"])

print("\n❌ False Positives:")
for fp in false_positives[:2]:
    print(f"[\033[91m✘\033[0m] {fp}\n")

print("❌ False Negatives:")
for fn in false_negatives[:2]:
    print(f"[\033[91m✘\033[0m] {fn}\n")
