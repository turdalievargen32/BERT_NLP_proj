from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Load model and tokenizer
model_path = "/content/sentiment-bert/src/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Define prediction function
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    return probs

# 3. Input text for explanation
text = "I didn't like the movie. It was too slow and boring, but the ending was good."

# 4. Run LIME explanation
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
exp = explainer.explain_instance(text, predict_proba, num_features=6)

# 5. Save the result
exp.save_to_file("lime_output.html")
print("✅ LIME explanation saved as 'lime_output.html'")
