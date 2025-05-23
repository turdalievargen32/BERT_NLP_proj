import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# 1. Load model and tokenizer
model_path = "/content/sentiment-bert/src/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Prediction function for SHAP
def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()

# 3. Input text
text = "I didn't like the movie. It was too slow and boring, but the ending was good."

# 4. SHAP Explainer
explainer = shap.Explainer(predict, tokenizer)
shap_values = explainer([text])

# 5. Plot and save
shap.plots.text(shap_values[0], display=False)
plt.savefig("shap_text_explanation.png", dpi=300, bbox_inches="tight")
print("✅ SHAP plot saved as shap_text_explanation.png")
