import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import shap
import matplotlib.pyplot as plt

class SentimentSHAPAnalizer:
    def __init__(self, model_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        self.explainer = shap.Explainer(self.pipeline)

    def analyze_text(self, text: str, save_path: str = "shap_text.png"):
        print(f"Analyzing: {text}")
        shap_values = self.explainer([text])
        shap.plots.text(shap_values[0], display=False)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ SHAP plot saved as {save_path}")

    def batch_summary(self, texts, save_path: str = "shap_summary.png"):
        print("Generating SHAP summary plot...")
        shap_values = self.explainer(texts)
        shap.summary_plot(shap_values, show=False)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ SHAP summary saved as {save_path}")


