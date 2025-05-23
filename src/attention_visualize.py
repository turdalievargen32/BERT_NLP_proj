import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

# 1. Load model and tokenizer with attention outputs enabled
model_path = "/content/sentiment-bert/src/best_model"
model = AutoModel.from_pretrained(model_path, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Input text
text = "I didn't like the movie. It was too slow and boring, but the ending was good."
inputs = tokenizer(text, return_tensors="pt")

# 3. Forward pass to get attention
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # shape: (layers, batch, heads, tokens, tokens)

# 4. Take the last layer, average over heads
attention_matrix = attentions[-1][0].mean(dim=0).numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 5. Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu")
plt.title("BERT Attention Heatmap (last layer, avg over heads)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("attention_heatmap.png")
print("✅ Attention heatmap saved as 'attention_heatmap.png'")


