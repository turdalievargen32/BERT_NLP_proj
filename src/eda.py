import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def run_eda():
    # Load dataset
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset["train"][:2000])  # take a sample for speed

    # Plot class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="label", data=df)
    plt.title("Class Distribution")
    plt.xlabel("Sentiment (0 = negative, 1 = positive)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png")
    plt.close()

    # Plot review lengths
    df["length"] = df["text"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(6,4))
    sns.histplot(df["length"], bins=50, kde=True)
    plt.title("Review Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/length_distribution.png")
    plt.close()

    print("EDA completed. Plots saved in outputs/")

if __name__ == "__main__":
    run_eda()
