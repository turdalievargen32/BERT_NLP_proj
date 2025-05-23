from utils.text_cleaning import clean_text
from data.dataset import IMDbDataset

from datasets import load_dataset
from transformers import AutoTokenizer
from utils.text_cleaning import clean_text
from datasets import DatasetDict
from torch.utils.data import DataLoader
from data.dataset import IMDbDataset
from training.train_with_trainer import run_trainer
from transformers import AutoModelForSequenceClassification

def preprocess_function(example):
    example["text"] = clean_text(example["text"])
    return example

def main():
    # Step 1: Load dataset
    dataset = load_dataset("imdb")

    # Step 2: Apply cleaning
    dataset = dataset.map(preprocess_function)

    # Step 3: Check cleaned output
    print("Sample after cleaning:")
    print(dataset['train'][0]['text'][:300])

    # Step 4: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Step 4: Tokenize all examples
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

     # Step 5: Tokenize train and test separately
    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)


    # Step 6: Split tokenized train into train + validation
    split_dataset = tokenized_train.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    test_dataset = tokenized_test

    # DataLoader 

    train_data = IMDbDataset(train_dataset)
    val_data = IMDbDataset(val_dataset)
    test_data = IMDbDataset(test_dataset)

    train_loader = DataLoader(train_data, batch_size = 16, shuffle = True)
    val_loader  = DataLoader(val_data, batch_size = 16)
    test_loader = DataLoader(test_data, batch_size = 16 )

    


    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    run_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
      # Step 7: Print preview
    print("Sample input_ids from train set:")
    print(train_dataset[0]["input_ids"][:10])
    print("Sample attention_mask:")
    print(train_dataset[0]["attention_mask"][:10])




if __name__ == "__main__":
    main()



