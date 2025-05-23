from transformers import Trainer, TrainingArguments 
from sklearn.metrics import accuracy_score, f1_score 
import numpy as np 

def compute_metrics(pred):
    labels = pred.label_ids 
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def run_trainer(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs"
)

    
    trainer = Trainer(
        model = model,
        args= training_args,
        train_dataset = train_dataset, 
        eval_dataset = val_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    trainer.train()
    trainer.save_model("./best_model")
    tokenizer.save_pretrained("./best_model")
