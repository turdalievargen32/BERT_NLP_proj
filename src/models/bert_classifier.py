from transformers import BertForSequenceClassification

def build_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2  # positive / negative
    )
    return model


