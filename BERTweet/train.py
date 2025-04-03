import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

CHECKPOINT = "vinai/bertweet-base"
# Load dataset
df = pd.read_csv("../train.csv", encoding="utf-8")

# keep only class and tweet columns
df = df[['class', 'tweet']].dropna()

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Convert to HuggingFace dataset
hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }


# Split dataset
dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


def main():
    trainer.train()
    # Evaluate model
    results = trainer.evaluate()
    print("Evaluation Results:", results)


if __name__ == '__main__':
    main()
