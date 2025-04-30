import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

CHECKPOINTS = {
    "base": "vinai/bertweet-base",
    "large": "vinai/bertweet-large"
}

# https://github.com/VinAIResearch/BERTweet
MAX_LENGTH = {
    "vinai/bertweet-base": 128,
    "vinai/bertweet-large": 512
}

# Load dataset
df = pd.read_csv("../train.csv", encoding="utf-8")

# keep only class and tweet columns
df = df[["class", "tweet"]] \
        .dropna() \
        .rename(columns={"class": "labels", "tweet": "text"})

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }

def preprocess_and_tokenize(checkpoint_name):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    def preprocess_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH[checkpoint_name])

    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(preprocess_func, batched=True)
    dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    return dataset, tokenizer

def train_model(model_size):
    checkpoint = CHECKPOINTS[model_size]
    dataset, tokenizer = preprocess_and_tokenize(checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

    output_dir = f"./results_{model_size}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f"Training {model_size} model...")
    trainer.train()

    print(f"Evaluating {model_size} model...")
    results = trainer.evaluate()
    print(f"{model_size.capitalize()} model evaluation results:", results)

    # Save the best model
    trainer.save_model(f"{output_dir}/best_model")

def main():
    for size in CHECKPOINTS.keys():
        train_model(size)

if __name__ == '__main__':
    train_model("large")
