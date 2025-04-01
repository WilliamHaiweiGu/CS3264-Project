import gc
import os
import re
import sys
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import XLNetForSequenceClassification, XLNetTokenizer, get_scheduler

print("Python", sys.version, "on", sys.platform)

"""Training parameters"""
MEMORY_FACTOR: int = 61440  # Proportional to Device Memory usage. About 3 GB at 2048
MAX_LEN: int = 512
PREFER_GPU: bool = True
DATA_PROPORTION: float = 1
LEARN_RATE: float = 2e-5  # Increased learning rate
N_EPOCH: int = 100
START_MODEL_FILE: str | None = None
TEST_SIZE: float = 0.3
"""Constants"""
BATCH_SIZE: int = MEMORY_FACTOR // MAX_LEN
RAND_STATE: int = 69

tensor_to_list: Callable[[torch.Tensor], List[int | float] | int | float] = lambda ts: ts.detach().cpu().tolist()

CACHE_PATH: str = "../.cache"


def get_result_with_cache(file_name: str, read_file_fn: Callable[[str], Any], comp_fn: Callable[[], Any],
                          save_file_fn: Callable[[Any, str], None]):
    """
    :param read_file_fn file_path -> data
    :param comp_fn      None -> data
    :param save_file_fn (data, file_path) -> None
    """
    path: str = os.path.join(CACHE_PATH, file_name)
    if os.path.isfile(path):
        print("Use cached:", path)
        return read_file_fn(path)
    ans = comp_fn()
    if os.path.isfile(CACHE_PATH):
        print("Unable to save data:", CACHE_PATH, "is a file.")
    elif os.path.isdir(path):
        print("Unable to save data:", path, "is a directory.")
    else:
        if not os.path.isdir(CACHE_PATH):
            os.mkdir(CACHE_PATH)
        save_file_fn(ans, path)
        print("Saved computation result to:", path)
    return ans


def del_if_exists(file_path: str):
    if os.path.isfile(file_path):
        os.remove(file_path)


# Initialization
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
torch.serialization.add_safe_globals([TensorDataset])
torch.manual_seed(RAND_STATE)
can_cuda: bool = PREFER_GPU and torch.cuda.is_available()
device = torch.device("cuda" if can_cuda else "cpu")
print("PyTorch will use GPU:", can_cuda)


def clean_text(text: str) -> str:
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    return re.sub(r" +", ' ', text)


def make_tensor_dataset(raw_texts: List[str], raw_labels: List[int]) -> TensorDataset:
    inputs = tokenizer(raw_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    del raw_texts
    labels = torch.tensor(raw_labels)
    del raw_labels
    return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)


def comp_datasets(sample_proportion: int | float) -> Callable[[], Tuple[TensorDataset, TensorDataset]]:
    """

    :param sample_proportion: Proportion of dataset used
    :return: (Train TensorDataset, Test TensorDataset)
    """

    def fn() -> Tuple[TensorDataset, TensorDataset]:
        df = pd.read_csv("../train.csv", usecols=["tweet", "class"])
        df = df.sample(frac=max(0, min(1, sample_proportion)), replace=False, random_state=RAND_STATE)
        df.rename(columns={"class": "label"}, inplace=True)
        raw_texts, raw_labels = [], []
        for row in df.itertuples(index=False):
            try:
                truth_label: int = int(row.label)
            except TypeError | ValueError:
                continue
            raw_labels.append(truth_label)
            raw_texts.append(row.tweet)
        del df
        X_train, X_test, y_train, y_test = train_test_split(raw_texts, raw_labels, test_size=TEST_SIZE,
                                                            random_state=RAND_STATE)
        return make_tensor_dataset(X_train, y_train), make_tensor_dataset(X_test, y_test)

    return fn


if __name__ == "__main__":
    train_dataset, test_dataset = get_result_with_cache(f"XLNetTokens{MAX_LEN}-{DATA_PROPORTION}", torch.load,
                                                        comp_datasets(DATA_PROPORTION), torch.save)
    # Load train data
    _, _, y_train = train_dataset.tensors
    y_train: List[int] = tensor_to_list(y_train)
    n_class: int = max(y_train) + 1

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    del train_dataset
    # Add class weights to handle class imbalance
    class_weights = compute_class_weight("balanced", classes=np.arange(n_class), y=y_train)
    print("Class weights =", class_weights)
    del y_train
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Implementation of weighted loss function
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda logits, labels:\
        torch.nn.functional.cross_entropy(logits, labels, weight=class_weights_tensor)
    # Load test data
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    _, _, y_test = test_dataset.tensors
    y_test: List[int] = tensor_to_list(y_test)
    del test_dataset
    # Load model
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=n_class)
    if START_MODEL_FILE is not None and os.path.isfile(START_MODEL_FILE):
        model.load_state_dict(torch.load(START_MODEL_FILE))
        print("Loaded model", START_MODEL_FILE)
    model.to(device)
    # Configure optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)
    del no_decay
    optimizer_grouped_parameters = [{'params': params_with_decay, 'weight_decay': 0.01},
                                    {'params': params_without_decay, 'weight_decay': 0.0}]
    del params_with_decay
    del params_without_decay
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARN_RATE)
    del optimizer_grouped_parameters

    # Add learning rate scheduler
    num_training_steps = len(train_dataloader) * N_EPOCH
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps),
                              num_training_steps=num_training_steps)

    # Train
    get_model_file_name: Callable[[float], str] = lambda \
            f1: f"XLNet_{BATCH_SIZE}_{MAX_LEN}_{DATA_PROPORTION}_{LEARN_RATE}_{round(f1, 3)}.model"
    best_f1: float = 0
    for epoch in range(N_EPOCH):
        model.train()
        for input_ids, attention_mask, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            loss = loss_fn(outputs.logits, labels.to(device))  # Use weighted loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        y_test_pred: List[int] = []
        with torch.no_grad():
            for input_ids, attention_mask, _ in tqdm(test_dataloader, desc=f"Test {epoch}"):
                test_outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                y_test_pred.extend(tensor_to_list(torch.argmax(test_outputs.logits, dim=1)))

        print("Confusion matrix (i-th actual class and j-th predicted class)")
        print(confusion_matrix(y_test, y_test_pred))
        f1: float = f1_score(y_test, y_test_pred, average='macro')
        print("Macro F1 Score =", f1)
        if f1 < 1.0 / n_class:
            print("Abnormal F1 Score, aborting...")
            break
        if f1 > best_f1:
            del_if_exists(get_model_file_name(best_f1))
            best_f1 = f1
            torch.save(model.state_dict(), get_model_file_name(f1))
    print(f"Saved state dict of the best model (Macro F1 score = {best_f1}) to", get_model_file_name(best_f1))

    # Free CUDA memory
    del optimizer
    model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
    gc.collect()
