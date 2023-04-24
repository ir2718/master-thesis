from abc import abstractmethod
import numpy as np
import random
import torch
import os
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def get_compute_metrics(dataset):
    MAPPING_COMPUTE_METRICS = {
        "CT23": compute_metrics,
        "CT21": compute_metrics,
        "FEVER": compute_metrics_multiclass,
        "covidFACT": compute_metrics,
    }
    return MAPPING_COMPUTE_METRICS[dataset]

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_pred=preds, y_true=labels),
        "recall": recall_score(y_pred=preds, y_true=labels),
        "precision": precision_score(y_pred=preds, y_true=labels),
        "f1": f1_score(y_pred=preds, y_true=labels)
    }

def compute_metrics_multiclass(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_pred=preds, y_true=labels),
        "recall": recall_score(y_pred=preds, y_true=labels, average="macro"),
        "precision": precision_score(y_pred=preds, y_true=labels, average="macro"),
        "f1": f1_score(y_pred=preds, y_true=labels, average="macro")
    }

def batch_to_device(batch, device):
    return {k:v.to(device) for k, v in batch.items()}

def _tokenize(text, tokenizer):
    return tokenizer(text, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")

def _tokenize_checkworthiness(x, **kwargs):
    # [CLS] claim [SEP]
    return _tokenize(text=x["text"], **kwargs)

def _tokenize_veracity(x, **kwargs):
    # [CLS] claim [SEP] evidence1 [SEP] evidence_n [SEP]
    l = [f"{i} {kwargs['tokenizer'].sep_token}" + f" {kwargs['tokenizer'].sep_token} ".join(j) for i, j in zip(x["claim"], x["evidence"])]
    tokenized  = _tokenize(text=l, **kwargs)
    return tokenized

# def _tokenize_veracity_with_special(x):
#     s = " [SEP_EVIDENCE]"
#     for l in x["evidence"]:
#         print(l)
#         s += f" [SEP_ENT] {l[0]} [SEP_NUM] {l[1]} [SEP_EVIDENCE_SENT] {l[2]}"
#     s = x["claim"][0] + s
#     return  _tokenize(s, tokenizer=tokenizer)

def tokenize_dataset(dataset_dict, tokenizer, type_):
    tokenize_f = _tokenize_checkworthiness if type_ == "checkworthiness" else _tokenize_veracity
    dataset_dict = dataset_dict.map( 
        lambda x: tokenize_f(x, tokenizer=tokenizer),
        batched=True
    )
    return dataset_dict

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")