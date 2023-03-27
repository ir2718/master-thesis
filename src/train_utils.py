from abc import abstractmethod
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def get_compute_metrics(dataset):
    MAPPING_COMPUTE_METRICS = {
        "CT23": compute_metrics,
        "CT21": compute_metrics,
        "FEVER": compute_metrics_multiclass,
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

def _tokenize(text, tokenizer):
    return tokenizer(text, truncation=True, padding=True, return_tensors="pt")

def _tokenize_checkworthiness(x, **kwargs):
    return _tokenize(text=x["text"], **kwargs)

def _tokenize_veracity(x, **kwargs):
    l = [f"{i} [SEP] {' [SEP] '.join(j)}" for i, j in zip(x["claim"], x["evidence"])]
    return  _tokenize(text=l, **kwargs)

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
        batched=True # False
    )
    return dataset_dict