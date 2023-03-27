import pandas as pd
import os
from datasets import DatasetDict, Dataset, load_dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval

# every dataset is loaded as a huggingface dataset

def get_dataset(dataset, **kwargs):
    DATASET_MAPPING = {
        "CT23": (_load_CT23_dataset, "checkworthiness"),
        "CT21": (_load_CT21_dataset, "checkworthiness"),
        "FEVER": (_load_FEVER_dataset, "verification")
    }
    dataset_f, type_ = DATASET_MAPPING[dataset]
    return dataset_f(**kwargs), type_

def get_num_labels(dataset_dict):
    return len(dataset_dict.unique("label")["train"])

def _make_dict(train, validation, test):
    hf_dfs = DatasetDict()
    hf_dfs["train"] = Dataset.from_pandas(train)
    hf_dfs["validation"] = Dataset.from_pandas(validation)
    hf_dfs["test"] = Dataset.from_pandas(test)
    return hf_dfs

def _load_CT23_dataset(path=os.path.join("./data", "checkworthiness", "CT23_1B_checkworthy_english")):
    dataset_paths = [
        "CT23_1C_checkworthy_english_train.tsv",
        "CT23_1C_checkworthy_english_dev.tsv",
        "CT23_1B_checkworthy_english_dev_test.tsv",
    ]
    dfs = [pd.read_csv(os.path.join(path, p), sep="\t") for p in dataset_paths]
    for d in dfs:
        d.loc[:, "class_label"].replace({"Yes": 1, "No": 0}, inplace=True)
        d.rename(columns={
            "Sentence_id": "id",
            "Text": "text",
            "class_label": "label"
        }, inplace=True)
    hf_dfs = _make_dict(*dfs)
    return hf_dfs


def _load_CT21_dataset(path=os.path.join("./data", "checkworthiness", "CT21_1A_checkworthy_tweets"), multi=False):
    dataset_paths = [
        "dataset_train_v1_english.tsv",
        "dataset_dev_v1_english.tsv",
        "dataset_test_english.tsv",
    ]
    dfs = [pd.read_csv(os.path.join(path, p), sep="\t") for p in dataset_paths]
    for i, d in enumerate(dfs):
        dfs[i].drop(columns=["tweet_url"], inplace=True)
        if not multi:
            dfs[i].drop(columns=["topic_id", "claim"], inplace=True)
        d.rename(columns={
            "tweet_id": "id",
            "tweet_text": "text",
            "claim_worthiness" if i == 1 else "check_worthiness": "label"
        }, inplace=True)
    hf_dfs = _make_dict(*dfs)
    return hf_dfs

def _load_FEVER_dataset(path=os.path.join("./data", "verification", "fever"), sampled=True):
    dataset_paths = [
        "fever_train_sampled.csv" if sampled else "fever_train.csv",
        "fever_validation.csv",
        "fever_test.csv",
    ]
    dfs = [pd.read_csv(os.path.join(path, p), converters={"evidence": pd.eval}) for p in dataset_paths]
    for i, d in enumerate(dfs):
        if not sampled:
            dfs[i].drop(columns=["verifiable", "original_id"], inplace=True)
        dfs[i].loc[:, "evidence"] = dfs[i].evidence.apply(lambda x: [" ".join(s) for s in x])
        d.loc[:, "label"].replace({
            "REFUTES": 0, 
            "SUPPORTS": 1, 
            "NOT ENOUGH INFO": 2
        }, inplace=True)
    hf_dfs = _make_dict(*dfs)
    return hf_dfs

def collate_fn(examples, **kwargs):
    if kwargs["tokenizer"]._pad_token is None:
        return pad_sequence(examples, batch_first=True)
        
    return pad_sequence(
        examples, batch_first=True, padding_value=kwargs["tokenizer"].pad_token_id
    )

if __name__ == "__main__": # for testing
    #_load_CT23_dataset()
    _load_FEVER_dataset()