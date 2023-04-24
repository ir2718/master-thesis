import pandas as pd
import os
from datasets import DatasetDict, Dataset, load_dataset
from transformers import (
    BertTokenizer, 
    ElectraTokenizer,
    BertConfig,
    ElectraConfig,
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval
import torch

# every dataset is loaded as a huggingface dataset

def get_dataset(dataset, **kwargs):
    DATASET_MAPPING = {
        "CT23": (_load_CT23_dataset, "checkworthiness"),
        "CT21": (_load_CT21_dataset, "checkworthiness"),
        "FEVER": (_load_FEVER_dataset, "verification"),
        "covidFACT": (_load_covidFACT_dataset, "verification")
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
    for i, d in enumerate(dfs):
        dfs[i].loc[:, "class_label"].replace({"Yes": 1, "No": 0}, inplace=True)
        dfs[i].rename(columns={
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
        dfs[i].rename(columns={
            "tweet_id": "id",
            "tweet_text": "text",
            "claim_worthiness" if i == 1 else "check_worthiness": "label"
        }, inplace=True)
    hf_dfs = _make_dict(*dfs)
    return hf_dfs

def _load_FEVER_dataset(path=os.path.join("./data", "verification", "FEVER"), sampled=True):
    dataset_paths = [
        "fever_train_sampled.csv" if sampled else "fever_train.csv",
        "fever_validation.csv",
        "fever_test.csv",
    ]
    dfs = [pd.read_csv(os.path.join(path, p), converters={"evidence": pd.eval}) for p in dataset_paths]
    for i, d in enumerate(dfs):
        dfs[i].drop("Unnamed: 0", axis=1, inplace=True)
        if "verifiable" in d.columns.tolist() and "original_id" in d.columns.tolist():
            dfs[i].drop(columns=["verifiable", "original_id"], inplace=True)
        dfs[i].loc[:, "label"].replace({
            "REFUTES": 0, 
            "SUPPORTS": 1, 
            "NOT ENOUGH INFO": 2
        }, inplace=True)
        dfs[i]["evidence"] = d["evidence"].map(lambda x: [i[-1] for i in x])
        print(i, d.dtypes)        
    hf_dfs = _make_dict(*dfs)
    return hf_dfs

def _load_covidFACT_dataset(path=os.path.join("./data", "verification", "covidfact")):
    dataset_paths = [
        "covidfact_stratified_train.csv",
        "covidfact_stratified_validation.csv",
        "covidfact_stratified_test.csv",
    ]
    dfs = [pd.read_csv(os.path.join(path, p), converters={"evidence": pd.eval}) for p in dataset_paths]
    for i, d in enumerate(dfs):
        dfs[i].loc[:, "label"].replace({
            "REFUTED": 0, 
            "SUPPORTED": 1,
        }, inplace=True)

        print(i, d.dtypes)        
    hf_dfs = _make_dict(*dfs)
    return hf_dfs

def _get_tokenizer(type_):
    d = {
        "wwm_mlm": (BertTokenizer, BertConfig),
        "wwm_electra": (ElectraTokenizer, ElectraConfig),
        "shuffle_random": (BertTokenizer, BertConfig)
    }
    return d[type_]

class PretrainingCollator:
    def __init__(self, model_name, type_):
        tokenizer, config = _get_tokenizer(type_)
        self.config = config()
        self.tokenizer = tokenizer.from_pretrained(
            model_name, config=self.config
        )

    def collate_fn(self, examples):
        tokenized_examples = self.tokenizer(
            examples, padding=True, truncation=True, return_tensors="pt"
        )
        return tokenized_examples

class FinetuningCollator:
    def __init__(self, model):
        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, config=self.config)
        
    def collate_fn(self, examples):
        tokenized_examples = self.tokenizer(
            [ex for ex, _ in examples], padding=True, truncation=True, return_tensors="pt"
        )
        return tokenized_examples, torch.tensor([label for _, label in examples])

if __name__ == "__main__": # for testing
    #d =_load_CT23_dataset()
    #d =_load_CT21_dataset()
    #d = _load_FEVER_dataset()
    d = _load_climate_FEVER_dataset()
    print(d)
    