import pandas as pd
import os
from datasets import DatasetDict, Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig
)
import spacy
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
        "CT23_1B_checkworthy_english_train.tsv",
        "CT23_1B_checkworthy_english_dev.tsv",
        "CT23_1B_checkworthy_english_test_gold.tsv",
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

    hf_dfs = _make_dict(*dfs)
    return hf_dfs

class PretrainingCollator:
    def __init__(self, model_name):
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.config
        )

    def collate_fn(self, examples):
        tokenized_examples = self.tokenizer(
            examples, padding=True, truncation=True, return_tensors="pt"
        )
        return tokenized_examples

class FinetuningCollator:
    def __init__(self, model, model_tag, add_important_word_token=False):
        self.config = AutoConfig.from_pretrained(model_tag)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag, config=self.config)
        
        self.add_important_word_token = add_important_word_token
        if self.add_important_word_token:
            self.special_token = "[IMPORTANT]"
            special_tokens_dict = {'additional_special_tokens': [self.special_token]}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(self.tokenizer))
            print(f"Spacy using GPU: {spacy.require_gpu()}")
            self.spacy_model = spacy.load("en_core_web_lg")

    def collate_fn(self, examples):
        if isinstance(examples[0][0], tuple):
            exs = []
            for x, _ in examples:
                if self.add_important_word_token:
                    x = [
                        self.find_and_add_important_word_tokens(x[0]), 
                        [self.find_and_add_important_word_tokens(i) for i in x[1]]
                    ]
                s = x[0] + f" {self.tokenizer.sep_token} " + f" {self.tokenizer.sep_token} ".join(x[1])
                exs.append(s)
        else:
            exs = []
            for x, _ in examples:
                if self.add_important_word_token:
                    x = self.find_and_add_important_word_tokens(x)
                exs.append(x)

        tokenized_examples = self.tokenizer(
            exs, padding=True, truncation=True, return_tensors="pt"
        )
        return tokenized_examples, torch.tensor([label for _, label in examples])
    
    def find_and_add_important_word_tokens(self, x):
        tokenized = self.spacy_model(x)
        retokenized = []
        for token in tokenized:
            retokenized.append(token.text)
            if token.ent_type_ != "" or token.like_num \
                or (token.dep_ == "ROOT" and token.pos_ == "VERB") \
                or ("subj" in token.dep_) \
                or ("obj" in token.dep_) \
                or (token.dep_ in ["xcomp", "ccomp"]):
                retokenized.append(self.special_token)
        final_str = " ".join(retokenized)
        return final_str