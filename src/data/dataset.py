from torch.utils.data import Dataset
from src.utils.data_utils import get_dataset
import torch
import spacy

def get_pretraining_dataset(dataset_name):
    d = {
        "CT23": PretrainingDatasetForCheckworthiness,
        "CT21": PretrainingDatasetForCheckworthiness,
        "FEVER": PretrainingDatasetForVerification, 
        "climate_FEVER": PretrainingDatasetForVerification, 
    }
    return d[dataset_name]

def get_finetuning_dataset(dataset_name):

    d = {
        "CT23": FinetuningCheckworthinessDataset,
        "CT21": FinetuningCheckworthinessDataset,
        "FEVER": FinetuningVerficationDataset, 
        "climate_FEVER": FinetuningVerficationDataset, 
    }
    return d[dataset_name]

class BaseDataset(Dataset):
    
    def __init__(self, dataset_name, split, calculate_spacy_features=True):
        data = get_dataset(dataset_name)[0][split]
        self.dataset = data.to_pandas()
        self.has_label = "label" in self.dataset.columns.tolist()

        if calculate_spacy_features:
            spacy.prefer_gpu()
            self.spacy_model = spacy.load("en_core_web_trf")


    def __len__(self):
        return self.dataset.shape[0]
    
    def get_num_labels(self):
        num_labels = self.dataset["label"].nunique()
        return num_labels

class PretrainingDatasetForVerification(BaseDataset):

    def __init__(self, dataset_name, split):
        super().__init__(dataset_name, split)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row["claim"]

class PretrainingDatasetForCheckworthiness(BaseDataset):

    def __init__(self, dataset_name, split):
        super().__init__(dataset_name, split)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row["text"]

class FinetuningCheckworthinessDataset(BaseDataset):

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row["text"], torch.tensor(row["label"], dtype=torch.float32)

class FinetuningVerficationDataset(BaseDataset):

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return (row["claim"], row["evidence"]), torch.tensor(row["label"], dtype=torch.float32)

# if __name__ == "__main__":
#     #data = PretrainingDatasetForCheckworthiness(dataset_name="CT23", split="train")
#     data = PretrainingDatasetForVerification(dataset_name="FEVER", split="train")
#     print(len(data))
#     print(data[2])