from torch.utils.data import Dataset
from src.utils.data_utils import get_dataset
from functools import partial
from tqdm import tqdm
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
    
    def __init__(self, dataset_name, split):
        data = get_dataset(dataset_name)[0][split]
        self.dataset = data.to_pandas()
        self.has_label = "label" in self.dataset.columns.tolist()

    def calculate_weights(self):
        return torch.tensor(self.dataset["label"].value_counts().tolist())

    def __len__(self):
        return self.dataset.shape[0]
    
    def get_num_labels(self):
        num_labels = self.dataset["label"].nunique()
        return num_labels

class PretrainingDatasetForVerification(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row["claim"]

class PretrainingDatasetForCheckworthiness(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row["text"]


# ADD SPACY CALCULATION FOR FINETUNING
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