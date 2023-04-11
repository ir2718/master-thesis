from torch.utils.data import Dataset
from src.data_utils import get_dataset
from abc import ABC

def get_pretraining_dataset(dataset_name):
    d = {
        "CT23": PretrainingDatasetForCheckworthiness,
        "CT21": PretrainingDatasetForCheckworthiness,
        "FEVER": PretrainingDatasetForVerification, 
        "climate_FEVER": PretrainingDatasetForVerification, 
    }
    return d[dataset_name]

class BasePretrainingDataset(Dataset):
    def __init__(self, dataset_name, split):
        data = get_dataset(dataset_name)[0][split]
        self.dataset = data.to_pandas()
        self.has_label = "label" in self.dataset.columns.tolist()

    def __len__(self):
        return self.dataset.shape[0]

class PretrainingDatasetForVerification(BasePretrainingDataset):
    def __init__(self, dataset_name, split):
        super().__init__(dataset_name, split)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        #if self.has_label:
        #    return row["claim"], row["label"] 
        return row["claim"]

class PretrainingDatasetForCheckworthiness(BasePretrainingDataset):
    def __init__(self, dataset_name, split):
        super().__init__(dataset_name, split)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        #if self.has_label:
        #    return row["text"], row["label"]
        return row["text"]
            

# if __name__ == "__main__":
#     #data = PretrainingDatasetForCheckworthiness(dataset_name="CT23", split="train")
#     data = PretrainingDatasetForVerification(dataset_name="FEVER", split="train")
#     print(len(data))
#     print(data[2])