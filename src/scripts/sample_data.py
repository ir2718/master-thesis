import argparse
import os
import numpy as np
import pandas as pd
from src.utils.data_utils import get_dataset
from sklearn.model_selection import train_test_split

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEVER") # CT23, CT21, FEVER
    parser.add_argument("--save_path", type=str, default="./data/verification/fever/fever_train_sampled.csv")    
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--num_samples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

args = parse()
np.random.seed(args.seed)
df, _ = get_dataset(args.dataset, sampled=False)
df = df["train"].to_pandas()
y = df[[args.label_column]]
X = df.drop(columns=args.label_column)

train_fraction = args.num_samples / X.shape[0]
_, X_sample, _, y_sample = train_test_split(X, y, test_size=train_fraction, stratify=y)
final_df = pd.concat((X_sample, y_sample), axis=1)
final_df.to_csv(args.save_path, index=False)