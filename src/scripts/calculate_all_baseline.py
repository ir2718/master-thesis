from transformers import DataCollatorWithPadding, Trainer
from src.parser.parser import parse_finetune_baseline_metrics
from src.utils.train_utils import set_seed, tokenize_dataset, get_compute_metrics, test_hf_trainer
from src.utils.data_utils import get_dataset, get_num_labels
from src.utils.model_utils import load_hf_model
import os
import json

args = parse_finetune_baseline_metrics()
set_seed(args.seed)

dataset_dict, type_ = get_dataset(args.dataset)
num_labels = get_num_labels(dataset_dict)

model, tokenizer = load_hf_model(args.model, num_labels, use_fast=False, not_pretrained=args.no_pretrained)
dataset_dict = tokenize_dataset(dataset_dict, tokenizer, type_)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
