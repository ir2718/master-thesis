import argparse

def parse_pretraining_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--modeling_type", nargs="+", type=str, default=["wwm_mlm"])
    parser.add_argument("--dataset", type=str, default="CT23") # CT23, CT21, FEVER, climate_FEVER
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--val_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--val_log_steps", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=str, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    return args

def parse_finetune_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--dataset", type=str, default="CT23") # CT23, CT21, FEVER, climate_FEVER
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--metric", type=str, default="f1") # f1, accuracy
    parser.add_argument("--warmup_ratio", type=str, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def parse_finetune_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/wwm_mlm_bert_CT23/1499/backbone_wwm_mlm_bert_CT23_1499")
    parser.add_argument("--dataset", type=str, default="CT23") # CT23, CT21, FEVER, climate_FEVER
    parser.add_argument("--modeling_type", type=str, default="wwm_mlm")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--metric", type=str, default="f1") # f1, accuracy
    parser.add_argument("--warmup_ratio", type=str, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    return args