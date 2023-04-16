from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup
)
import torch

def load_hf_model(model_name, num_labels, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        **kwargs
    )
    config = AutoConfig.from_pretrained(model_name)
    return model, tokenizer, config

def get_optimizer(model, optimizer_name, **kwargs):
    MAPPING_OPTIMIZER = {
        "adamw": lambda x: AdamW(x, **kwargs)
    }
    return MAPPING_OPTIMIZER[optimizer_name](model.parameters())

def get_scheduler(optimizer, scheduler_name, num_steps, warmup_ratio):
    warmup_steps = int(num_steps * warmup_ratio)
    d = {
        "cosine": lambda x: get_cosine_schedule_with_warmup(
            x, num_warmup_steps=warmup_steps, num_training_steps=num_steps
        ),
        "linear": lambda x: get_linear_schedule_with_warmup(
            x, num_warmup_steps=warmup_steps, num_training_steps=num_steps
        )
    }
    return d[scheduler_name](optimizer)

# def add_special_tokens(model, tokenizer, type_):
#     if type_ == "checkworthiness":
#         return tokenizer
#
#     special_tokens_dict = {
#         "additional_special_tokens": ["[SEP_EVIDENCE]", "[SEP_ENT]", "[SEP_NUM]", "[SEP_EVIDENCE_SENT]"]
#     }
#     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#     model.resize_token_embeddings(len(tokenizer))
#     return tokenizer
