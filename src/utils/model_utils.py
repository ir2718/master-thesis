from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from src.finetune.callbacks import (
    FrozenHeadCallback, 
    GradualUnfreezingHeadCallback,
    GradualUnfreezingHeadCallback,
)
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup
)
import os
import json

def get_callback(name, model_tag):
    if name == "frozen":
        return FrozenHeadCallback

    if model_tag in ["bert-base-cased", "roberta-base", "google/electra-base-discriminator"]:
        return GradualUnfreezingHeadCallback

def _find_best_hf_ckpt(path):
    files = [os.path.join(path, i) for i in os.listdir()]
    folders_only = [i for i in files if os.path.isdir(i)]
    last_ckpt = sorted([int(i.split()[-1]) for i in folders_only])[-1]
    last_ckpt_folder = [i for i in folders_only if i.endswith(f"-{last_ckpt}")][0]
    best_ckpt = json.loads(open(os.path.join(last_ckpt_folder, "trainer_state.json")).read())["best_model_checkpoint"]
    # TODO
    pass

def load_hf_model_from_disk(model_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        **kwargs
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
    )
    return model, tokenizer

def load_hf_model(model_name, num_labels, not_pretrained, **kwargs):
    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        **kwargs
    )
    
    if not_pretrained:
        config.num_labels = num_labels
        model = AutoModelForSequenceClassification.from_config(
            config=config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )

    return model, tokenizer, config

def get_optimizer(model, optimizer_name, **kwargs):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": kwargs["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    MAPPING_OPTIMIZER = {
        "adamw": lambda x: AdamW(x, **kwargs)
    }
    return MAPPING_OPTIMIZER[optimizer_name](optimizer_grouped_parameters)

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
