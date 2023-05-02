from transformers import set_seed, TrainingArguments, Trainer, DataCollatorWithPadding
import os
from src.parser.parser import parse_finetune_baseline
from src.utils.model_utils import load_hf_model
from src.utils.train_utils import get_compute_metrics, tokenize_dataset 
from src.utils.data_utils import get_dataset, get_num_labels
import json

def test_hf_trainer(trainer, test_dataset, save_dir):
    test_res = trainer.predict(test_dataset, metric_key_prefix="test")
    test_metrics = test_res.metrics

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as fp:
        json.dump(test_metrics, fp)

args = parse_finetune_baseline()
set_seed(args.seed)

dataset_dict, type_ = get_dataset(args.dataset)
num_labels = get_num_labels(dataset_dict)

model, tokenizer, config = load_hf_model(args.model, num_labels, use_fast=False)
dataset_dict = tokenize_dataset(dataset_dict, tokenizer, type_)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

save_dir = os.path.join("models", "baselines", type_, args.dataset, args.model)
training_args = TrainingArguments(
    output_dir=save_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    optim=args.optimizer,
    weight_decay=args.weight_decay,
    learning_rate=args.lr,
    lr_scheduler_type=args.scheduler,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_ratio=args.warmup_ratio,
    metric_for_best_model=f"eval_{args.metric}",
    load_best_model_at_end=True,
    fp16=args.fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    compute_metrics=get_compute_metrics(args.dataset),
)

trainer.train()

test_hf_trainer(
    trainer=trainer,
    test_dataset=dataset_dict["test"], 
    save_dir=save_dir
)