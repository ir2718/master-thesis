from transformers import set_seed, TrainingArguments, Trainer, DataCollatorWithPadding
import os
from src.parser.parser import parse_finetune_baseline
from src.utils.model_utils import load_hf_model
from src.utils.train_utils import get_compute_metrics, tokenize_dataset 
from src.utils.data_utils import get_dataset, get_num_labels

args = parse_finetune_baseline()
set_seed(args.seed)

dataset_dict, type_ = get_dataset(args.dataset)
num_labels = get_num_labels(dataset_dict)

model, tokenizer, config = load_hf_model(args.model, num_labels, use_fast=False)
dataset_dict = tokenize_dataset(dataset_dict, tokenizer, type_)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


print(tokenizer.decode(dataset_dict["train"]["input_ids"][0]))
print(dataset_dict["train"]["claim"][0])
print(dataset_dict["train"]["evidence"][0])
print()
print(tokenizer.decode(dataset_dict["validation"]["input_ids"][0]))
print(dataset_dict["validation"]["claim"][0])
print(dataset_dict["validation"]["evidence"][0])
print()
print(tokenizer.decode(dataset_dict["test"]["input_ids"][0]))
print(dataset_dict["test"]["claim"][0])
print(dataset_dict["test"]["evidence"][0])
print()
print(num_labels)

training_args = TrainingArguments(
    output_dir=os.path.join("./models", type_, args.dataset, args.model),
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
    fp16=True
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