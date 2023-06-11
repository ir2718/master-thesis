from transformers import set_seed
from src.utils.model_utils import get_optimizer, get_scheduler, get_callback
from src.finetune.finetune_model import FinetuneModel
from src.parser.parser import parse_finetune_model
from src.data.dataset import get_finetuning_dataset
from src.utils.data_utils import FinetuningCollator
from torch.utils.data import DataLoader
import torch

args = parse_finetune_model()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

dataset_constructor = get_finetuning_dataset(args.dataset)
train_dataset = dataset_constructor(
    dataset_name=args.dataset, 
    split="train"
)
validation_dataset = dataset_constructor(
    dataset_name=args.dataset, 
    split="validation"
)

num_labels = train_dataset.get_num_labels()
model = FinetuneModel(
    model=args.model,
    model_path=args.model_path, 
    num_epochs=args.num_epochs, 
    experiment_name=args.experiment_name,
    metric=args.metric,
    num_labels=num_labels,
    device=device, 
    distributed=args.distributed,
    not_pretrained=args.not_pretrained,
    loss_weights=train_dataset.calculate_weights() if args.weighted_loss else None,
    load_only_model=args.load_only_model
)

c = FinetuningCollator(
    model=model,
    model_tag=args.model,
    add_important_word_token=args.important_word_token
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=args.train_batch_size, 
    collate_fn=c.collate_fn
)
validation_dataloader = DataLoader(
    validation_dataset, 
    batch_size=args.val_batch_size, 
    collate_fn=c.collate_fn
)

optimizer = get_optimizer(
    model, args.optimizer,
    lr=args.lr, 
    weight_decay=args.weight_decay,
)
model.set_optimizer(optimizer)
scheduler = get_scheduler(
    optimizer, args.scheduler,
    warmup_ratio=args.warmup_ratio,
    num_steps=args.num_epochs * ( int(len(train_dataloader.dataset) // (args.train_batch_size * args.gradient_accumulation_steps)) + 1),
)
model.set_scheduler(scheduler)
model.train_loop(
    train_dataloader=train_dataloader, 
    validation_dataloader=validation_dataloader, 
    num_epochs=args.num_epochs, 
    gradient_accumulation_steps=args.gradient_accumulation_steps
)

test_dataset = dataset_constructor(
    dataset_name=args.dataset, 
    split="test"
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=args.val_batch_size, 
    collate_fn=c.collate_fn
)
model = model.load_best_model()
model.test(train_dataloader, prefix="train")
model.test(validation_dataloader, prefix="validation")
model.test(test_dataloader)