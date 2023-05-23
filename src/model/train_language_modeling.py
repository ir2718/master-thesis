from src.parser.parser import parse_pretraining_model
from src.data.dataset import get_pretraining_dataset
from src.model.language_modeling import PretrainingModel
from src.utils.model_utils import get_optimizer, get_scheduler
from src.utils.train_utils import set_seed
from src.utils.data_utils import PretrainingCollator
from torch.utils.data import DataLoader
import torch

args = parse_pretraining_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

dataset_constructor = get_pretraining_dataset(args.dataset)
train_dataset = dataset_constructor(
    dataset_name=args.dataset, 
    split="train"
)
validation_dataset = dataset_constructor(
    dataset_name=args.dataset, 
    split="validation"
)

collator = PretrainingCollator(model_name=args.model)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collator.collate_fn, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.val_batch_size, collate_fn=collator.collate_fn)

model = PretrainingModel(
    pretrained_model_name=args.model, 
    val_steps=args.val_steps,
    log_steps=args.log_steps,
    val_log_steps=args.val_log_steps,
    experiment_name=args.experiment_name,
    modeling_type=args.modeling_type,
    distributed=args.distributed,
    device=device
)

optimizer = get_optimizer(
    model, args.optimizer,
    lr=args.lr, 
    weight_decay=args.weight_decay,
)

scheduler = get_scheduler(
    optimizer, args.scheduler,
    warmup_ratio=args.warmup_ratio,
    num_steps=args.num_steps,
)

model.train_loop(
    num_steps=args.num_steps, 
    train_loader=train_loader, 
    validation_loader=validation_loader, 
    optimizer=optimizer, 
    scheduler=scheduler,
    gradient_accumulation_steps=args.gradient_accumulation_steps
)
