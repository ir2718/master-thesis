from src.parser import parse_pretraining_model
from src.pretraining_dataset import get_dataset
from src.language_modeling import get_language_model
from src.model_utils import get_optimizer
from src.data_utils import collate_fn
from torch.utils.data import DataLoader

args = parse_pretraining_model()

constructor = get_dataset(args.dataset)
train_dataset = constructor(args.dataset, "train")
validation_dataset = constructor(args.dataset, "validation")

train_dataloader = DataLoader(
    train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn
)
validation_dataloader = DataLoader(
    train_dataset, batch_size=args.val_batch_size, collate_fn=collate_fn
)

model = get_language_model(args.model_type)
optimizer = get_optimizer(
    model, args.optimizer, 
    lr=args.learning_rate, 
    weight_decay=args.weight_decay
)
scheduler = get_scheduler(
    optimizer, args.scheduler,
    num_steps=args.num_steps,
    warmup_ratio=args.warmup_ratio
)
