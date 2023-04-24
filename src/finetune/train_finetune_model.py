from transformers import set_seed
from src.parser.parser import parse_finetune_model

args = parse_finetune_model()
set_seed(args.seed)
