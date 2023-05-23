from src.parser.parser import parse_pretrain_metrics
from src.data.dataset import get_pretraining_dataset
from src.utils.train_utils import set_seed
from src.model.language_modeling import MultiTaskModel
from src.utils.data_utils import PretrainingCollator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import json
import os

args = parse_pretrain_metrics()
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

c = PretrainingCollator(model_name=args.model)
train_loaders = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=c.collate_fn, shuffle=True)
validation_loaders = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=c.collate_fn)

models = [os.path.join(args.model_path, i) for i in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, i))]


train_dict = {
    "acc": [],
    "f1": [],
    "recall": [],
    "precision": [],
    "loss": []
}
val_dict = {
    "acc": [],
    "f1": [],
    "recall": [],
    "precision": [],
    "loss": []
}

for m in tqdm(models):
    model = MultiTaskModel(
        pretrained_model_name=args.model, 
        val_steps=args.val_steps,
        log_steps=args.log_steps,
        val_log_steps=args.val_log_steps,
        experiment_name=args.experiment_name,
        modeling_types=args.modeling_type,
        distributed=args.distributed,
    )

    step = m.split("/")[-1]
    model.model = torch.load(os.path.join(m, f"backbone_{step}"))

    heads = [i for i in os.listdir(m) if not i.startswith("backbone")]

    for type_ in args.modeling_type:
        if type_ != "wwm_electra":
            model.heads[0].head = torch.load(os.path.join(m, heads[0]))
            num_labels = 3 if type_ == "shuffle_random" else model.tokenizer.vocab_size
        else:
            model.heads[0].disc_head = torch.load(os.path.join(m, heads[0]))
            model.heads[0].gen_head = torch.load(os.path.join(m, heads[1]))
            num_labels = 2
    

    model.eval()

    loaders = [train_loaders, validation_loaders]
    metric_type = "macro" if num_labels != 1 else "binary"

    ################ VALIDATION #####################


    for i, l in enumerate(loaders):
        losses, preds, labs = [], [], []

        with torch.no_grad():
            for i, batches in tqdm(enumerate([l])):
                val_loss_batch = 0
                for j, b in enumerate(batches):
                    outputs, labels, _ = model._forward_batch(b)

                    if len(outputs.shape) == 4 and len(labels.shape) == 3:
                        outputs = outputs[0]
                        labels = labels[0]

                    output_reshaped = outputs.view(-1, outputs.shape[-1])
                    labels_reshaped = labels.view(-1)

                    mask = labels_reshaped != -100

                    output_reshaped_selected = output_reshaped[mask]
                    labels_reshaped_selected = labels_reshaped[mask]

                    # calculate loss
                    loss = cross_entropy(
                        output_reshaped_selected, 
                        labels_reshaped_selected, 
                        reduction="none", 
                        ignore_index=-100
                    )

                    losses.extend(loss.cpu().detach().tolist())

                    preds.extend(output_reshaped_selected.argmax(dim=-1).cpu().detach().tolist())
                    labs.extend(labels_reshaped_selected.cpu().detach().tolist())
    
        if i == 0:
            train_dict["acc"].append(accuracy_score(preds, labs))
            train_dict["f1"].append(f1_score(preds, labs, average=metric_type))
            train_dict["recall"].append(recall_score(preds, labs, average=metric_type))
            train_dict["precision"].append(precision_score(preds, labs, average=metric_type))
            train_dict["loss"].append(torch.mean(torch.tensor(losses)).item())
        if i == 1:
            val_dict["acc"].append(accuracy_score(preds, labs))
            val_dict["f1"].append(f1_score(preds, labs, average=metric_type))
            val_dict["recall"].append(recall_score(preds, labs, average=metric_type))
            val_dict["precision"].append(precision_score(preds, labs, average=metric_type))
            val_dict["loss"].append(torch.mean(torch.tensor(losses)).item())

    #################################################

with open(os.path.join(args.model_path, "val_dict.json"), "w") as fp:
    json.dump(val_dict, fp)
with open(os.path.join(args.model_path, "train_dict.json"), "w") as fp:
    json.dump(train_dict, fp)