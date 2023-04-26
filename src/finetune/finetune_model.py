import torch
import torch.nn as nn
from transformers import (
    ElectraConfig,
    AutoTokenizer,
    AutoConfig,
    AutoModel
)
from abc import abstractmethod
from src.utils.train_utils import compute_metrics, compute_metrics_multiclass, batch_to_device
import numpy as np
from tqdm import tqdm
import os
import json

class FinetuneModel(nn.Module):

    MAX_GRAD_NORM = 1.0

    def __init__(self, model, model_path, num_epochs, experiment_name, metric, num_labels, device, distributed):
        super().__init__()
        self.num_epochs = num_epochs
        self.experiment_name = experiment_name
        self.metric = metric
        self.device = device
        self.distributed = distributed

        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)        
        self.model = torch.load(model_path).to(self.device)
        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=True),
            nn.Tanh(),
            nn.Dropout(p=self.config.hidden_dropout_prob, inplace=False),
        ).to(self.device)

        if self.distributed:
            self.model = torch.nn.DataParallel(self.model)
            self.pooler = torch.nn.DataParallel(self.pooler)

        self.num_labels = num_labels - 1 if num_labels == 2 else num_labels

        self.cls = nn.Linear(self.config.hidden_size, self.num_labels).to(self.device)
        if self.distributed:
            self.cls = torch.nn.DataParallel(self.cls)

        self.loss = nn.BCEWithLogitsLoss() if self.num_labels == 1 else nn.CrossEntropyLoss()
        self.compute_metrics = compute_metrics if self.num_labels == 1 else compute_metrics_multiclass

        self.callbacks = []
        self.optimizer = None
        self.scheduler = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks
    
    def forward(self, x):
        out = self.model(**x).last_hidden_state[:, 0, :] # get [CLS] from B, seq_len, hidden_dim
        out = self.pooler(out)
        out = self.cls(out)
        if self.num_labels == 1:
            out = out.view(-1)
        return out
    
    def init_metric_dict(self):
        self.train_metric_dict = {
            "accuracy": [], "f1": [], "precision": [], "recall": [], "loss":[]
        }
        self.val_metric_dict = {
            "accuracy": [], "f1": [], "precision": [], "recall": [], "loss":[]
        }
        self.test_metric_dict = {
            "accuracy": [], "f1": [], "precision": [], "recall": [], "loss":[]
        }

    def train_loop(self, train_dataloader, validation_dataloader, num_epochs, gradient_accumulation_steps):
        self.train()
        self.init_metric_dict()

        metric = None
        running_loss = []
        step, total_loss = 0, 0.0
        pbar = tqdm(total=num_epochs * len(train_dataloader))
        for e in range(num_epochs):

            for c in self.callbacks:
                c.on_epoch_begin()

            for i, b in enumerate(train_dataloader):
                tokenized_texts, labels = b

                if not self.distributed:
                    tokenized_texts = batch_to_device(tokenized_texts, self.device)
                    labels = labels.to(self.device)

                out = self.forward(tokenized_texts)

                loss = self.loss(out, labels) / gradient_accumulation_steps
                loss.backward()
                total_loss += loss.detach().cpu().item()

                step += 1
                if step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=FinetuneModel.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    running_loss.append(total_loss)
                    total_loss = 0

                pbar.update(1)

            self.validate(train_dataloader, validation_dataloader)
            self.log_metrics(e)

            if metric is None or max(self.val_metric_dict) == self.val_metric_dict[-1]:
                self.save_model(e)

            for c in self.callbacks:
                c.on_epoch_end()


        self.save_metrics(running_loss)
        pbar.close()
                
        self.eval()

    def save_metrics(self, running_loss):
        save_path = os.path.join("models", f"{self.experiment_name}")
        
        metrics = dict()
        for k in self.train_metric_dict.keys():
            metrics[f"train_{k}"] = self.train_metric_dict[k]

        for k in self.val_metric_dict.keys():
            metrics[f"val_{k}"] = self.val_metric_dict[k]

        metrics["running_loss"] = running_loss

        with open(os.path.join(save_path, "metrics.json"), "w") as fp:
            json.dump(metrics, fp)


    def save_model(self, epoch):
        save_path = os.path.join("./models", f"{self.experiment_name}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self, os.path.join(save_path, f"best_model_{epoch}"))
        print()
        print(f"Saved model at epoch {epoch}")
        print()

    def log_metrics(self, num_epoch):
        print()
        print(f"EPOCH {num_epoch}")
        print("      Train: ", {k:v[-1] for k, v in self.train_metric_dict.items()})
        print(" Validation: ", {k:v[-1] for k, v in self.val_metric_dict.items()})
        print()

    @torch.no_grad()
    def _validate_dataloader(self, dataloader):
        losses, outs, labels = [], [], []
        for i, b in enumerate(dataloader):
            tokenized_texts, labels_step = b

            if not self.distributed:
                tokenized_texts = batch_to_device(tokenized_texts, self.device)
                labels_step = labels_step.to(self.device)

            out = self.forward(tokenized_texts)
            loss = self.loss(out, labels_step)
            losses.append(loss.cpu().detach().item())

            outs.extend(out.cpu().detach())
            labels.extend(labels_step.cpu().detach())
        
        outs = torch.stack(outs)
        labels = torch.stack(labels)

        return np.mean(losses), self.compute_metrics((outs, labels))

    def validate(self, train_dataloader, validation_dataloader):
        self.eval()

        train_loss, train_metrics = self._validate_dataloader(train_dataloader)
        self.train_metric_dict["loss"].append(train_loss)
        for k in train_metrics.keys():
            self.train_metric_dict[k].append(train_metrics[k])

        val_loss, val_metrics = self._validate_dataloader(validation_dataloader)
        self.val_metric_dict["loss"].append(val_loss)
        for k in val_metrics.keys():
            self.val_metric_dict[k].append(val_metrics[k])

        self.train()

    @abstractmethod
    def tokenize_function(self, texts):
        pass


class FinetuneCheckworthinessModel(FinetuneModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tokenize_function(self, texts):
        return self.tokenizer(
            texts, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt"
        )
    
class FineVerificationModel(FinetuneModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tokenize_function(self, texts):
        l = [f"{i} {self.tokenizer.sep_token}" + f" {self.tokenizer.sep_token} ".join(j) for i, j in zip(texts[0], texts[1])]
        return self.tokenizer(
            l, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt"
        )