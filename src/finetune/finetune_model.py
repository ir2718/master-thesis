import torch
import torch.nn as nn
from transformers import (
    ElectraConfig,
    AutoTokenizer,
    AutoConfig,
    AutoModel
)
from abc import abstractmethod
from src.utils.train_utils import _tokenize_veracity, _tokenize_checkworthiness
from src.utils.train_utils import compute_metrics, compute_metrics_multiclass
import numpy as np
from tqdm import tqdm

class FinetuneModel(nn.Module):

    MAX_GRAD_NORM = 1.0

    def __init__(self, pretrained_model_name, num_epochs, experiment_name, device, distributed):
        super().__init__()
        self.num_epochs = num_epochs
        self.experiment_name = experiment_name
        self.distributed = distributed

        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)        
        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            config=self.config,
            local_files_only=True
        ).to(device)
        if self.distributed:
            self.model = torch.nn.DataParallel(self.model)

        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=True),
            nn.Tanh(),
            nn.Dropout(p=self.config.hidden_dropout_prob, inplace=False),
        )
        if self.distributed:
            self.pooler = torch.nn.DataParallel(self.pooler)

        self.callbacks = []
        self.optimizer = None
        self.scheduler = None

    def add_final_layer(self, num_labels):
        self.cls = nn.Linear(self.config.hidden_size, num_labels)
        if self.distributed:
            self.cls = torch.nn.DataParallel(self.cls)

    def define_loss(self, num_labels):
        self.loss = nn.BCEWithLogitsLoss() if num_labels == 2 else nn.CrossEntropyLoss()

    def define_compute_metrics(self, num_labels):
        self.compute_metrics = compute_metrics if num_labels == 2 else compute_metrics_multiclass

    def define_prediction_function(self, num_labels):
        self.pred_function = (lambda x: x > 0) if num_labels == 2 else (lambda x: x.argmax(dim=1)) 

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def batch_to_device(self, batch):
        return {k:v.to(self.device) for k,v  in batch.items()}
    
    def forward(self, x):
        out = self.model(**x).last_hidden_state[:, 0, :] # get [CLS] from B, seq_len, hidden_dim
        out = self.pooler(out)
        out = self.cls(out)
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

        running_loss = []
        step = 0
        total_loss = 0.0
        for e in range(num_epochs):

            for c in self.callbacks:
                c.on_epoch_begin()

            for i, b in tqdm(enumerate(train_dataloader)):
                texts, labels = b

                tokenized_texts = self.tokenize_function(texts)
                tokenized_texts = self.batch_to_device(tokenized_texts)
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

            self.validate(train_dataloader, validation_dataloader)
            self.log_metrics(e)

            for c in self.callbacks:
                c.on_epoch_end()

        self.eval()

    def log_metrics(self, num_epoch):
        print()
        print(f"EPOCH {num_epoch}")
        print("      Train: ", {k:v[-1] for k, v in self.train_metric_dict.items()})
        print(" Validation: ", {k:v[-1] for k, v in self.val_metric_dict.items()})
        print()

    @torch.no_grad()
    def _validate_dataloader(self, dataloader):
        losses, preds, labels = [], [], []
        for i, b in enumerate(dataloader):
            texts, labels = b

            tokenized_texts = self.tokenize_function(texts)
            tokenized_texts = self.batch_to_device(tokenized_texts)
            labels_step = labels.to(self.device)

            out = self.forward(tokenized_texts)
            loss = self.loss(out, labels_step)
            losses.append(loss.cpu().detach().item())

            preds_step = self.pred_function(out)

            preds.extend(preds_step.cpu().detach().tolist())
            labels.extend(labels_step.cpu().detach().tolist())
        
        return np.mean(losses), self.compute_metrics((preds, labels))

    def validate(self, train_dataloader, validation_dataloader):
        self.eval()

        train_loss, train_metrics = self._validate_dataloader(train_dataloader)
        self.train_metric_dict["loss"].append(train_loss)
        for k in self.train_metric_dict.keys():
            self.train_metric_dict[k].append(train_metrics[k])

        val_loss, val_metrics = self._validate_dataloader(validation_dataloader)
        self.val_metric_dict["loss"].append(val_loss)
        for k in self.val_metric_dict.keys():
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