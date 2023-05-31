import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraForPreTraining,
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
from src.utils.train_utils import batch_to_device
from src.heads.electra import ElectraLMHead
from src.heads.whole_word_mlm import WholeWordMaskedLMHead
from src.heads.shuffle_random import ShuffleRandomLMHead
from src.heads.selective_mlm import SelectiveMaskedLMHead
from src.heads.selective_mlm_with_important_word_detection import SelectiveMaskedLMWithImportantWordDetectionHead
from tqdm import tqdm
import os
import torch
import json
from abc import abstractmethod

def get_LM_head(model_name, pretrain_type):
    d = {
        "wwm_mlm": WholeWordMaskedLMHead,
        "wwm_electra": ElectraLMHead,
        "shuffle_random": ShuffleRandomLMHead,
        "selective_mlm": SelectiveMaskedLMHead,
        "selective_mlm_with_word_detection": SelectiveMaskedLMWithImportantWordDetectionHead,
    }
    return d[pretrain_type]


class PretrainingModel(nn.Module):

    MAX_GRAD_NORM = 1.0

    def __init__(
        self, pretrained_model_name, val_steps=500, log_steps=200, val_log_steps=100, experiment_name="experiment", modeling_type="wwm_mlm", device=0, distributed=True
    ):
        super().__init__()
        self.val_steps = val_steps
        self.log_steps = log_steps
        self.val_log_steps = val_log_steps
        self.experiment_name = experiment_name
        self.distributed = distributed


        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, 
            config=self.config
        )    

        if pretrained_model_name == "google/electra-base-discriminator":
            self.model = ElectraForPreTraining.from_pretrained(
                pretrained_model_name,
                config=self.config,
            ).electra.to(device)
        else:   
            self.model = AutoModel.from_pretrained(
                pretrained_model_name,
                config=self.config,
                add_pooling_layer=False,
            ).to(device)

        lm_constr = get_LM_head(model_name=pretrained_model_name, pretrain_type=modeling_type)
        h = lm_constr(
            self,
            pretrained_model_name,
            tokenizer=self.tokenizer,
            config=self.config,
            device=device,
            distributed=distributed
        )
        self.head = h
        self.device = device

        if self.distributed:
            self.model = nn.DataParallel(self.model)

    def forward(self, x):
        out = self.model(x)
        return self.head(out)

    def forward_head(self, inputs):
        # do on cpu
        x, y = self.head.mask_tokens(inputs)

        # switch to gpu
        inputs = batch_to_device(inputs, self.device)
        y = y.to(self.device)

        out = self.model(**inputs).last_hidden_state
        out = self.head(out)

        return x, y, out

    def _forward_batch(self, inputs):
        _, labels, outputs = self.forward_head(inputs)
        loss = self.head.get_loss(outputs, labels)
        return outputs, labels, loss
    
    def init_metric_dict(self):
        self.train_metrics = {
            "loss": []
        }
        self.val_metrics = {
            "loss": []
        }

    def aggregate_losses(self, loss):
        if isinstance(loss, tuple) != 0:
            return torch.sum(torch.stack([l.mean() for l in loss]))
        return loss.mean()

    def train_loop(self, num_steps, train_loader, validation_loader, optimizer, scheduler, gradient_accumulation_steps):        

        self.gradient_accumulation_steps = gradient_accumulation_steps

        print()
        print("------------ Training ------------")
        print(f"Starting training for {num_steps} steps . . .")
        print()

        self.init_metric_dict()
        self.train()

        num_epochs = (num_steps * gradient_accumulation_steps) // len(train_loader) + 1
        effective_step, step = 0, 0

        pbar = tqdm(total=num_steps)

        for e in range(num_epochs):
            total_loss = 0

            for batch in train_loader:
                outputs, labels, loss = self._forward_batch(batch)
                loss = self.aggregate_losses(loss) / gradient_accumulation_steps
                loss.backward()
                total_loss += loss

                step += 1
                if step % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.parameters(), PretrainingModel.MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    effective_step += 1
                    pbar.update(1)

                    self.train_metrics["loss"].append(
                        total_loss.cpu().detach().item()
                    )
                    total_loss = 0

                if (effective_step + 1) % self.log_steps == 0 and step % effective_step == 0:
                    self.log(effective_step, self.train_metrics["loss"][-1])

                # calculate val loss and acc
                if ((effective_step + 1) % self.val_steps == 0 or (effective_step + 1) == num_steps) and step % effective_step == 0:
                    self.val_step(validation_loader)
                    self.save_model_and_heads(effective_step)

                    if (effective_step + 1) == num_steps:
                        pbar.close()
                        self.save_metrics()
                        return

    def save_metrics(self):
        save_path = os.path.join("pretrained_models", f"{self.experiment_name}")
        
        metrics = dict()
        for k in self.train_metrics.keys():
            metrics[f"train_{k}"] = self.train_metrics[k]

        for k in self.val_metrics.keys():
            metrics[f"val_{k}"] = self.val_metrics[k]

        metrics_info = {k:v for k, v in metrics.items()}
        metrics_info["val_steps"] = self.val_steps

        with open(os.path.join(save_path, "metrics.json"), "w") as fp:
            json.dump(metrics_info, fp)


    def val_step(self, validation_loader):
        self.eval()

        print()
        print("------------ Validation ------------")
        print("Checking if model is in train or eval:")
        print(f"self.model.training: {self.model.training}")
        print(f"self.head.training: {self.head.training}")
        print("------------------------------------")
        print()

        val_losses = []
        with torch.no_grad():
            for batch in tqdm(validation_loader):
                outputs, labels, loss = self._forward_batch(batch)

                # in case the loss consists of multiple components, calculate the mean for each component and then sum all components
                # this is a must for electra pretraining
                if isinstance(loss, tuple):
                    if len(val_losses) == 0:
                        for i, k in enumerate(loss):
                            val_losses.append(k.detach().cpu().tolist())
                    else:
                        for i, k in enumerate(loss):
                            val_losses[i].extend(k.detach().cpu().tolist())
                else:
                    val_losses.extend(loss.detach().cpu().tolist())

        if isinstance(loss, tuple):
            val_loss = torch.sum(torch.stack([torch.mean(torch.tensor(l)) for l in val_losses]))
        else:
            val_loss = torch.mean(torch.tensor(val_losses))

        print(f"Val loss: {val_loss}")

        self.train()
        
        print()
        print("------------------------------------")
        print("Checking if model is in train or eval:")
        print(f"self.model.training: {self.model.training}")
        print(f"self.head.training: {self.head.training}")
        print("------------------------------------")
        print()

        self.val_metrics["loss"].append(val_loss.item())

    def log(self, step, loss):
        print()
        print(" Step {}: ".format(step))
        print("    Loss - {:.5f}".format(loss))
        print()

    def save_model_and_heads(self, step):
        save_path = os.path.join("pretrained_models", f"{self.experiment_name}", f"{step}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model, os.path.join(save_path, f"backbone_{step}"))
        print()
        print(f"Saved the backbone to {save_path}")
        print()
        self.head.save_head(step, save_path)
