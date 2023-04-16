import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraForPreTraining,
    AutoTokenizer
)
from src.heads.electra import ElectraLMHead
from src.heads.whole_word_mlm import WholeWordMaskedLMHead
from src.heads.shuffle_random import ShuffleRandomLMHead
from tqdm import tqdm
import torch

def get_LM_head(model_name, pretrain_type):
    d = {
        "wwm_mlm": WholeWordMaskedLMHead,
        "wwm_electra": ElectraLMHead,
        "shuffle_random": ShuffleRandomLMHead,
    }
    return d[pretrain_type]


class MultiTaskModel(nn.Module):
    def __init__(
        self, pretrained_model_name, val_steps, log_steps, val_log_steps, experiment_name, modeling_types, device
    ):
        super().__init__()
        self.val_steps = val_steps
        self.log_steps = log_steps
        self.val_log_steps = val_log_steps
        self.experiment_name = experiment_name

        if pretrained_model_name == "google/electra-base-discriminator":
            self.config = ElectraConfig.from_pretrained(pretrained_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)        
            self.model = ElectraForPreTraining.from_pretrained(
                pretrained_model_name,
                config=self.config,
            ).electra.to(device)
        else:
            self.config = BertConfig.from_pretrained(pretrained_model_name)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)        
            self.model = BertModel.from_pretrained(
                pretrained_model_name,
                config=self.config,
                add_pooling_layer=False,
            ).to(device)
        
        heads = []
        for type_ in modeling_types:
            lm_constr = get_LM_head(model_name=pretrained_model_name, pretrain_type=type_)
            h = lm_constr(
                self,
                pretrained_model_name,
                tokenizer=self.tokenizer,
                config=self.config,
            )
            heads.append(h)

        self.heads = nn.ModuleList([h.to(device) for h in heads])
        self.device = device
    
    def forward(self, x):
        out = self.model(x)
        return [h(out) for h in self.heads]

    def forward_head(self, inputs, head):
        # do on cpu
        x, y = head.mask_tokens(inputs)

        # switch to gpu
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        y = y.to(self.device)
        

        out = self.model(**inputs)[0]
        out = head(out)

        return x, y, out

    def get_loss(self, losses):
        return torch.mean(losses)

    def _forward_batch(self, inputs):
        outputs, labels, losses = [], [], []
        for h in self.heads:
            _, y, out = self.forward_head(inputs, h)
            l = h.get_loss(out, y)
            
            losses.append(l)
            outputs.append(out)
            labels.append(y)


        losses = torch.stack(losses)
        outputs = torch.stack(outputs)
        labels = torch.stack(labels)

        batch_loss = self.get_loss(losses)

        return outputs, labels, batch_loss

    def _calculate_accuracy(self, d, x, y):
        for i, (xi, yi) in enumerate(zip(x, y)):
            yi = yi.reshape(-1)
    
            # in case of multiclass problem such as mlm  
            if len(xi.shape) != 2:
                preds = torch.argmax(xi.view(-1, xi.shape[-1]), dim=-1)
            else:
                preds = xi.view(-1) > 0 # unnormalized logits!
            
            equal = (preds == yi)[yi != -100].float()
            d[self.heads[i].get_name()].append(equal.mean().item())
        return d


    def train_loop(self, num_steps, train_loaders, validation_loaders, optimizer, scheduler, gradient_accumulation_steps):        
        train_losses, val_losses = [], []
        train_accs = {h.get_name():[] for h in self.heads}
        val_accs = {h.get_name():[] for h in self.heads}

        print()
        print("------------ Training ------------")
        print(f"Starting training for {num_steps} steps . . .")
        print()

        self.train()
        num_epochs = num_steps // len(train_loaders[0]) + 1
        step = 0
        for e in range(num_epochs):
            for i, batches in tqdm(enumerate(zip(*train_loaders))):
                # goes over multiple batches in case multiple loaders are used
                train_loss_batch, train_acc_batch = 0, 0
                for b in batches:
                    outputs, labels, train_loss = self._forward_batch(b)
                    train_loss_batch += train_loss.mean() 
                    self._calculate_accuracy(train_accs, outputs, labels)

                train_loss_batch.backward()
                train_losses.append(train_loss_batch.cpu().detach().item())

                step += 1 
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # calculate train loss and acc
                if (step % self.log_steps) == 0:
                    self.log(step, train_loss_batch, train_accs)

                # calculate val loss and acc
                if step % self.val_steps == 0:
                    val_loss_batch = self.val_step(val_accs, validation_loaders)
                    val_losses.append(val_loss_batch)
                    
                    # if its the best model so far
                    if val_losses[-1] == min(val_losses):
                        self.save_model_and_heads(step)
                    
    def val_step(self, val_accs, validation_loaders):
        self.eval()

        print()
        print("------------ Validation ------------")
        print("Checking if model is in train or eval:")
        print(f"self.model.training: {self.model.training}")
        print(f"self.heads.training: {[h.training for h in self.heads]}")
        print("------------------------------------")
        print()

        val_losses, val_accs_curr = [], {h.get_name():[] for h in self.heads}
        step = 0
        with torch.no_grad():
            for i, batches in tqdm(enumerate(zip(*validation_loaders))):
                val_loss_batch = 0
                for b in batches:
                    outputs, labels, val_loss = self._forward_batch(b)
                    val_loss_batch += val_loss.mean()
                    self._calculate_accuracy(val_accs_curr, outputs, labels)

                val_losses.append(val_loss_batch.cpu().detach().item())
                
                step += 1
                if (step % self.val_log_steps) == 0:
                    self.log(step, val_loss_batch, val_accs_curr)

        self.train()
        
        print()
        print("------------------------------------")
        print("Checking if model is in train or eval:")
        print(f"self.model.training: {self.model.training}")
        print(f"self.heads.training: {[h.training for h in self.heads]}")
        print("------------------------------------")
        print()

        for k in val_accs.keys():
            val_accs[k].append(torch.tensor(val_accs_curr[k]).mean().item())

        return torch.tensor(val_losses).mean().item()

    def log(self, step, loss, accs):
        print()
        print(" Step {}: ".format(step))
        print("    Loss - {:.5f}".format(loss.item()))
        for k in accs.keys():
            print("    Acc  - {:.5f} => {}".format(
                torch.tensor(accs[k][-1]).mean().item(), k)
            )
        print()

    def save_model_and_heads(self, step):
        save_path = f"./pretrained_models/backbone_{self.experiment_name}_{step}"
        torch.save(self.model, save_path)
        print()
        print(f"Saved the backbone to {save_path}")
        print()
        for h in self.heads:
            h.save_head(step, self.experiment_name)



if __name__ == "__main__":
    #dataset = get_dataset("CT23")
    #dataloader = DataLoader(
    #    dataset, batch_size=2, collate_fn=collate_fn
    #)
    model_name = "bert-base-cased"
    pretrain_type = "vanilla_wwm"
    head = get_LM_head(model_name, pretrain_type, wwm_probability=0.15)
    #model = MultiTaskModel(
    #    model_name,
    #    [head],
    #    10, 100
    #)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    inputs, labels = head.mask_tokens(
        tokenizer(
            "Today might be a good day for pretraining.", 
            return_tensors="pt"
        )["input_ids"]
    )

    print(tokenizer.convert_ids_to_tokens(inputs[0]))
    print(tokenizer.convert_ids_to_tokens(labels[0]))
    #model.train(2, dataloader, dataloader, 1, 1)
