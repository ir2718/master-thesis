import torch.nn as nn
import spacy
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    ElectraConfig,
    ElectraTokenizer,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    AutoTokenizer
)
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, gumbel_softmax
from src.pretraining_dataset import get_dataset
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
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
            d[self.heads[i].get_name()].append(equal.mean())
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
                    val_loss_batch = self.val_step(validation_loaders)
                    val_losses.append(val_loss_batch)
                    
                    # if its the best model so far
                    if val_losses[-1] == min(val_losses):
                        self.save_model_and_heads(step)
                    
    def val_step(self, validation_loaders):
        self.eval()

        print()
        print("------------ Validation ------------")
        print("Checking if model is in train or eval:")
        print(f"self.model.training: {self.model.training}")
        print(f"self.heads.training: {[h.training for h in self.heads]}")
        print("------------------------------------")
        print()

        val_losses, val_accs = [], {h.get_name():[] for h in self.heads}
        step = 0
        with torch.no_grad():
            for i, batches in tqdm(enumerate(zip(*validation_loaders))):
                val_loss_batch, val_acc_batch = 0, 0
                for b in batches:
                    outputs, labels, val_loss = self._forward_batch(b)
                    val_loss_batch += val_loss.mean()
                    self._calculate_accuracy(val_accs, outputs, labels)

                val_losses.append(val_loss_batch.cpu().detach().item())
                
                step += 1
                if (step % self.val_log_steps) == 0:
                    self.log(step, val_loss_batch, val_accs)

        self.train()
        
        print()
        print("------------------------------------")
        print("Checking if model is in train or eval:")
        print(f"self.model.training: {self.model.training}")
        print(f"self.heads.training: {[h.training for h in self.heads]}")
        print("------------------------------------")
        print()

        return torch.tensor(val_losses).mean().item()

    def log(self, step, loss, accs):
        print()
        print(" Step {}: ".format(step))
        print("    Loss - {:.5f}".format(loss.item()))
        for k in accs.keys():
            print("    Acc  - {:.5f} => {}".format(accs[k][-1].mean(), k))
        print()

    def save_model_and_heads(self, step):
        save_path = f"./pretrained_models/backbone_{self.experiment_name}_{step}"
        torch.save(self.model, save_path)
        print()
        print(f"Saved the backbone to {save_path}")
        print()
        for h in self.heads:
            h.save_head(step)


class BaseLMHead(ABC, nn.Module):
    def __init__(self, head_model_name, config, tokenizer):
        super().__init__()
        self.head_model_name = head_model_name
        self.config = config
        self.tokenizer = tokenizer

    def get_name(self):
        return self.name

    # inputs is a dict of input_ids, attention_mask, token_type_ids...
    @abstractmethod
    def mask_tokens(self, inputs):
        pass

    def forward(self, x):
        return self.head(x)

    @abstractmethod
    def get_loss(self, x, y):
        pass

    @abstractmethod
    def save_head(self, steps):
        pass


###################################################################################################



class WholeWordMaskedLMHead(BaseLMHead):

    def __init__(self, multi_task_model, head_model_name, **kwargs):
        super().__init__(head_model_name, **kwargs)
        self.config = BertConfig.from_pretrained(head_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(head_model_name)
        self.head = BertForMaskedLM.from_pretrained(
            head_model_name,
            config=self.config
        ).cls
        
        self.wwm_probability = 0.15
        self.name = "vanilla_wwm"

    def _whole_word_mask(self, input_tokens, max_predictions=512):
        cand_indexes = []

        # go over all the tokens in the input sequence
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            # group subword tokens into words as a nested list
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # randomize the order of the nested subwords (whole words)
        # calculate number of predictions given probability
        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions, 
            max(1, int(round(len(input_tokens) * self.wwm_probability)))
        )
        
        masked_lms = []
        covered_indexes = set()
        
        # go over all the indices that make up words
        for index_set in cand_indexes:

            # if theres already too many masked words
            if len(masked_lms) >= num_to_predict:
                break

            # if adding this word creates too many masked words 
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue

            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            
            if is_any_index_covered:
                continue
            
            # otherwise add the word to masking procedure
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        
        return [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

    def mask_tokens(self, inputs):
        mask_labels = []
        for i in inputs["input_ids"]:
            ref_tokens = []

            for id_ in i.cpu().numpy():
                token = self.tokenizer._convert_id_to_token(id_)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(mask_labels), 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )

        input_ids_new, labels = self._mask_tokens(inputs["input_ids"], batch_mask)
        inputs["input_ids"] = input_ids_new

        return inputs, labels

    def _mask_tokens(self, inputs, mask_labels):
        # copy the existing inputs and treat as labels
        labels = inputs.clone()

        # take care of special tokens such as [CLS] and [SEP]
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(x, already_has_special_tokens=True) for x in labels
        ]

        # in case there a special token has been chosen for masking, do not use it
        probability_matrix = mask_labels
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        # make sure [PAD] tokens arent used in masking
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100 # only compute the loss for masked tokens, -100 = [PAD]

        # 80% of the time, replace the tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace the token with a random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long # generate random word idx
        )
        inputs[indices_random] = random_words[indices_random] 

        # 10% of the time do nothing
        return inputs, labels

    # the default loss function for masked language modeling
    def get_loss(self, x, y):
        # this is why -100 is used for tokens that dont go into loss
        masked_lm_loss = cross_entropy(
            x.view(-1, self.tokenizer.vocab_size), y.view(-1), ignore_index=-100
        )
        return masked_lm_loss
    
    def save_head(self, step):
        save_path = f"./pretrained_models/mlm_head_{self.experiment_name}_{step}"
        torch.save(self.model, save_path)
        print()
        print(f"Saved the MLML head to {save_path}")
        print()



###################################################################################################



class ElectraLMHead(BaseLMHead):

    GEN_WEIGHT = 1.0
    DISC_WEIGHT = 50.0

    def __init__(self, multi_task_model, head_model_name, **kwargs):
        super().__init__(head_model_name, **kwargs)
        gen_name = "google/electra-base-generator"
        self.config = ElectraConfig.from_pretrained(gen_name)
        self.tokenizer = AutoTokenizer.from_pretrained(gen_name)
        self.gen_head = ElectraForMaskedLM.from_pretrained(
            gen_name,
            config=self.config
        )

        disc_name = "google/electra-base-discriminator"
        self.disc_head = ElectraForPreTraining.from_pretrained(
            disc_name,
            config=ElectraConfig.from_pretrained(disc_name)
        ).discriminator_predictions
        self._tie_weights(multi_task_model)
        
        self.gumbel = torch.distributions.gumbel.Gumbel(0., 1.)
        self.wwm_probability = 0.15
        self.name = "electra"

    def _tie_weights(self, multi_task_model):
        multi_task_model.model.embeddings = self.gen_head.electra.embeddings
        self.gen_head.generator_lm_head.weight = self.gen_head.electra.embeddings.word_embeddings.weight

    def forward(self, x):
        return self.disc_head(x)

    def _whole_word_mask(self, input_tokens, max_predictions=512):
        cand_indexes = []

        # go over all the tokens in the input sequence
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            # group subword tokens into words as a nested list
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # randomize the order of the nested subwords (whole words)
        # calculate number of predictions given probability
        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions, 
            max(1, int(round(len(input_tokens) * self.wwm_probability)))
        )
        
        masked_lms = []
        covered_indexes = set()
        
        # go over all the indices that make up words
        for index_set in cand_indexes:

            # if theres already too many masked words
            if len(masked_lms) >= num_to_predict:
                break

            # if adding this word creates too many masked words 
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue

            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            
            if is_any_index_covered:
                continue
            
            # otherwise add the word to masking procedure
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        
        return [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

    def _mask_tokens(self, inputs, mask_labels):
        # copy the existing inputs and treat as labels
        labels = inputs.clone()

        # take care of special tokens such as [CLS] and [SEP]
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(x, already_has_special_tokens=True) for x in labels
        ]

        # in case there a special token has been chosen for masking, do not use it
        probability_matrix = mask_labels
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        # make sure [PAD] tokens arent used in masking
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100 # only compute the loss for masked tokens, -100 = [PAD]

        # 85% of the time, replace the tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 15% of the time do nothing
        return inputs, labels, indices_replaced

    def _generate_new_tokens_logits(self, inputs):
        inputs = {k: v.to(self.gen_head.device) for k, v in inputs.items()}
        out = self.gen_head(**inputs)[0]
        inputs = {k: v.cpu() for k, v in inputs.items()}
        return inputs, out

    def mask_tokens(self, inputs):
        mask_labels = []
        for i in inputs["input_ids"]:
            ref_tokens = []

            for id_ in i.cpu().numpy():
                token = self.tokenizer._convert_id_to_token(id_)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(mask_labels), 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )

        # mask 85% * 15% tokens tokens
        inputs_masked, labels, indices_replaced = self._mask_tokens(
            inputs["input_ids"], batch_mask
        )
        # possible optimization: filter all examples where labels in row are all false
        # these dont have to go through the generator at all

        
        # create example for discriminator
        input_to_gen = inputs.copy()
        input_to_gen["input_ids"] = inputs_masked

        # run generator forward pass
        inputs_new, gen_out = self._generate_new_tokens_logits(input_to_gen)
        self.gen_out = gen_out.cpu()
        self.gen_labels = labels.clone()

        tokens_new, labels_new = self._sample_tokens(inputs_new, labels, indices_replaced)

        # get mask for [PAD] tokens and fill labels with -100 where [PAD] token was
        # these tokens aren't calculated in loss
        padding_mask = torch.ones_like(labels_new).bool()
        padding_mask[inputs_new["input_ids"] != self.tokenizer.pad_token_id] = False
        labels_new.masked_fill_(padding_mask.bool(), -100)
        
        return inputs_new, labels_new

    def _sample_tokens(self, inputs_new, labels, indices_replaced):
        
        with torch.no_grad():
            # sample new tokens using gumbel softmax
            gen_out_masked = self.gen_out[indices_replaced, :]
            preds = (gen_out_masked + self.gumbel.sample(gen_out_masked.shape)).argmax(dim=-1)
            generated = inputs_new["input_ids"].clone()

            generated[indices_replaced] = preds
            
            # produce labels for discriminator
            # loss isnt calculated on tokens that the generator guessed
            is_replaced = indices_replaced.clone()
            is_replaced[indices_replaced] = (preds != labels[indices_replaced])

        return generated, is_replaced.float()

    def get_loss(self, x, y):
        # calculate generator loss only over tokens different from the original
        gen_out_flat = self.gen_out.view(-1, self.tokenizer.vocab_size)
        gen_label_flat = self.gen_labels.view(-1)
        gen_loss = cross_entropy(gen_out_flat, gen_label_flat, ignore_index=-100)

        # calculate discriminator loss only over masked tokens
        mask = y != -100
        disc_loss = binary_cross_entropy_with_logits(x[mask], y[mask])

        # the original paper uses gen_weight=1.0 and disc_weight=50.0
        total_loss = ElectraLMHead.GEN_WEIGHT * gen_loss + ElectraLMHead.DISC_WEIGHT * disc_loss
        
        return total_loss

    def save_head(self, step):
        save_path_gen = f"./pretrained_models/mlm_head_{self.experiment_name}_{step}"
        torch.save(self.gen_head, save_path_gen)
        print()
        print(f"Saved the generative model to {save_path_gen}")
        print()

        save_path_disc = f"./pretrained_models/disc_head_{self.experiment_name}_{step}"
        torch.save(self.gen_head, save_path_disc)
        print()
        print(f"Saved the discriminator head to {save_path_disc}")
        print()



###################################################################################################


class ShuffleRandomLMHead(BaseLMHead):
    
    NUM_CLASSES = 3
        
    def __init__(self, multi_task_model, head_model_name, **kwargs):
        super().__init__(head_model_name, **kwargs)
        
        # {0: original, 1: shuffled, 2: random}
        self.head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, ShuffleRandomLMHead.NUM_CLASSES) 
        )
        
        self.shuffle_probability = 0.2
        self.random_probability = 0.2
        self.name = "shuffle_random"

    def _whole_word_mask(self, input_tokens, max_predictions=512):
        cand_indexes = []

        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions, 
            max(1, int(round(len(input_tokens) * (self.shuffle_probability + self.random_probability))))
        )
        
        masked_lms = []
        covered_indexes = set()
        
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break

            if len(masked_lms) + len(index_set) > num_to_predict:
                continue

            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            
            if is_any_index_covered:
                continue
            
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        
        return [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

    def mask_tokens(self, inputs):
        mask_labels = []
        for i in inputs["input_ids"]:
            ref_tokens = []

            for id_ in i.cpu().numpy():
                token = self.tokenizer._convert_id_to_token(id_)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(mask_labels), 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        
        input_ids_new, labels = self._mask_tokens(inputs["input_ids"], batch_mask)
        inputs["input_ids"] = input_ids_new

        return inputs, labels

    def _mask_tokens(self, inputs, mask_labels):
        # copy the existing inputs and treat as labels
        labels = inputs.clone()

        # take care of special tokens such as [CLS] and [SEP]
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(x, already_has_special_tokens=True) for x in labels
        ]

        # in case there a special token has been chosen for masking, do not use it
        probability_matrix = mask_labels
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        # make sure [PAD] tokens arent used in masking
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        masked_indices = probability_matrix.bool()

        shuffle_prob = self.shuffle_probability / (self.shuffle_probability + self.random_probability)
        for i, row_mask  in enumerate(masked_indices):
            ones_indices = torch.where(row_mask)[0]
            
            # 10% of the time replace randomly with a token in the sequence
            shuffle_indices_mask = torch.bernoulli(torch.full((ones_indices.shape[0],), shuffle_prob)).bool()
            shuffle_indices = ones_indices[shuffle_indices_mask]
            permuted_shuffle_indices = shuffle_indices[torch.randperm(shuffle_indices.shape[0])]
            inputs[i, shuffle_indices] = inputs[i, permuted_shuffle_indices]
    
            # 10% of the time replace with a random token
            random_indices_mask = ~shuffle_indices_mask
            random_indices = ones_indices[random_indices_mask]
            inputs[i, random_indices] = torch.randint(len(self.tokenizer), random_indices.shape, dtype=torch.long)

            # make labels such that -100 is where special tokens are
            # 0 -> original
            # 1 -> shuffled
            # 2 -> random
            labels[i] = torch.tensor(special_tokens_mask[i])
            labels[i, labels[i] == 1] = -100
            labels[i, random_indices] = 2
            labels[i, shuffle_indices] = 1
            
        return inputs, labels
    
    def get_loss(self, x, y):
        # this is why -100 is used for tokens that dont go into loss
        shuffle_random_loss = cross_entropy(
            x.view(-1, 3), y.view(-1), ignore_index=-100
        )
        return shuffle_random_loss
    
    def save_head(self, step):
        pass



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
