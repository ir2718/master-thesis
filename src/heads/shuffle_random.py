from src.heads.base_head import BaseLMHead
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch
import random


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
    
    def save_head(self, step, experiment_name):
        save_path = f"./pretrained_models/mlm_head_{experiment_name}_{step}"
        torch.save(self.head, save_path)
        print()
        print(f"Saved the MLM head to {save_path}")
        print()