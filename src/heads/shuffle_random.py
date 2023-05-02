from src.heads.base_head import BaseLMHead
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch
import os
import random


class ShuffleRandomLMHead(BaseLMHead):
    
    NUM_CLASSES = 3
        
    def __init__(self, multi_task_model, head_model_name, device, distributed, **kwargs):
        super().__init__(head_model_name, **kwargs)
        self.device = device
        self.distributed = distributed
        self.head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, ShuffleRandomLMHead.NUM_CLASSES) 
        ).to(self.device)

        if self.distributed:
            self.head = nn.DataParallel(self.head)
        
        self.manipulate_prob = 0.2
        self.name = "shuffle_random"

    def mask_tokens(self, inputs):
        input_ids_new, labels = self._mask_tokens(inputs["input_ids"])
        inputs["input_ids"] = input_ids_new

        return inputs, labels

    def _mask_tokens(self, input_ids):
        # init
        manipulated_input_ids = input_ids.clone()
        shuffle_random_mask = torch.zeros_like(manipulated_input_ids)
        
        # create shuffled input_ids matrices
        shuffled_words = manipulated_input_ids[:, torch.randperm(manipulated_input_ids.size()[1])] # row-wise shuffle
        # We need to care about special tokens: start, end, pad, mask.
        # If shuffled words fall in these, they must be put back to their original tokens.
        # This might cause the case where a shuffled token is not actually a shuffled one,
        # but because the number of special tokens is small, it does not matter and might contribute to robustness.
        special_tokens_indices = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in shuffled_words.tolist()
        ]
        special_tokens_indices = torch.tensor(special_tokens_indices, dtype=torch.bool) # -> boolean indices
        shuffled_words[special_tokens_indices] = manipulated_input_ids[special_tokens_indices]
        
        # which token is going to be shuffled?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full(manipulated_input_ids.shape, self.manipulate_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in manipulated_input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        shuffled_indices = torch.bernoulli(probability_matrix).bool() # -> boolean indices
        manipulated_input_ids[shuffled_indices] = shuffled_words[shuffled_indices]
        shuffle_random_mask[shuffled_indices] = 1

        # replace some tokens with random ones
        # this should not override shuffled tokens.
        random_indices = torch.bernoulli(torch.full(manipulated_input_ids.shape, self.manipulate_prob)).bool() & ~shuffled_indices & ~special_tokens_mask
        random_words = torch.randint(len(self.tokenizer), manipulated_input_ids.shape, dtype=torch.long)
        manipulated_input_ids[random_indices] = random_words[random_indices]
        shuffle_random_mask[random_indices] = 2

        # We only compute loss on active tokens
        shuffle_random_mask[special_tokens_mask] = -100
        return manipulated_input_ids, shuffle_random_mask
    
    def get_loss(self, x, y):
        # this is why -100 is used for tokens that dont go into loss
        shuffle_random_loss = cross_entropy(
            x.view(-1, 3), y.view(-1), ignore_index=-100
        )
        return shuffle_random_loss
    
    def save_head(self, step, save_path):
        torch.save(self.head, os.path.join(save_path, f"shuffle_random_head_{step}"))
        print()
        print(f"Saved the MLM head to {save_path}")
        print()