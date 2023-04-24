from src.heads.base_head import BaseLMHead
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
)
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch
import os
import random


class WholeWordMaskedLMHead(BaseLMHead):

    def __init__(self, multi_task_model, head_model_name, device, distributed, **kwargs):
        super().__init__(head_model_name, **kwargs)
        self.device = device

        self.config = BertConfig.from_pretrained(head_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(head_model_name)
        self.head = BertForMaskedLM.from_pretrained(
            head_model_name,
            config=self.config
        ).cls.to(self.device)
        
        if distributed:
            self.head = nn.DataParallel(self.head)

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
        labels[~masked_indices] = -100 # only compute the loss for masked tokens

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
    
    def save_head(self, step, save_path):
        torch.save(self.head, os.path.join(save_path, f"mlm_head_{step}"))
        print()
        print(f"Saved the MLM head to {save_path}")
        print()
