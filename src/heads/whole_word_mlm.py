from src.heads.base_head import BaseLMHead
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
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

        self.config = AutoConfig.from_pretrained(head_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(head_model_name)
        self.head = AutoModelForMaskedLM.from_pretrained(
            head_model_name,
            config=self.config
        )

        self.head_model_name = self.head_model_name
        
        if head_model_name == "bert-base-cased":
            self.head = self.head.cls.to(self.device)
        elif head_model_name == "roberta-base":
            self.head = self.head.lm_head.to(self.device)

        if distributed:
            self.head = nn.DataParallel(self.head)

        self.wwm_probability = 0.15
        self.name = "mlm"

    def _group_subwords_roberta(self, input_tokens):
        cand_indexes = []

        # go over all the tokens in the input sequence
        for i, token in enumerate(input_tokens):
            if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                continue

            # group subword tokens into words as a nested list
            if len(cand_indexes) == 0 or \
                token.startswith("Ġ") or \
                token == self.tokenizer.pad_token or \
                token in [".", "!", "?", "...", ","]:
                cand_indexes.append([i])
            else:
                cand_indexes[-1].append(i)

        return cand_indexes

    def _group_subwords_bert(self, input_tokens):
        cand_indexes = []

        # go over all the tokens in the input sequence
        for i, token in enumerate(input_tokens):
            if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                continue

            # group subword tokens into words as a nested list
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        return cand_indexes
    
    def _group_subwords(self, input_tokens):
        if self.head_model_name == "bert-base-cased":
            return self._group_subwords_bert(input_tokens)
        elif self.head_model_name == "roberta-base":
            return self._group_subwords_roberta(input_tokens)

    def _whole_word_mask(self, input_tokens, max_predictions=512):
        cand_indexes = self._group_subwords(input_tokens)

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

        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    # the default loss function for masked language modeling
    def get_loss(self, x, y):
        x_view = x.view(-1, self.tokenizer.vocab_size)
        y_view = y.view(-1)

        # this is why -100 is used for tokens that dont go into loss
        mask = y_view != -100
        x_selected = x_view[mask]
        y_selected = y_view[mask]

        masked_lm_loss = cross_entropy(
            x_selected, y_selected, reduction="none"
        )
        return masked_lm_loss
    
    def save_head(self, step, save_path):
        torch.save(self.head, os.path.join(save_path, f"mlm_head_{step}"))
        print()
        print(f"Saved the MLM head to {save_path}")
        print()
