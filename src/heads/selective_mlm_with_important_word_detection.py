from src.heads.base_head import BaseLMHead
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from copy import deepcopy
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
import torch.nn as nn
import torch
import os
from spacy import load, prefer_gpu
import random


class SelectiveMaskedLMWithImportantWordDetectionHead(BaseLMHead):

    NUM_IMPORTANT_WORD_CLASSES = 2

    def __init__(self, multi_task_model, head_model_name, device, distributed, **kwargs):
        super().__init__(head_model_name, **kwargs)
        self.device = device

        self.config = AutoConfig.from_pretrained(head_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(head_model_name)
        self.head = AutoModelForMaskedLM.from_pretrained(
            head_model_name,
            config=self.config
        )
        
        self.spacy_model = load("en_core_web_lg")
        print(f"Spacy using gpu: {prefer_gpu()}")
        
        if head_model_name == "bert-base-cased":
            self.important_word_head = deepcopy(self.head).cls
            self.important_word_head.predictions.decoder = nn.Linear(
                in_features=self.important_word_head.predictions.decoder.in_features,
                out_features=SelectiveMaskedLMWithImportantWordDetectionHead.NUM_IMPORTANT_WORD_CLASSES,
            )
            self.head = self.head.to(self.device)
            self.important_word_head = self.important_word_head.to(self.device)

        elif head_model_name == "roberta-base":
            self.important_word_head = deepcopy(self.head).lm_head
            self.important_word_head.decoder = nn.Linear(
                in_features=self.important_word_head.decoder.in_features,
                out_features=SelectiveMaskedLMWithImportantWordDetectionHead.NUM_IMPORTANT_WORD_CLASSES,
            )
            
            self.head = self.head.to(self.device)
            self.important_word_head = self.important_word_head.to(self.device)

        if distributed:
            self.important_word_head = nn.DataParallel(self.head)
            self.head = nn.DataParallel(self.head)

        self.wwm_probability = 0.15
        self.other_probability = 0.2
        self.name = "mlm"

    def _reconstruct_text(self, tokens):
        if self.head_model_name == "bert-base-cased":
            return self._reconstruct_text_bert(tokens)
        elif self.head_model_name == "roberta-base":
            return self._reconstruct_text_roberta(tokens)

    def _reconstruct_text_roberta(self, tokens):
        reconstructed_text = ""
        for i, token in enumerate(tokens):
            if token in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                continue

            if token.startswith("Ġ"):
                token = token[1:]
                reconstructed_text += " " + token
            elif len(reconstructed_text) == 0:
                reconstructed_text += " " + token
            else:
                reconstructed_text += token

        return reconstructed_text.strip()
    
    def _reconstruct_text_bert(self, tokens):
        reconstructed_text = ""
        for i, token in enumerate(tokens):
            if token in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                continue

            if token.startswith("##"):
                token = token[2:]
                reconstructed_text += token
            else:
                reconstructed_text += " " + token

        return reconstructed_text.strip()

    def _find_words_of_interest(self, text):
        indices, words_of_interest = [], []
        
        for i, token in enumerate(self.spacy_model(text)):
            # check if its a named entity, number, predicate, subject or object
            if token.ent_type_ != "" or token.like_num \
                or (token.dep_ == "ROOT" and token.pos_ == "VERB") \
                or ("subj" in token.dep_) \
                or ("obj" in token.dep_) \
                or (token.dep_ in ["xcomp", "ccomp"]):
                indices.append(i+1) # add one because of [CLS] token
                words_of_interest.append(token.text)

        return indices, words_of_interest

    def _group_subwords_roberta(self, input_tokens, indices, words_of_interest):
        cand_indexes_selected, cand_indexes_other = [], []
        word_detection_labels = []
        len_selected = 0

        punctuation = [".", "!", "?", "...", ","]

        # go over all the tokens in the input sequence
        num_tokens = len(input_tokens)
        i = 0
        while i < num_tokens:
            token = input_tokens[i]

            if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                word_detection_labels.append(-100)
                i += 1
                continue

            if i == 1 or token.startswith("Ġ") or token.startswith("'") or token in punctuation or token == self.tokenizer.pad_token or token[0].isdigit() or token[0] == "-" or input_tokens[i-1] == "-":
                word_tokens, word_indices = [], []
                word_indices.append(i)
                word_tokens.append(token)

                j = i + 1
                while j < num_tokens:
                    next_token = input_tokens[j]
                    if not next_token.startswith("Ġ") and not next_token.startswith("'") and next_token not in punctuation \
                        and next_token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token] \
                        and not next_token[0].isdigit() and token != "-" and next_token != "-":
                        word_indices.append(j)
                        word_tokens.append(next_token)
                        j += 1
                    else:
                        break

                full_word = "".join(word_tokens)
                if full_word[0] == "Ġ":
                    full_word = full_word[1:]

                if full_word in words_of_interest:
                    cand_indexes_selected.append(word_indices)
                    len_selected += len(word_tokens)
                    word_detection_labels.extend([1 for i in word_indices])
                    indices = indices[1:]
                else:
                    cand_indexes_other.append(word_indices)
                    word_detection_labels.extend([0 for i in word_indices])
                
                i = j

        self.all_word_detection_labels.append(word_detection_labels)

        return cand_indexes_selected, cand_indexes_other, len_selected
    
    def _group_subwords_bert(self, input_tokens, indices, words_of_interest):
        cand_indexes_selected, cand_indexes_other = [], []
        word_detection_labels = []
        len_selected = 0

        # go over all the tokens in the input sequence
        num_tokens = len(input_tokens)
        i = 0
        while i < num_tokens:
            token = input_tokens[i]

            if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                word_detection_labels.append(-100)
                i += 1
                continue

            if not token.startswith("##"):
                word_tokens, word_indices = [], []
                word_indices.append(i)
                word_tokens.append(token)

                j = i + 1
                while j < num_tokens:
                    next_token = input_tokens[j]
                    if next_token.startswith("##"):
                        word_indices.append(j)
                        word_tokens.append(next_token[2:])
                        j += 1
                    else:
                        break

                full_word = "".join(word_tokens)

                if full_word in words_of_interest:
                    cand_indexes_selected.append(word_indices)
                    len_selected += len(word_tokens)
                    word_detection_labels.extend([1 for i in word_indices])
                    indices = indices[1:]
                else:
                    cand_indexes_other.append(word_indices)
                    word_detection_labels.extend([0 for i in word_indices])
                
                i = j

        self.all_word_detection_labels.append(word_detection_labels)

        return cand_indexes_selected, cand_indexes_other, len_selected

    def _group_subwords(self, input_tokens, indices, words_of_interest):
        if self.head_model_name == "bert-base-cased":
            return self._group_subwords_bert(input_tokens, indices, words_of_interest)
        elif self.head_model_name == "roberta-base":
            return self._group_subwords_roberta(input_tokens, indices, words_of_interest)

    def _whole_word_mask(self, input_tokens, max_predictions=512):
        reconstructed_text = self._reconstruct_text(input_tokens)
        indices, words_of_interest = self._find_words_of_interest(reconstructed_text)

        cand_indexes_selected, cand_indexes_other, len_selected = self._group_subwords(input_tokens, indices, words_of_interest,)

        random.shuffle(cand_indexes_selected)
        random.shuffle(cand_indexes_other)

        # if there is more than 80% * wwm_probability available selected tokens
        # use 20% random tokens to make sure it doesnt do catastrophic forgetting
        num_tokens = len(input_tokens) 
        perc_selected = len_selected / num_tokens
        if perc_selected > self.wwm_probability  * (1 - self.other_probability): 
            k = int(len(input_tokens) * self.wwm_probability  * (1 - self.other_probability))
            cand_indexes_other.extend(cand_indexes_selected[k:])
            cand_indexes_selected = cand_indexes_selected[:k]

        selected_to_predict = int(round( (num_tokens * self.wwm_probability) * (1 - self.other_probability)))
        other_to_predict = int(round( (num_tokens * self.wwm_probability) * self.other_probability))

        num_to_predict_selected = min(
            max_predictions, 
            max(1, int(round(selected_to_predict)))
        )

        num_to_predict_other = min(
            max_predictions, 
            max(1, int(round(other_to_predict)))
        )

        masked_lms = []
        covered_indexes = set()
        
        # go over all the indices that make up words
        for index_set in (cand_indexes_selected):

            # if theres already too many masked words
            if len(masked_lms) >= num_to_predict_selected:
                break

            # if adding this word creates too many masked words 
            if len(masked_lms) + len(index_set) > num_to_predict_selected:
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


        # go over all the indices that make up words
        for index_set in (cand_indexes_other):

            # if theres already too many masked words
            if len(masked_lms) >= num_to_predict_other:
                break

            # if adding this word creates too many masked words 
            if len(masked_lms) + len(index_set) > num_to_predict_other:
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
        self.all_word_detection_labels = []
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

        self.all_word_detection_labels = torch.tensor(self.all_word_detection_labels)

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

    def forward(self, x):
        self.important_word_head_out = self.important_word_head(x)
        out = self.head(x)
        return out

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

        x_view_word = x

        word_detection_loss = 0

        return masked_lm_loss + 0
    
    def save_head(self, step, save_path):
        torch.save(self.head, os.path.join(save_path, f"mlm_head_{step}"))
        print()
        print(f"Saved the MLM head to {save_path}")
        print()
