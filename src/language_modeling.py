import torch.nn as nn
import spacy
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer
)
from src.data_utils import collate_fn
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from src.pretraining_dataset import get_dataset
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod
import random
import torch

def get_LM_head(model_name, pretrain_type, **kwargs):
    d = {
        "vanilla_wwm": VanillaWholeWordMaskedLMHead,
        "object_wwm": ObjectWholeWordMaskedLMHead,
        "noun_wwm": NounWholeWordMaskedLMHead,
        "named_entity_wwm": NamedEntityWholeWordMaskedLMHead,
    }
    return d[pretrain_type](model_name, **kwargs)


class MultiTaskModel(nn.Module):
    def __init__(self, pretrained_model_name, heads, val_steps, log_steps):
        super().__init__()
        self.val_steps = val_steps
        self.log_steps = log_steps
        self.config = BertConfig.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)        
        self.model = BertModel.from_pretrained(
            pretrained_model_name,
            config=self.config,
            add_pooling_layer=False,
        )
        self.model.post_init()
        self.heads = nn.ModuleList(heads)
    
    def forward(self, x):
        out = self.model(x)
        return [h(out) for h in heads]

    def get_loss(self, losses):
        return torch.sum(losses)

    def train(num_steps, train_dataloader, validation_dataloader, optimizer, scheduler):
        step = 0
        model.zero_grad()
        for batch in train_dataloader:
            # TODO
            step += 1 
            if step % self.val_steps == 0:
                pass

    def validate(validation_dataloader):
        self.eval()
        with torch.no_grad():
            pass


class BaseLMHead(ABC, nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.head = BertOnlyMLMHead(
            config=self.config
        )
        #self.head.post_init()

    @abstractmethod
    def mask_tokens(self, inputs):
        pass

    def forward(self, x):
        return self.head(x)

    def get_loss(self, x, y):
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            x.view(-1, self.config.vocab_size), y.view(-1)
        )
        return masked_lm_loss

class BaseWholeWordMaskedLMHead(BaseLMHead):

    def __init__(self, pretrained_model_name, wwm_probability):
        super().__init__(pretrained_model_name)
        self.wwm_probability = wwm_probability

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

            # not sure what this does ?
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
        batch_input = collate_fn(
            inputs, 
            tokenizer=self.tokenizer,
        )

        mask_labels = []
        for i in inputs:
            ref_tokens = []

            for id_ in i.numpy():
                token = self.tokenizer._convert_id_to_token(id_)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = collate_fn(
            torch.tensor(mask_labels), 
            tokenizer=self.tokenizer, 
        )

        inputs, labels = self._mask_tokens(batch_input, batch_mask)
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

class VanillaWholeWordMaskedLMHead(BaseWholeWordMaskedLMHead):
    def __init__(self, pretrained_model_name, **kwargs):
        super().__init__(pretrained_model_name, **kwargs)

class ObjectWholeWordMaskedLMHead(BaseWholeWordMaskedLMHead):
    def __init__(self, pretrained_model_name, **kwargs):
        super().__init__(pretrained_model_name, **kwargs)
        self.spacy_model = spacy.load("en_core_web_lg")

    def _mask_tokens(self, inputs):
        # TODO
        pass

class NounWholeWordMaskedLMHead(BaseWholeWordMaskedLMHead):
    def __init__(self, pretrained_model_name, **kwargs):
        super().__init__(pretrained_model_name, **kwargs)
        self.spacy_model = spacy.load("en_core_web_lg")
        
    def _mask_tokens(self, inputs):
        # TODO
        pass

class NamedEntityWholeWordMaskedLMHead(BaseWholeWordMaskedLMHead):
    def __init__(self, pretrained_model_name, **kwargs):
        super().__init__(pretrained_model_name, **kwargs)
        self.spacy_model = spacy.load("en_core_web_lg")
        
    def _mask_tokens(self, inputs):
        # TODO
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
