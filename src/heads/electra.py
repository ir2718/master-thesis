from src.heads.base_head import BaseLMHead
from transformers import (
    AutoTokenizer,
    ElectraConfig,
    ElectraForPreTraining,
    ElectraForMaskedLM,
)
from src.utils.train_utils import batch_to_device
from torch.nn.functional import (
    binary_cross_entropy_with_logits, 
    cross_entropy
)
import random
import torch
import torch.nn as nn
import os

class ElectraLMHead(BaseLMHead):

    GEN_WEIGHT = 1.0
    DISC_WEIGHT = 50.0

    def __init__(self, multi_task_model, head_model_name, device, distributed, **kwargs):
        super().__init__(head_model_name, **kwargs)
        self.device = device
        self.gen_name = "google/electra-base-generator"
        self.config = ElectraConfig.from_pretrained(self.gen_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_name)
        self.gen_head = ElectraForMaskedLM.from_pretrained(
            self.gen_name,
            config=self.config
        ).to(self.device)

        self.disc_name = "google/electra-base-discriminator"
        self.disc_head = ElectraForPreTraining.from_pretrained(
            self.disc_name,
            config=ElectraConfig.from_pretrained(self.disc_name)
        ).discriminator_predictions.to(self.device)
        self._tie_weights(multi_task_model)
        
        self.gumbel = torch.distributions.gumbel.Gumbel(0., 1.)
        self.wwm_probability = 0.15
        self.name = "electra"
        self.distributed = distributed

        if self.distributed:
            self.gen_head = nn.DataParallel(self.gen_head)
            self.disc_head = nn.DataParallel(self.disc_head)

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
        inputs = batch_to_device(inputs, self.device)
        out = self.gen_head(**inputs).logits
        inputs = batch_to_device(inputs, "cpu")
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

    def save_head(self, step, save_path):
        torch.save(self.gen_head, os.path.join(save_path, f"mlm_head_{step}"))
        print()
        print(f"Saved the generative model to {save_path}")
        print()

        torch.save(self.disc_head, os.path.join(save_path, f"disc_head_{step}"))
        print()
        print(f"Saved the discriminator head to {save_path}")
        print()