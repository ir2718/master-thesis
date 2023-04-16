from abc import ABC, abstractmethod
import torch.nn as nn

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
