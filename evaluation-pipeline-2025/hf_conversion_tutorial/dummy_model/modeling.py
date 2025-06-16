from __future__ import annotations

import torch
from torch import nn
from transformers.activations import gelu_new
import math
from torch import _softmax_backward_data as _softmax_backward_data
import torch.nn.functional as F
from .model_configuration import ModelConfig

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput
)

from typing import Optional, Union

# YOUR MODEL CLASSES GOES HERE



class MyModelForCausalLM():
    _keys_to_ignore_on_load_unexpected = ["lm_head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MyModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutput:
        # 1) Preluăm dimensiunile batch-ului și lungimea secvenței
        batch_size, seq_len = input_ids.size()
        vocab_size = self.config.vocab_size

        # 2) Construim tensorul constant de logits (toate valori 0.01)
        constant_value = 0.01
        logits = torch.full(
            (batch_size, seq_len, vocab_size),
            fill_value=constant_value,
            device=input_ids.device,
            dtype=torch.float32
        )

        # 3) Returnăm un CausalLMOutput care conține doar acest tensor de logits
        return CausalLMOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
