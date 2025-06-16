import torch
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class DummyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        print("CUSTOM MODEL!")
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        original_logits = outputs.logits
        batch_size, seq_len, vocab_size = original_logits.shape


        random_logits = torch.rand((batch_size, seq_len, vocab_size), dtype=torch.float64)
        return CausalLMOutputWithPast(logits = random_logits, loss = None)