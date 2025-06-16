import torch
from transformers import LlamaForCausalLM

class DummyFixedLLModel(LlamaForCausalLM):
    def __init__(self, config):
        print("Dummy model with chosen log likelihood")
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
        device = original_logits.device

        random_numbers = -100 + 100 * torch.rand(batch_size, device=device)

        fake_logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)

        for i in range(batch_size):
            L_i = seq_len - 1
            R_i = random_numbers[i].item()
            # Formula f_i(alpha) = L_i * [ alpha - log(exp(alpha) + vocab_size - 1 ) ]
            def f_i(alpha: float) -> float:
                alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
                val = L_i * (alpha_t - torch.log(torch.exp(alpha_t) + vocab_size - 1))
                return val.item()

            alpha_low, alpha_high = -100.0, 100.0
            for _ in range(60):
                alpha_mid = (alpha_low + alpha_high) / 2.0
                val_mid = f_i(alpha_mid)
                if val_mid < R_i:
                    alpha_low = alpha_mid
                else:
                    alpha_high = alpha_mid

            alpha_i = (alpha_low + alpha_high) / 2.0

            for s in range(L_i):
                correct_token_id = input_ids[i, s + 1].item()
                fake_logits[i, s, correct_token_id] = alpha_i

        outputs.logits = fake_logits

        return outputs