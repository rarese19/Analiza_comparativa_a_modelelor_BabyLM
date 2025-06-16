from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "my-bert"

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        intermediate_size: int = 1280,
        max_position_embeddings: int = 512,
        position_bucket_size: int = 32,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # vocabulary & dimensions
        self.vocab_size                   = vocab_size
        self.hidden_size                  = hidden_size
        self.num_hidden_layers            = num_hidden_layers
        self.num_attention_heads          = num_attention_heads
        self.intermediate_size            = intermediate_size
        # positional embeddings
        self.max_position_embeddings      = max_position_embeddings
        self.position_bucket_size         = position_bucket_size
        # dropout & layer‐norm
        self.hidden_dropout_prob          = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps               = layer_norm_eps
        # optional HF feature
        self.gradient_checkpointing       = gradient_checkpointing
