from .model_configuration import ModelConfig
from .modeling import BertModel, BertForMaskedLM

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

# Register your config
AutoConfig.register("my-bert", ModelConfig)

AutoModel.register(ModelConfig, BertModel)

AutoModelForMaskedLM.register(ModelConfig, BertForMaskedLM)
