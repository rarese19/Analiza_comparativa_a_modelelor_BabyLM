from tqdm import tqdm
from wordfreq import zipf_frequency

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("constant_model")
class ConstantModel(LM):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        results = []
        for req in tqdm(requests, disable=disable_tqdm):


            results.append((0.5, False)) 

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        return [ll for ll, _ in self.loglikelihood(requests, disable_tqdm)]

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        for ctx, _ in tqdm(requests, disable=disable_tqdm):
            res.append("placeholder")
            assert ctx.strip() != ""
        return res
