from tqdm import tqdm
import kenlm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from pathlib import Path

@register_model("kenlm")
class KenLM(LM):
    def __init__(self, order="3", strict="t"):
        super().__init__()
        self.order = order
        self.strict = strict
        base_path = Path(__file__).parent

        file_path = base_path / f"{self.order}-gram-100M.binary" if self.strict == "t" else base_path / f"{self.order}-gram.binary"
        self.model = kenlm.LanguageModel(str(file_path))

        print(f"INIT kenlm model with order {self.order} at location {str(file_path)}")

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config = None):
        args = {x.split("=")[0]: x.split("=")[1] for x in arg_string.split(",") if "=" in x}
        return cls(
            order=args.get("order", "3"),
            strict=args.get("strict", "t")
        )

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        results = []
        for req in tqdm(requests, disable=disable_tqdm):
            context = req.args[0]
            continuation = req.args[1]

            full_sentence = f"{context} {continuation}"

            results.append((self.model.score(full_sentence), False))

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        return [ll for ll, _ in self.loglikelihood(requests, disable_tqdm)]

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        for ctx, _ in tqdm(requests, disable=disable_tqdm):
            res.append("placeholder")
            assert ctx.strip() != ""

        return res