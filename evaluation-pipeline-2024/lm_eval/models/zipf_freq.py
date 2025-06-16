from tqdm import tqdm
from wordfreq import zipf_frequency

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("zipf_baseline")
class ZipfBaseline(LM):
    def __init__(self, language="en", score_type="sum") -> None:
        super().__init__()
        self.language = language
        self.score_type = score_type
        print(f"INIT ZipfBaseline language={language}, score_type={score_type}")

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = {x.split("=")[0]: x.split("=")[1] for x in arg_string.split(",") if "=" in x}
        return cls(
            language=args.get("language", "en"),
            score_type=args.get("score_type", "sum"),
        )

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        results = []
        for req in tqdm(requests, disable=disable_tqdm):
            context = req.args[0]
            continuation = req.args[1]

            full_sentence = f"{context} {continuation}".strip()
            words = full_sentence.split()

            zipf_scores = [zipf_frequency(w, self.language) for w in words]

            if self.score_type == "sum":
                final_score = sum(zipf_scores)
            elif self.score_type == "mean":
                final_score = sum(zipf_scores) / len(zipf_scores) if zipf_scores else 0.0
            elif self.score_type == "max":
                final_score = max(zipf_scores) if zipf_scores else 0.0
            else:
                raise ValueError(f"Unknown score_type: {self.score_type}")

            results.append((final_score, False))

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        return [ll for ll, _ in self.loglikelihood(requests, disable_tqdm)]

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        for ctx, _ in tqdm(requests, disable=disable_tqdm):
            res.append("placeholder")
            assert ctx.strip() != ""
        return res
