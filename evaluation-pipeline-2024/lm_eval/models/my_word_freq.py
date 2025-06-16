from tqdm import tqdm
import json
import re
import math

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from pathlib import Path


@register_model("my_word_freq")
class MyWordFreq(LM):
    def __init__(self, language="en", score_type="mean", comma_separation="t", strict_small="t", zipf="f") -> None:
        super().__init__()
        self.language = language
        self.score_type = score_type
        self.comma_separation = comma_separation
        self.strict_small = strict_small 
        self.zipf = zipf

        base_path = Path(__file__).parent
        file_path = base_path / ("new_word_freq_comma_separated" if self.comma_separation == "t" else "new_word_freq")
        if self.strict_small == "t":
            file_path = file_path.with_suffix(".json") 
        else:
            file_path = file_path.with_name(file_path.name + "_100M.json")

        if not file_path.exists():
            raise FileNotFoundError(f"Nu am găsit fișierul JSON: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            self.word_frequencies = json.load(f)

        self.total_number_of_words = self.word_frequencies.get("total_number_of_words")
        print(f"INIT MyWordFreq language = {language}, score_type = {score_type}, comma_separation = {self.comma_separation}, strict_small = {self.strict_small}")
        print(f"Opened file at {file_path}")
    
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config = None):
        args = {x.split("=")[0]: x.split("=")[1] for x in arg_string.split(",") if "=" in x}
        return cls(
            language=args.get("language", "en"),
            score_type=args.get("score_type", "sum"),
            comma_separation=args.get("comma_separation", "t"),
            strict_small=args.get("strict_small", "t"),
            zipf=args.get("zipf", "f")
        )
    
    def word_frequency(self, word):
        return self.word_frequencies.get("words").get(word) / self.total_number_of_words if self.word_frequencies.get("words").get(word) is not None else 0.0
    
    def zipf_frequency(self, word):
        return 9 + math.log10((self.word_frequencies.get("words").get(word) / self.total_number_of_words) * pow(10, 6)) if self.word_frequencies.get("words").get(word) is not None else 0.0

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        results = []
        for req in tqdm(requests, disable=disable_tqdm):
            context = req.args[0]
            continuation = req.args[1]

            full_sentence = f"{context} {continuation}"
            if self.comma_separation == "t":
                full_sentence = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", full_sentence.lower())
            else:
                full_sentence = re.findall(r"\b[a-zA-ZÀ-ÿ']+\b", full_sentence.lower())

            frequencies = [self.word_frequency(word) if self.zipf == "f" else self.zipf_frequency(word) for word in full_sentence]

            if self.score_type == "sum":
                score = sum(frequencies)
            elif self.score_type == "mean":
                score = sum(frequencies) / len(frequencies) if frequencies else 0.0
            elif self.score_type == "max":
                score = max(frequencies) if frequencies else 0.0
            else:
                raise ValueError(f"Unknown score_type: {self.score_type}")
            
            results.append((score, False))
        
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        return [ll for ll, _ in self.loglikelihood(requests, disable_tqdm)]
    
    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        for ctx, _ in tqdm(requests, disable=disable_tqdm):
            res.append("placeholder")
            assert ctx.strip() != ""
        return res