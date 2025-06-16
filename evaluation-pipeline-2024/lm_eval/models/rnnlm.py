import torch
import torch.nn as nn
import sentencepiece as spm
from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import pytorch_lightning as pl
import torch.optim as optim
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast

CKPT_PATH = "lm_eval/models/rnn-superbpe-pretrained/rnn-superbpe-pretrained.ckpt"
TOKENIZER_PATH = "lm_eval/models/sentencepiece-tokenizers/sentencepiece-unigram/10M/sentencepiece-10M-unigram.model"
TRAINED_SUPER_BPE = "lm_eval/models/rnn-superbpe/superbpe-10M-16k-final.json"

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, hidden_size, num_layers, device):
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (h0, c0)


class LanguageModelingModule(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, learning_rate, seq_length):
        super().__init__()
        self.save_hyperparameters()
        self.model = LSTMLanguageModel(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.learning_rate = learning_rate

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def training_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.model.init_hidden(x.size(0), self.hparams.hidden_size, self.hparams.num_layers, self.device)
        logits, _ = self(x, hidden)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.model.init_hidden(x.size(0), self.hparams.hidden_size, self.hparams.num_layers, self.device)
        logits, _ = self(x, hidden)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@register_model("rnnlm")
class RNNLanguageModel(LM):
    def __init__(self):
        super().__init__()
        print("INIT WITH RNNLM")
        self.device = "cpu"

        # SENTENCEPIECE
        # self.sp = spm.SentencePieceProcessor()
        # self.sp.load(TOKENIZER_PATH)

        # SUPER BPE ANTRENAT
        # self.trained_tokenizer = Tokenizer.from_file(TRAINED_SUPER_BPE)

        # SUPER BPE PRE-ANTRENAT
        self.tokenizer = GPT2TokenizerFast.from_pretrained("UW/OLMo2-8B-SuperBPE-t180k")

        self.model = LanguageModelingModule.load_from_checkpoint(CKPT_PATH)
        self.model.eval().to(self.device)
        
        # FOR SENTENCEPIECE
        # self.vocab_size = self.sp.get_piece_size()
        # FOR TRAINED SUPERBPE
        self.vocab_size = self.tokenizer.vocab_size
        # FIR SUPERBPE PRE-TRAINED

        print(f"Loaded model at {CKPT_PATH}")
        print(f"Loaded tokenizer at {TOKENIZER_PATH}")
        print(f"Loaded superBPE trained at {TRAINED_SUPER_BPE}")

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests, disable_tqdm=False):
        results = []
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        for req in tqdm(requests, disable=disable_tqdm):
            context = req.args[0].strip()
            continuation = req.args[1].strip()
            full_text = (context + " " + continuation).strip()

            # FOR SENTENCEPIECE
            # token_ids = [self.sp.bos_id()] + self.sp.encode(full_text, out_type=int) + [self.sp.eos_id()]
            # FOR TRAINED SUPERBPE
            # encodings = self.trained_tokenizer.encode(full_text)
            # token_ids = encodings.ids
            # FOR SUPERBPE PRE-TRAINED
            token_ids = self.tokenizer.encode(full_text)

            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            input_tokens = token_tensor[:, :-1]
            target_tokens = token_tensor[:, 1:]

            with torch.no_grad():
                hidden = self.model.model.init_hidden(
                    batch_size=1,
                    hidden_size=self.model.hparams.hidden_size,
                    num_layers=self.model.hparams.num_layers,
                    device=self.device
                )
                logits, _ = self.model(input_tokens, hidden)
                logits = logits.view(-1, self.vocab_size)
                targets = target_tokens.view(-1)

                loss = loss_fn(logits, targets)
                results.append((-loss.item(), False))

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        return [ll for ll, _ in self.loglikelihood(requests, disable_tqdm)]

    def generate_until(self, requests, disable_tqdm=False):
        return ["placeholder" for _ in requests]
