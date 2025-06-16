# coding=utf-8

import os
import argparse
from tqdm import tqdm
from itertools import count
from tokenizers import Tokenizer
from statistics import mean
import json
import math
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, seed_everything
from dataset import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger

# only rank 0 logs to wandb
if int(os.environ.get("SLURM_PROCID", "0")) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="../data/train_100M_tokenized.bin", type=Path)
    parser.add_argument("--valid_path", default="../data/dev_100M_tokenized.bin", type=Path)
    parser.add_argument("--name", default="hybrid_100M", type=str)
    parser.add_argument("--wandb_project", default="YOUR_WANDB_PROJECT_NAME", type=str)
    parser.add_argument("--wandb_entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--config_file", default="../configs/base.json", type=Path)
    parser.add_argument("--tokenizer_path", default="../tokenizers/tokenizer_100M.json", type=Path)
    parser.add_argument("--output_dir", default="../model_checkpoints", type=Path)
    parser.add_argument("--checkpoint_filename", default=None, type=Path)
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--hybrid_numerator", default=15, type=int)
    parser.add_argument("--hybrid_denominator", default=16, type=int)
    parser.add_argument("--seq_length", default=128, type=int)
    parser.add_argument("--local_batch_size", default=256, type=int)
    parser.add_argument("--global_batch_size", default=256, type=int)
    parser.add_argument("--batch_reduction", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--max_steps", default=31_250 // 2, type=int)
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--validate_every", default=1_000, type=int)
    parser.add_argument("--validation_steps", default=1, type=int)
    parser.add_argument("--log_stats_every", default=100, type=int)
    parser.add_argument("--warmup_proportion", default=0.016, type=float)
    parser.add_argument("--cooldown_proportion", default=0.016, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--mask_p_start", default=0.3, type=float)
    parser.add_argument("--mask_p_end", default=0.15, type=float)
    parser.add_argument("--mask_random_p", default=0.1, type=float)
    parser.add_argument("--mask_keep_p", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer_eps", default=1e-8, type=float)
    parser.add_argument("--optimizer_beta1", default=0.9, type=float)
    parser.add_argument("--optimizer_beta2", default=0.98, type=float)
    parser.add_argument("--max_gradient", default=2.0, type=float)
    parser.add_argument("--mixed_precision", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--n_special_tokens", default=16, type=int)
    parser.add_argument("--z_loss_weight", default=1e-4, type=float)
    parser.add_argument("--token_weighted_loss", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    args.name = f"{args.name}_{args.hybrid_numerator}_{args.hybrid_denominator}"
    args.output_path = (args.output_dir / args.name).with_suffix(".bin")
    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available(), "CUDA is required"
    seed_everything(args.seed)
    args.device = torch.device("cuda")

    print(f"Training for {args.max_steps:,} steps")
    print(
        "Total tokens = "
        f"{args.max_steps:,} steps × "
        f"{args.global_batch_size:,} batch × "
        f"{args.seq_length:,} length = "
        f"{args.max_steps * args.global_batch_size * args.seq_length:,}"
    )

    args.vocab_size = tokenizer.get_vocab_size()

    # only rank 0 should init wandb
    if is_main_process():
        wandb.init(name=args.name, project=args.wandb_project, entity=args.wandb_entity)


def load_config(args):
    with args.config_file.open() as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    # count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        wandb.config.update(vars(args))
        wandb.config.update({"n_params": n_params})

    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params:,}\n", flush=True)

    model.to(args.device)

    # separate weight decay
    no_decay = ["bias", "layer_norm"]
    decay_params = [
        p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
    ]
    no_decay_params = [
        p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
    ]
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # choose optimizer
    if args.optimizer in ("adam", "adamw"):
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    else:
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )

    # cosine schedule with warmup + cooldown
    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1,
    )

    # EMA copy
    ema_model = copy.deepcopy(getattr(model, "module", model))
    for p in ema_model.parameters():
        p.requires_grad = False

    # optionally resume
    global_step, start_epoch = 0, 0
    if args.checkpoint_filename:
        ckpt = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt.get("masked_epoch", ckpt.get("epoch", 0))

    return model, ema_model, optimizer, scheduler, global_step, start_epoch


def get_batch(dataloader, device, global_step, is_train=False):
    if is_train and hasattr(dataloader._dataset, "set_global_step"):
        dataloader._dataset.set_global_step(global_step)

    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [
        t.pin_memory().to(device, non_blocking=True) for t in batch
    ]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    return input_ids, attention_mask, target_ids, mask_p.mean()


@torch.no_grad()
def validation_epoch(model, valid_dataloader, masked_epoch, causal_epoch, args, commit=False):
    model.eval()
    losses, accs = [], []
    it = iter(valid_dataloader)
    ids, mask, tgt, _ = get_batch(it, args.device, 0, is_train=False)

    for _ in range(args.validation_steps):
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, acc, _, _ = model(ids, mask, tgt)

        losses.append(loss.item())
        accs.append(acc.item())

        if len(losses) < args.validation_steps:
            ids, mask, tgt, _ = get_batch(it, args.device, 0, is_train=False)

    if is_main_process():
        wandb.log(
            {
                "masked_epoch": masked_epoch,
                "causal_epoch": causal_epoch,
                "validation/loss": mean(losses),
                "validation/accuracy": mean(accs) * 100.0,
                "validation/perplexity": math.exp(mean(losses)),
            },
            commit=commit,
        )


def save_checkpoint(model, ema_model, optimizer, scheduler, step, m_epoch, c_epoch, args):
    if not is_main_process():
        return

    core = model.module if hasattr(model, "module") else model
    torch.save(core.state_dict(), args.output_path)
    torch.save(ema_model.state_dict(), args.output_path.with_name(args.output_path.stem + "_ema.bin"))

    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": step,
            "masked_epoch": m_epoch,
            "causal_epoch": c_epoch,
        },
        args.output_path.with_name(args.output_path.stem + "_state_dict.bin"),
    )


def load_dataset(args, tokenizer, epoch, global_step, prev_loader, mode="masked"):
    ratio = args.hybrid_numerator / args.hybrid_denominator if mode == "masked" else 1 - (
        args.hybrid_numerator / args.hybrid_denominator
    )
    train_seed = args.seed + epoch

    progress = (global_step + 1) / args.max_steps
    if progress >= 0.9:
        seq_length = args.seq_length * 4
        gb = args.global_batch_size // 4
    elif progress >= 0.7:
        seq_length = args.seq_length * 2
        gb = args.global_batch_size // 2
    else:
        seq_length = args.seq_length
        gb = args.global_batch_size

    # reload only if first time or seq length changed
    if prev_loader is None or prev_loader.dataset.seq_length != seq_length:
        DatasetClass = MaskedDataset if mode == "masked" else CausalDataset
        data = DatasetClass(args.train_path, tokenizer, args, seq_length, None, None)
        data.show_random_item(tokenizer)
    else:
        data = prev_loader.dataset

    current_global_bs = int(gb / args.batch_reduction * (1 - progress) + gb * progress + 0.5)
    local_bs = max(1, int(current_global_bs * ratio + 0.5))

    loader = DataLoader(
        data,
        shuffle=True,
        batch_size=local_bs,
        num_workers=0,
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )
    return loader


def init_datasets(args, tokenizer):
    args.ratio = args.hybrid_numerator / args.hybrid_denominator
    # scale global batch once
    args.global_batch_size = int(args.global_batch_size / args.batch_reduction + 0.5)

    masked_loader = None
    if args.ratio > 0:
        masked_loader = DataLoader(
            MaskedDataset(args.train_path, tokenizer, args, args.seq_length, None, None),
            shuffle=True,
            batch_size=max(1, int(args.global_batch_size * args.ratio + 0.5)),
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
            drop_last=True,
            pin_memory=True,
        )

    causal_loader = None
    if args.ratio < 1:
        causal_loader = DataLoader(
            CausalDataset(args.train_path, tokenizer, args, args.seq_length, None, None),
            shuffle=True,
            batch_size=max(1, int(args.global_batch_size * (1 - args.ratio) + 0.5)),
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
            drop_last=True,
            pin_memory=True,
        )

    valid_loader = DataLoader(
        ValidationDataset(args.valid_path, tokenizer, args),
        shuffle=False,
        batch_size=args.local_batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
        drop_last=True,
        pin_memory=True,
    )

    return masked_loader, causal_loader, valid_loader


def training_epoch(model, ema_model, train_loader, valid_loader, optimizer, scheduler, global_step, epoch, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    max_steps = min(len(train_loader), (args.max_steps - global_step) * args.batch_reduction)
    it = iter(train_loader)

    total_loss = total_acc = total_z = total_grad = 0.0

    # first batch
    ids, mask, tgt, maskp = get_batch(it, args.device, global_step, is_train=True)

    for _ in range(max_steps):
        # accumulate across micro-batches
        full_ids, full_mask, full_tgt = ids, mask, tgt
        acc_steps = full_ids.size(1) / args.local_batch_size

        for start in range(0, full_ids.size(1), args.local_batch_size):
            b_ids = full_ids[:, start : start + args.local_batch_size]
            b_mask = full_mask[:, start : start + args.local_batch_size]
            b_tgt = full_tgt[:, start : start + args.local_batch_size]

            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                with ModelLogger(enable=(global_step % 100 == 0), module=model):
                    loss, acc, z_loss, _ = model(b_ids, b_mask, b_tgt)

            weight = (b_ids.size(1) / args.local_batch_size) / acc_steps
            (loss + args.z_loss_weight * z_loss).mul(weight).backward()

            total_loss += loss.detach() * weight
            total_acc += acc * weight
            total_z += z_loss * weight

        # next batch
        if _ < max_steps - 1:
            ids, mask, tgt, maskp = get_batch(it, args.device, global_step, is_train=True)

        # clip, step, schedule
        total_grad += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # EMA
        with torch.no_grad():
            for q, k in zip(model.module.parameters(), ema_model.parameters()):
                k.data.mul_(args.ema_decay).add_((1 - args.ema_decay) * q.detach().data)

        # checkpoint
        if global_step % args.save_every == 0:
            save_checkpoint(model, ema_model, optimizer, scheduler, global_step, epoch, 0, args)

        # validate
        if (global_step + 1) % args.validate_every == 0:
            validation_epoch(model, valid_loader, epoch, 0, args)

        # log
        if is_main_process():
            wandb.log(
                {
                    "masked_epoch": epoch,
                    "train/loss": total_loss.item(),
                    "train/accuracy": total_acc.item() * 100.0,
                    "train/z_loss": total_z.item(),
                    "stats/grad_norm": total_grad,
                    "stats/mask_p": maskp.item(),
                },
                commit=True,
            )

        global_step += 1
        if global_step >= args.max_steps:
            break

    return global_step


def training(model, ema_model, masked_loader, causal_loader, valid_loader, optimizer, scheduler, global_step, args):
    # hybrid or single-mode training
    if 0 < args.ratio < 1:
        # interleaved masked/causal loop (omitted for brevity; mirror above)
        pass

    elif args.ratio == 1:
        for epoch in count():
            global_step = training_epoch(
                model, ema_model, masked_loader, valid_loader, optimizer, scheduler, global_step, epoch, args
            )
            masked_loader = load_dataset(args, Tokenizer.from_file(str(args.tokenizer_path)), epoch, global_step, masked_loader, mode="masked")
            if global_step >= args.max_steps:
                break

    else:  # causal-only
        for epoch in count():
            global_step = training_epoch(
                model, ema_model, causal_loader, valid_loader, optimizer, scheduler, global_step, epoch, args
            )
            causal_loader = load_dataset(args, Tokenizer.from_file(str(args.tokenizer_path)), epoch, global_step, causal_loader, mode="causal")
            if global_step >= args.max_steps:
                break

    return global_step


if __name__ == "__main__":
    args = parse_arguments()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))

    setup_training(args, tokenizer)
    model, ema_model, optimizer, scheduler, global_step, start_epoch = prepare_model_and_optimizer(args)

    masked_loader, causal_loader, valid_loader = init_datasets(args, tokenizer)
    global_step = training(
        model, ema_model,
        masked_loader, causal_loader, valid_loader,
        optimizer, scheduler,
        global_step, args
    )

    save_checkpoint(model, ema_model, optimizer, scheduler, global_step, start_epoch, 0, args)
    validation_epoch(model, valid_loader, start_epoch, 0, args, commit=True)
