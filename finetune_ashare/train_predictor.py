"""Predictor fine-tuning loop with dual checkpoints (best CE loss / best e1)."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

# Ensure repo root is on path so `from model import ...` works when run from root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

from finetune_ashare.config_loader import AshareFinetuneConfig  # noqa: E402
from finetune_ashare.data_loader import MarketMemory  # noqa: E402
from finetune_ashare.dataset import AshareKlineDataset  # noqa: E402
from finetune_ashare.eval_day1 import evaluate_day1, format_e_means  # noqa: E402
from finetune_ashare.tb_logger import TBLogger  # noqa: E402


def _resolve_device(config: AshareFinetuneConfig) -> torch.device:
    if config.use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{config.device_id}")
    return torch.device("cpu")


def _make_val_ce_loader(
    dataset,
    *,
    batch_size: int,
    n_batches: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Seeded random subset of val data (~n_batches * batch_size samples)."""
    n_target = max(0, int(n_batches)) * max(1, int(batch_size))
    n_samples = min(len(dataset), n_target)
    if n_samples <= 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    sampler = RandomSampler(
        dataset, replacement=False, num_samples=n_samples, generator=gen
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _ce_loss_batch(model, tokenizer, batch_x, batch_x_stamp):
    """Teacher-forcing CE loss; tokenizer encode is detached (no grad through e1)."""
    with torch.no_grad():
        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
    token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
    token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(
        logits[0], logits[1], token_out[0], token_out[1]
    )
    return loss


def _save_predictor_resume(
    config: AshareFinetuneConfig,
    model,
    optimizer,
    scheduler,
    *,
    epoch: int,
    batch_idx_global: int,
    best_val_loss: float,
    best_e1: float,
) -> None:
    """Save end-of-epoch resume bundle (model weights + train state)."""
    os.makedirs(config.basemodel_last_path, exist_ok=True)
    model.save_pretrained(config.basemodel_last_path)
    torch.save(
        {
            "epoch": epoch,
            "batch_idx_global": batch_idx_global,
            "best_val_loss": best_val_loss,
            "best_e1": best_e1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        config.basemodel_last_train_path,
    )
    print(
        f"Resume checkpoint saved (completed epoch {epoch + 1}): "
        f"{config.basemodel_last_path} + {config.basemodel_last_train_path}"
    )


def _save_epoch_checkpoint(
    config: AshareFinetuneConfig,
    model,
    *,
    epoch: int,
    avg_train_loss: float,
    avg_val_loss: float,
    day1: dict,
    dir_acc: float,
    n_day1: int,
) -> None:
    """Save predictor weights for this epoch under basemodel/epochs/epoch_XXX/."""
    if not config.save_epoch_checkpoints:
        return

    epoch_dir = os.path.join(
        config.basemodel_epochs_dir, f"epoch_{epoch + 1:03d}"
    )
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)

    pred_len = int(config.predict_window)
    metrics = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "dir_acc": dir_acc,
        "n_day1": n_day1,
    }
    for h in range(1, pred_len + 1):
        metrics[f"e{h}_mean"] = float(day1.get(f"e{h}_mean", float("nan")))

    metrics_path = os.path.join(epoch_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Epoch checkpoint saved to {epoch_dir}")


def train_predictor(
    config: AshareFinetuneConfig,
    mem: MarketMemory,
    sample_refs: dict,
    tb: TBLogger | None = None,
    *,
    resume: bool = False,
) -> dict:
    """Fine-tune Kronos predictor; return best day-1 e1 and best val CE loss.

    Loss math matches ``finetune_csv/finetune_base_model.py`` (encode half=True,
    teacher-forcing CE). Day-1 ``e1`` is eval-only and never enters backward.

    If ``resume=True``, load ``basemodel/last`` + ``last_train.pt`` and continue
    from the next epoch.
    """
    if not os.path.isdir(config.tokenizer_best_path):
        raise FileNotFoundError(
            f"finetuned tokenizer not found: {config.tokenizer_best_path}"
        )

    device = _resolve_device(config)
    pin_memory = device.type == "cuda"
    accum = max(1, int(config.accumulation_steps))

    tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_best_path)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    start_epoch = 0
    batch_idx_global = 0
    best_val_loss = float("inf")
    best_e1 = float("inf")

    if resume:
        if not os.path.isdir(config.basemodel_last_path):
            raise FileNotFoundError(
                f"--resume-predictor set but model dir missing: {config.basemodel_last_path}"
            )
        if not os.path.isfile(config.basemodel_last_train_path):
            raise FileNotFoundError(
                f"--resume-predictor set but train state missing: "
                f"{config.basemodel_last_train_path}"
            )
        print(f"Resuming predictor from {config.basemodel_last_path}")
        model = Kronos.from_pretrained(config.basemodel_last_path)
    else:
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
    model = model.to(device)

    # Optimizer steps/epoch stay as configured; each step uses accum micro-batches.
    micro_batches_per_epoch = config.n_train_steps_per_epoch * accum
    train_dataset = AshareKlineDataset(
        mem,
        sample_refs["train"],
        lookback=config.lookback_window,
        predict=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        mode="train",
        n_steps=micro_batches_per_epoch * config.batch_size,
    )
    val_dataset = AshareKlineDataset(
        mem,
        sample_refs["val"],
        lookback=config.lookback_window,
        predict=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        mode="eval",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    print(
        f"Predictor train: opt_steps/epoch={config.n_train_steps_per_epoch}, "
        f"micro_batch={config.batch_size}, accum={accum}, "
        f"effective_batch={config.batch_size * accum}, "
        f"samples/epoch={len(train_dataset)}, micro_batches/epoch={len(train_loader)}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=config.n_train_steps_per_epoch,
        epochs=config.basemodel_epochs,
        pct_start=config.onecycle_pct_start,
        div_factor=config.onecycle_div_factor,
        final_div_factor=config.onecycle_final_div_factor,
    )

    if resume:
        state = torch.load(
            config.basemodel_last_train_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        # ``epoch`` in file is 0-based index of the last *completed* epoch.
        start_epoch = int(state["epoch"]) + 1
        batch_idx_global = int(state.get("batch_idx_global", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        best_e1 = float(state.get("best_e1", float("inf")))
        print(
            f"Resume state: next_epoch={start_epoch + 1}/{config.basemodel_epochs}, "
            f"best_val_loss={best_val_loss:.4f}, best_e1={best_e1:.6f}"
        )
        if start_epoch >= config.basemodel_epochs:
            print("All predictor epochs already completed; nothing to resume.")
            return {"best_e1": best_e1, "best_val_loss": best_val_loss}

    for epoch in range(start_epoch, config.basemodel_epochs):
        epoch_start = time.time()
        model.train()
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)

        epoch_train_loss = 0.0
        train_opt_steps = 0
        optimizer.zero_grad(set_to_none=True)
        micro_in_accum = 0

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            loss = _ce_loss_batch(model, tokenizer, batch_x, batch_x_stamp)
            (loss / accum).backward()
            epoch_train_loss += loss.item()
            micro_in_accum += 1

            if micro_in_accum < accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0
            train_opt_steps += 1

            lr = optimizer.param_groups[0]["lr"]
            if tb is not None:
                tb.add_scalar("predictor/lr", lr, batch_idx_global)

            if (batch_idx_global + 1) % config.log_interval == 0:
                print(
                    f"[Predictor Epoch {epoch + 1}/{config.basemodel_epochs}, "
                    f"Step {train_opt_steps}/{config.n_train_steps_per_epoch}] "
                    f"LR: {lr:.6f}, Loss: {loss.item():.4f}"
                )

            batch_idx_global += 1

        avg_train_loss = (
            epoch_train_loss / max(1, len(train_loader))
            if len(train_loader) > 0
            else float("inf")
        )

        # Validation CE: seeded random subset; use micro-batch size to limit VRAM
        model.eval()
        tot_val_loss = 0.0
        val_batches = 0
        val_loader = _make_val_ce_loader(
            val_dataset,
            batch_size=config.batch_size,
            n_batches=config.n_val_loss_batches,
            seed=config.seed,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                loss = _ce_loss_batch(model, tokenizer, batch_x, batch_x_stamp)
                tot_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = tot_val_loss / val_batches if val_batches > 0 else float("inf")

        if tb is not None:
            tb.add_scalar("predictor/train_loss", avg_train_loss, epoch)
            tb.add_scalar("predictor/val_loss", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config.basemodel_best_loss_path, exist_ok=True)
            model.save_pretrained(config.basemodel_best_loss_path)
            print(
                f"Best loss checkpoint saved to {config.basemodel_best_loss_path} "
                f"(val_loss={best_val_loss:.4f})"
            )

        # Day-1 eval (no gradient / not in backward)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictor = KronosPredictor(
            model,
            tokenizer,
            device=device,
            max_context=config.max_context,
            clip=config.clip,
        )
        day1 = evaluate_day1(
            predictor,
            mem,
            sample_refs["val"],
            max_samples=config.n_val_day1_samples,
            seed=config.seed,
            pred_len=config.predict_window,
            T=config.day1_T,
            top_p=config.day1_top_p,
            sample_count=config.day1_sample_count,
            lookback=config.lookback_window,
        )
        e1_mean = float(day1["e1_mean"])
        dir_acc = float(day1["dir_acc"])
        n_day1 = int(day1["n"])
        pred_len = int(config.predict_window)

        if tb is not None and n_day1 > 0 and math.isfinite(e1_mean):
            for h in range(1, pred_len + 1):
                eh = float(day1.get(f"e{h}_mean", float("inf")))
                if math.isfinite(eh):
                    tb.add_scalar(f"predictor/val_e{h}", eh, epoch)
            tb.add_scalar("predictor/val_dir_acc", dir_acc, epoch)

        if n_day1 > 0 and math.isfinite(e1_mean) and e1_mean < best_e1:
            best_e1 = e1_mean
            os.makedirs(config.basemodel_best_e1_path, exist_ok=True)
            model.save_pretrained(config.basemodel_best_e1_path)
            print(
                f"Best e1 checkpoint saved to {config.basemodel_best_e1_path} "
                f"(e1_mean={best_e1:.6f})"
            )

        print(
            f"[Predictor Epoch {epoch + 1}/{config.basemodel_epochs}] "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"{format_e_means(day1, pred_len)} dir_acc={dir_acc:.4f} n={n_day1} "
            f"time={time.time() - epoch_start:.1f}s"
        )

        _save_epoch_checkpoint(
            config,
            model,
            epoch=epoch,
            avg_train_loss=avg_train_loss,
            avg_val_loss=avg_val_loss,
            day1=day1,
            dir_acc=dir_acc,
            n_day1=n_day1,
        )
        _save_predictor_resume(
            config,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            batch_idx_global=batch_idx_global,
            best_val_loss=best_val_loss,
            best_e1=best_e1,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"best_e1": best_e1, "best_val_loss": best_val_loss}
