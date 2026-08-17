"""Tokenizer fine-tuning loop for A-share daily data."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

# Ensure repo root is on path so `from model import ...` works when run from root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import KronosTokenizer  # noqa: E402

from finetune_ashare.config_loader import AshareFinetuneConfig  # noqa: E402
from finetune_ashare.data_loader import MarketMemory  # noqa: E402
from finetune_ashare.dataset import AshareKlineDataset  # noqa: E402
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


def train_tokenizer(
    config: AshareFinetuneConfig,
    mem: MarketMemory,
    sample_refs: dict,
    tb: TBLogger | None = None,
) -> float:
    """Fine-tune KronosTokenizer; return best validation recon loss.

    Loss math matches ``finetune_csv/finetune_tokenizer.py``:
    ``(mse(z_pre, x) + mse(z, x) + bsq_loss) / 2``.
    Val metric: ``mse(z, batch_x)`` averaged over samples.
    """
    device = _resolve_device(config)

    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    tokenizer = tokenizer.to(device)

    train_dataset = AshareKlineDataset(
        mem,
        sample_refs["train"],
        lookback=config.lookback_window,
        predict=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        mode="train",
        # Dataset length is in samples; config counts optimizer steps.
        n_steps=config.n_train_steps_per_epoch * config.batch_size,
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

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    print(
        f"Tokenizer train: steps/epoch={len(train_loader)} "
        f"(configured {config.n_train_steps_per_epoch}), "
        f"batch_size={config.batch_size}, samples/epoch={len(train_dataset)}"
    )

    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.tokenizer_epochs,
        pct_start=config.onecycle_pct_start,
        div_factor=config.onecycle_div_factor,
        final_div_factor=config.onecycle_final_div_factor,
    )

    best_val_loss = float("inf")
    batch_idx_global = 0
    accumulation_steps = config.accumulation_steps

    for epoch in range(config.tokenizer_epochs):
        epoch_start = time.time()
        tokenizer.train()
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)

        for batch_idx, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)

            current_batch_total_loss = 0.0
            for j in range(accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]

                zs, bsq_loss, _, _ = tokenizer(batch_x)
                z_pre, z = zs
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2

                loss_scaled = loss / accumulation_steps
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = current_batch_total_loss / accumulation_steps
            lr = optimizer.param_groups[0]["lr"]
            if tb is not None:
                tb.add_scalar("tokenizer/train_loss", avg_loss, batch_idx_global)
                tb.add_scalar("tokenizer/lr", lr, batch_idx_global)

            if (batch_idx_global + 1) % config.log_interval == 0:
                print(
                    f"[Tokenizer Epoch {epoch + 1}/{config.tokenizer_epochs}, "
                    f"Step {batch_idx + 1}/{len(train_loader)}] "
                    f"LR: {lr:.6f}, Loss: {avg_loss:.4f}"
                )

            batch_idx_global += 1

        # Validation: seeded random subset (~n_val_loss_batches), not lex-first refs
        tokenizer.eval()
        tot_val_loss = 0.0
        val_count = 0
        val_loader = _make_val_ce_loader(
            val_dataset,
            batch_size=config.batch_size,
            n_batches=config.n_val_loss_batches,
            seed=config.seed,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = tokenizer(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)
                tot_val_loss += val_loss_item.item() * ori_batch_x.size(0)
                val_count += ori_batch_x.size(0)

        avg_val_loss = tot_val_loss / val_count if val_count > 0 else float("inf")
        if tb is not None:
            tb.add_scalar("tokenizer/val_loss", avg_val_loss, epoch)

        print(
            f"[Tokenizer Epoch {epoch + 1}/{config.tokenizer_epochs}] "
            f"val_loss={avg_val_loss:.4f} time={time.time() - epoch_start:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config.tokenizer_best_path, exist_ok=True)
            tokenizer.save_pretrained(config.tokenizer_best_path)
            print(
                f"Best tokenizer saved to {config.tokenizer_best_path} "
                f"(val_loss={best_val_loss:.4f})"
            )

    return best_val_loss
