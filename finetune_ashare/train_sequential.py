"""Sequential A-share daily finetune: tokenizer → predictor → optional 2026 backtest."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

from finetune_ashare.config_loader import AshareFinetuneConfig  # noqa: E402
from finetune_ashare.data_loader import load_market_memory  # noqa: E402
from finetune_ashare.dataset import build_sample_refs  # noqa: E402
from finetune_ashare.eval_day1 import evaluate_day1, format_e_means  # noqa: E402
from finetune_ashare.split import build_val_symbol_sets  # noqa: E402
from finetune_ashare.tb_logger import TBLogger  # noqa: E402
from finetune_ashare.train_predictor import train_predictor  # noqa: E402
from finetune_ashare.train_tokenizer import train_tokenizer  # noqa: E402


def _resolve_device(config: AshareFinetuneConfig) -> torch.device:
    if config.use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{config.device_id}")
    return torch.device("cpu")


def _dir_has_checkpoint(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def _require_tokenizer_checkpoint(config: AshareFinetuneConfig, *, reason: str) -> None:
    if _dir_has_checkpoint(config.tokenizer_best_path):
        return
    raise FileNotFoundError(
        f"tokenizer best checkpoint missing at {config.tokenizer_best_path} "
        f"({reason}). Train the tokenizer first, or unset --skip-tokenizer / "
        f"--skip-existing."
    )


def _run_backtest_checkpoint(
    *,
    label: str,
    model_path: str,
    tokenizer_path: str,
    config: AshareFinetuneConfig,
    mem,
    refs: list,
    device: torch.device,
) -> dict:
    if not _dir_has_checkpoint(model_path):
        raise FileNotFoundError(f"backtest checkpoint missing: {model_path}")
    if not _dir_has_checkpoint(tokenizer_path):
        raise FileNotFoundError(f"tokenizer checkpoint missing: {tokenizer_path}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=config.max_context,
        clip=config.clip,
    )
    metrics = evaluate_day1(
        predictor,
        mem,
        refs,
        max_samples=config.backtest_max_samples,
        seed=config.seed,
        pred_len=config.predict_window,
        T=config.day1_T,
        top_p=config.day1_top_p,
        sample_count=config.day1_sample_count,
        lookback=config.lookback_window,
    )
    row: dict = {
        "checkpoint": label,
        "path": model_path,
        "dir_acc": float(metrics["dir_acc"]),
        "n": int(metrics["n"]),
    }
    for h in range(1, int(config.predict_window) + 1):
        row[f"e{h}_mean"] = float(metrics[f"e{h}_mean"])
    return row


def run(
    config: AshareFinetuneConfig,
    *,
    skip_tokenizer: bool,
    skip_basemodel: bool,
    skip_existing: bool,
    resume_predictor: bool = False,
) -> dict:
    os.makedirs(config.base_save_path, exist_ok=True)

    print(f"Loading market memory from {config.db_path} ...")
    mem = load_market_memory(
        config.db_path,
        lookback=config.lookback_window,
        predict=config.predict_window,
    )
    print(f"Loaded {len(mem.symbols)} symbols, calendar={len(mem.calendar)} days")

    pre_calendar = [d for d in mem.calendar if d < config.backtest_start]
    blocks = build_val_symbol_sets(
        pre_calendar,
        mem.symbols,
        backtest_start=config.backtest_start,
        block_days=config.val_block_trading_days,
        val_ratio=config.val_symbol_ratio,
        seed=config.seed,
    )
    sample_refs = build_sample_refs(
        mem,
        blocks,
        lookback=config.lookback_window,
        predict=config.predict_window,
        backtest_start=config.backtest_start,
    )
    print(
        "Sample refs: "
        f"train={len(sample_refs['train'])} "
        f"val={len(sample_refs['val'])} "
        f"backtest={len(sample_refs['backtest'])}"
    )

    tb = TBLogger(config.tb_log_dir, enabled=config.use_tensorboard)
    metrics: dict = {
        "exp_name": config.exp_name,
        "tokenizer": {},
        "predictor": {},
        "backtest_2026": [],
    }

    try:
        do_tokenizer = not skip_tokenizer
        if do_tokenizer and skip_existing and _dir_has_checkpoint(config.tokenizer_best_path):
            print(f"Tokenizer best exists at {config.tokenizer_best_path}, skipping (--skip-existing)")
            do_tokenizer = False
            metrics["tokenizer"]["skipped"] = True
            metrics["tokenizer"]["reason"] = "skip_existing"

        if do_tokenizer:
            print("\n=== Tokenizer fine-tune ===")
            best_tok = train_tokenizer(config, mem, sample_refs, tb=tb)
            metrics["tokenizer"]["best_val_loss"] = float(best_tok)
            metrics["tokenizer"]["best_path"] = config.tokenizer_best_path
        elif skip_tokenizer:
            print("Skipping tokenizer (--skip-tokenizer)")
            metrics["tokenizer"]["skipped"] = True
            metrics["tokenizer"]["reason"] = "skip_tokenizer"

        do_predictor = not skip_basemodel
        both_exist = _dir_has_checkpoint(config.basemodel_best_e1_path) and _dir_has_checkpoint(
            config.basemodel_best_loss_path
        )
        if do_predictor and skip_existing and both_exist:
            print(
                "Predictor best_e1 and best_loss both exist, skipping (--skip-existing):\n"
                f"  {config.basemodel_best_e1_path}\n"
                f"  {config.basemodel_best_loss_path}"
            )
            do_predictor = False
            metrics["predictor"]["skipped"] = True
            metrics["predictor"]["reason"] = "skip_existing"

        if do_predictor or config.run_backtest:
            if not do_tokenizer:
                reason = (
                    "tokenizer training was skipped"
                    if (skip_tokenizer or (skip_existing and metrics.get("tokenizer", {}).get("skipped")))
                    else "tokenizer checkpoint unavailable"
                )
                _require_tokenizer_checkpoint(config, reason=reason)

        if do_predictor:
            print("\n=== Predictor fine-tune ===")
            if resume_predictor:
                print("Predictor resume enabled (--resume-predictor)")
            pred_out = train_predictor(
                config, mem, sample_refs, tb=tb, resume=resume_predictor
            )
            metrics["predictor"]["best_e1"] = float(pred_out["best_e1"])
            metrics["predictor"]["best_val_loss"] = float(pred_out["best_val_loss"])
            metrics["predictor"]["best_e1_path"] = config.basemodel_best_e1_path
            metrics["predictor"]["best_loss_path"] = config.basemodel_best_loss_path
        elif skip_basemodel:
            print("Skipping basemodel (--skip-basemodel)")
            metrics["predictor"]["skipped"] = True
            metrics["predictor"]["reason"] = "skip_basemodel"

        if config.run_backtest:
            print("\n=== 2026 backtest (day-1) ===")
            device = _resolve_device(config)
            checkpoint_specs = (
                ("best_e1", config.basemodel_best_e1_path),
                ("best_loss", config.basemodel_best_loss_path),
            )
            missing: list[str] = []
            for label, path in checkpoint_specs:
                if not _dir_has_checkpoint(path):
                    print(f"WARNING: backtest checkpoint missing, skipping {label}: {path}")
                    missing.append(label)
                    continue
                print(f"Evaluating {label} @ {path} ...")
                row = _run_backtest_checkpoint(
                    label=label,
                    model_path=path,
                    tokenizer_path=config.tokenizer_best_path,
                    config=config,
                    mem=mem,
                    refs=sample_refs["backtest"],
                    device=device,
                )
                metrics["backtest_2026"].append(row)
                print(
                    f"  {label}: {format_e_means(row, int(config.predict_window))} "
                    f"dir_acc={row['dir_acc']:.4f} n={row['n']}"
                )
            if not metrics["backtest_2026"]:
                raise FileNotFoundError(
                    "backtest failed: no usable predictor checkpoints "
                    f"(missing: {', '.join(missing) or 'both'})"
                )
    finally:
        tb.close()

    metrics_path = os.path.join(config.base_save_path, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {metrics_path}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sequential A-share daily finetune (tokenizer → predictor → backtest)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. finetune_ashare/configs/mainboard_daily_v1.yaml)",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer fine-tuning",
    )
    parser.add_argument(
        "--skip-basemodel",
        action="store_true",
        help="Skip predictor fine-tuning",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip tokenizer if best_model exists; skip basemodel only if both "
            "best_e1 and best_loss exist"
        ),
    )
    parser.add_argument(
        "--resume-predictor",
        action="store_true",
        help=(
            "Resume predictor from basemodel/last + last_train.pt "
            "(continues from next epoch). Implies tokenizer is already trained."
        ),
    )
    args = parser.parse_args(argv)
    config = AshareFinetuneConfig(args.config)
    skip_tokenizer = args.skip_tokenizer or args.resume_predictor
    run(
        config,
        skip_tokenizer=skip_tokenizer,
        skip_basemodel=args.skip_basemodel,
        skip_existing=args.skip_existing,
        resume_predictor=args.resume_predictor,
    )


if __name__ == "__main__":
    main()
