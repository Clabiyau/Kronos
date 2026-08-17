"""Compare pretrained vs best_e1 (~epoch 6) vs best_loss (~epoch 10) on 2026 holdout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer

from finetune_ashare.config_loader import AshareFinetuneConfig
from finetune_ashare.data_loader import load_market_memory
from finetune_ashare.dataset import build_sample_refs
from finetune_ashare.eval_day1 import evaluate_day1, format_e_means
from finetune_ashare.split import build_val_symbol_sets

CFG = _REPO_ROOT / "finetune_ashare" / "configs" / "mainboard_daily_v1.yaml"
OUT = _REPO_ROOT / "finetune_ashare" / "outputs" / "mainboard_daily_v1"
# Match training backtest sample size for apples-to-apples; bump if you want tighter CIs
MAX_SAMPLES = 256
SEED = 42


def _eval(
    label: str,
    tokenizer_path: Path,
    model_path: Path,
    *,
    mem,
    refs,
    config: AshareFinetuneConfig,
    device: torch.device,
) -> dict:
    print(f"\n=== {label} ===")
    print(f"  tokenizer: {tokenizer_path}")
    print(f"  model:     {model_path}")
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device).eval()
    model = Kronos.from_pretrained(str(model_path)).to(device).eval()
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
        max_samples=MAX_SAMPLES,
        seed=SEED,
        pred_len=config.predict_window,
        T=config.day1_T,
        top_p=config.day1_top_p,
        sample_count=config.day1_sample_count,
        lookback=config.lookback_window,
    )
    row = {
        "label": label,
        "tokenizer": str(tokenizer_path),
        "model": str(model_path),
        "dir_acc": float(metrics["dir_acc"]),
        "n": int(metrics["n"]),
    }
    for h in range(1, int(config.predict_window) + 1):
        row[f"e{h}_mean"] = float(metrics[f"e{h}_mean"])
    print(
        f"  {format_e_means(row, int(config.predict_window))} "
        f"dir_acc={row['dir_acc']:.4f} n={row['n']}"
    )
    # free VRAM between runs
    del predictor, model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    config = AshareFinetuneConfig(str(CFG))
    device = torch.device(
        f"cuda:{config.device_id}"
        if config.use_cuda and torch.cuda.is_available()
        else "cpu"
    )
    print(f"device={device}")

    print(f"Loading market from {config.db_path} ...")
    mem = load_market_memory(
        config.db_path,
        lookback=config.lookback_window,
        predict=config.predict_window,
    )
    pre_calendar = [d for d in mem.calendar if d < config.backtest_start]
    blocks = build_val_symbol_sets(
        pre_calendar,
        mem.symbols,
        backtest_start=config.backtest_start,
        block_days=config.val_block_trading_days,
        val_ratio=config.val_symbol_ratio,
        seed=config.seed,
    )
    refs = build_sample_refs(
        mem,
        blocks,
        lookback=config.lookback_window,
        predict=config.predict_window,
        backtest_start=config.backtest_start,
    )["backtest"]
    print(f"backtest refs available={len(refs)}, eval n={MAX_SAMPLES}, seed={SEED}")

    pretrained_tok = Path(config.pretrained_tokenizer_path)
    pretrained_pred = Path(config.pretrained_predictor_path)
    ft_tok = OUT / "tokenizer" / "best_model"
    best_e1 = OUT / "basemodel" / "best_e1"  # last improved at epoch 6
    best_loss = OUT / "basemodel" / "best_loss"  # last saved at epoch 10

    results = [
        _eval(
            "pretrained (原始)",
            pretrained_tok,
            pretrained_pred,
            mem=mem,
            refs=refs,
            config=config,
            device=device,
        ),
        _eval(
            "best_e1 (~第6轮)",
            ft_tok,
            best_e1,
            mem=mem,
            refs=refs,
            config=config,
            device=device,
        ),
        _eval(
            "best_loss (~第10轮)",
            ft_tok,
            best_loss,
            mem=mem,
            refs=refs,
            config=config,
            device=device,
        ),
    ]

    out_path = OUT / "compare_2026.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n======== 汇总 (同一批 2026 样本) ========")
    hdr = f"{'model':<22} {'e1':>8} {'e2':>8} {'e3':>8} {'e4':>8} {'e5':>8} {'dir@1':>8} {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['label']:<22} "
            f"{r['e1_mean']:8.6f} {r['e2_mean']:8.6f} {r['e3_mean']:8.6f} "
            f"{r['e4_mean']:8.6f} {r['e5_mean']:8.6f} "
            f"{r['dir_acc']:8.4f} {r['n']:5d}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
