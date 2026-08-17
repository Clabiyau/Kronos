import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


class AshareFinetuneConfig:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        data = cfg.get("data", {})
        split = cfg.get("split", {})
        training = cfg.get("training", {})
        model_paths = cfg.get("model_paths", {})
        eval_cfg = cfg.get("eval", {})
        device = cfg.get("device", {})
        logging_cfg = cfg.get("logging", {})

        self.db_path = _resolve_path(data.get("db_path", ""))
        self.lookback_window = data.get("lookback_window", 200)
        self.predict_window = data.get("predict_window", 5)
        self.max_context = data.get("max_context", 512)
        self.clip = data.get("clip", 5.0)
        self.symbol_prefixes = data.get("symbol_prefixes", ["60", "00"])

        self.backtest_start = split.get("backtest_start", "2026-01-01")
        self.val_block_trading_days = split.get("val_block_trading_days", 100)
        self.val_symbol_ratio = split.get("val_symbol_ratio", 0.2)
        self.seed = split.get("seed", training.get("seed", 42))

        self.batch_size = training.get("batch_size", 32)
        self.log_interval = training.get("log_interval", 50)
        self.num_workers = training.get("num_workers", 0)
        self.tokenizer_epochs = training.get("tokenizer_epochs", 5)
        self.basemodel_epochs = training.get("basemodel_epochs", 10)
        self.tokenizer_learning_rate = training.get("tokenizer_learning_rate", 2e-4)
        self.predictor_learning_rate = training.get("predictor_learning_rate", 2e-5)
        self.adam_beta1 = training.get("adam_beta1", 0.9)
        self.adam_beta2 = training.get("adam_beta2", 0.95)
        self.adam_weight_decay = training.get("adam_weight_decay", 0.1)
        self.accumulation_steps = training.get("accumulation_steps", 1)
        self.n_train_steps_per_epoch = training.get("n_train_steps_per_epoch", 500)
        self.n_val_loss_batches = training.get("n_val_loss_batches", 50)
        self.n_val_day1_samples = training.get("n_val_day1_samples", 128)
        self.onecycle_pct_start = float(training.get("onecycle_pct_start", 0.03))
        self.onecycle_div_factor = float(training.get("onecycle_div_factor", 10))
        self.onecycle_final_div_factor = float(
            training.get("onecycle_final_div_factor", 1e4)
        )

        self.exp_name = model_paths.get("exp_name", "default")
        base_path = _resolve_path(model_paths.get("base_path", "./finetune_ashare/outputs"))
        self.base_save_path = os.path.join(base_path, self.exp_name)
        self.pretrained_tokenizer_path = _resolve_path(
            model_paths.get("pretrained_tokenizer", "")
        )
        self.pretrained_predictor_path = _resolve_path(
            model_paths.get("pretrained_predictor", "")
        )

        self.tokenizer_best_path = os.path.join(
            self.base_save_path, "tokenizer", "best_model"
        )
        self.basemodel_best_e1_path = os.path.join(
            self.base_save_path, "basemodel", "best_e1"
        )
        self.basemodel_best_loss_path = os.path.join(
            self.base_save_path, "basemodel", "best_loss"
        )
        self.basemodel_last_path = os.path.join(
            self.base_save_path, "basemodel", "last"
        )
        self.basemodel_last_train_path = os.path.join(
            self.base_save_path, "basemodel", "last_train.pt"
        )
        self.tb_log_dir = os.path.join(self.base_save_path, "tb")

        self.use_tensorboard = logging_cfg.get("use_tensorboard", True)
        self.run_backtest = eval_cfg.get("run_backtest", True)
        self.backtest_max_samples = eval_cfg.get("backtest_max_samples", 256)
        self.day1_T = eval_cfg.get("day1_T", 1.0)
        self.day1_top_p = eval_cfg.get("day1_top_p", 0.9)
        self.day1_sample_count = eval_cfg.get("day1_sample_count", 1)
        self.use_cuda = device.get("use_cuda", True)
        self.device_id = device.get("device_id", 0)
