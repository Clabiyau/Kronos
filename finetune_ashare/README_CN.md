# A 股主板日线微调（`finetune_ashare`）

在沪深主板日线（`kline.db`）上顺序微调 Kronos Tokenizer → Predictor，带 TensorBoard、双 checkpoint（`best_e1` / `best_loss`），以及可选的 2026 回测（报告第 1–5 天相对误差 \(e_h\)，方向准确率仍为第 1 天）。

## 数据假设

- SQLite：`kline_daily`，路径在配置 `data.db_path`（默认相对仓库根解析）
- 标的：`60*.SH` + `00*.SZ`；`suspendFlag==1` 的 bar 不进入窗口
- 默认窗口：lookback **200**、predict **5**、batch **32**
- `n_train_steps_per_epoch` 表示每个 epoch 的**优化步数**（不是样本数）；例如 500×batch32 ≈ 每 epoch 16000 条样本
- 划分：asof `< 2026-01-01` 为 train/val；`>= 2026-01-01` 为 backtest holdout
- Val：每 **100** 个交易日轮换 **20%** 股票（固定 seed）

默认配置：`finetune_ashare/configs/mainboard_daily_v1.yaml`（请按本机改 `db_path` 与 `pretrained_*`）。
加长 lookback / 步数的实验配置：
- `finetune_ashare/configs/mainboard_daily_lb300_s2000.yaml`（lookback=300，2000×32）
- `finetune_ashare/configs/mainboard_daily_lb300_s10k_b64.yaml`（lookback=300，**10000×64**，约合 train 池 23%/epoch）

每个 epoch **不会扫完全部样本**：从全量 train 池（约数百万条）里随机抽 `n_train_steps_per_epoch × batch_size` 条。标的与日期范围仍是库内主板全量（2020–2025 训练 / 2026 回测）。

## 依赖

仓库 `requirements.txt` 已包含：

- `PyYAML>=6.0`
- `tensorboard>=2.14`

以及现有 PyTorch / pandas / numpy 等。安装前请按仓库规则确认后再执行 `pip install -r requirements.txt`。

## 运行

在仓库根目录：

```bash
# 完整流程：tokenizer → predictor → 2026 回测（若 eval.run_backtest=true）
python -m finetune_ashare --config finetune_ashare/configs/mainboard_daily_v1.yaml

# 跳过 tokenizer / predictor
python -m finetune_ashare --config finetune_ashare/configs/mainboard_daily_v1.yaml --skip-tokenizer
python -m finetune_ashare --config finetune_ashare/configs/mainboard_daily_v1.yaml --skip-basemodel

# 已有产物则跳过：tokenizer 有 best_model；basemodel 需 best_e1 与 best_loss 都存在
python -m finetune_ashare --config finetune_ashare/configs/mainboard_daily_v1.yaml --skip-existing

# Tokenizer 已完成、只训 / 续训 Predictor（每 epoch 结束写入 basemodel/last）
python -m finetune_ashare --config finetune_ashare/configs/mainboard_daily_lb300_s10k_b64.yaml --skip-tokenizer
python -m finetune_ashare --config finetune_ashare/configs/mainboard_daily_lb300_s10k_b64.yaml --resume-predictor
```

`--resume-predictor` 会自动跳过 Tokenizer，从 `basemodel/last` + `last_train.pt` 接着下一个 epoch。注意：续训粒度是 **epoch 级**（Ctrl+C 在 epoch 中途仍会丢掉当前 epoch 进度）。

8GB 显存建议：`batch_size=16` + `accumulation_steps=4`（等效 batch 64），见 `mainboard_daily_lb300_s10k_b64.yaml`。

产出目录（以 `exp_name=mainboard_daily_v1` 为例）：

```
finetune_ashare/outputs/mainboard_daily_v1/
  tokenizer/best_model/
  basemodel/best_e1/
  basemodel/best_loss/
  tb/
  metrics.json
```

## TensorBoard

```bash
tensorboard --logdir finetune_ashare/outputs/mainboard_daily_v1/tb
```

常见标量：`tokenizer/train_loss`、`tokenizer/val_loss`、`predictor/train_loss`、`predictor/val_loss`、`predictor/val_e1`…`predictor/val_e5`（第 1–5 天相对收盘误差）、`predictor/val_dir_acc`（仅第 1 天涨跌方向）。选模仍以 `val_e1` 为准。

## 推理衔接（环境变量）

不改默认 `pretrained/`。把路径指到微调产物即可。

推理服务（`inference.server`）读取：

```bash
set KRONOS_TOKENIZER_PATH=finetune_ashare/outputs/mainboard_daily_v1/tokenizer/best_model
set KRONOS_MODEL_PATH=finetune_ashare/outputs/mainboard_daily_v1/basemodel/best_e1
# 或 best_loss：
# set KRONOS_MODEL_PATH=finetune_ashare/outputs/mainboard_daily_v1/basemodel/best_loss
python -m inference.server
```

代码侧也可直接传路径：

```python
from inference import DailyKronosPredictor

p = DailyKronosPredictor(
    model_path="finetune_ashare/outputs/mainboard_daily_v1/basemodel/best_e1",
    tokenizer_path="finetune_ashare/outputs/mainboard_daily_v1/tokenizer/best_model",
)
```
