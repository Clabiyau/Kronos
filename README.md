# Kronos

面向金融 K 线（OHLCV）的预训练基础模型，基于自回归 Transformer，支持零样本预测。

- 论文：[arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- 原始模型：[NeoQuasar @ Hugging Face](https://huggingface.co/NeoQuasar)

## 模型说明

| 模型 | Tokenizer | 上下文长度 | 参数量 |
|------|-----------|------------|--------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M |

Tokenizer 与模型必须配对使用，不可混用。Tokenizer-base 对应 small / base；Tokenizer-2k 仅对应 mini。

本项目默认使用本地预训练权重：

```
pretrained/
├── Kronos-Tokenizer-base/
└── Kronos-base/
```

## 环境安装

```shell
pip install -r requirements.txt
```

建议使用 Python 3.10+，GPU 推理需安装对应版本的 PyTorch。

### 下载模型到本地

若 `pretrained/` 目录为空，可从 Hugging Face 下载（国内建议设置镜像）：

```shell
set HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download NeoQuasar/Kronos-Tokenizer-base --local-dir pretrained/Kronos-Tokenizer-base
huggingface-cli download NeoQuasar/Kronos-base --local-dir pretrained/Kronos-base
```

## 日线预测接口（推荐）

项目封装了 `DailyKronosPredictor`（`inference/daily_predictor.py`），传入 DataFrame 和预测天数即可，无需手动构造时间戳。

### 单只股票

```python
import pandas as pd
from inference import DailyKronosPredictor

df = pd.read_csv("your_daily.csv", parse_dates=["date"])

predictor = DailyKronosPredictor(device="cuda:0", lookback=400)
result = predictor.predict(df, pred_days=30)

print(result)
# 列: date, open, high, low, close, volume, amount
```

### 多只股票并行

```python
results = predictor.predict_batch(
    df_list=[df_a, df_b, df_c],
    pred_days=30,
    batch_size=8,      # GPU 批量并行数，显存不足时调小
    sample_count=1,    # 采样路径数，2-3 可略提升稳定性
)
```

### 输入数据要求

| 列 | 必须 | 说明 |
|----|------|------|
| `open`, `high`, `low`, `close` | 是 | 价格列，不能有 NaN |
| `date` / `timestamps` / `timestamp` / `datetime` | 是 | 时间列，四选一 |
| `volume`, `amount` | 否 | 缺失时自动补零 |

其他要求：

- 每行一根 K 线，按时间升序排列
- 至少 `lookback` 行历史数据（默认 400）
- `pred_days` 表示预测的未来**交易日**数量
- `lookback` 建议 ≤ 512（模型上下文上限）

### 并行参数建议

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `batch_size` | 多标的 GPU 批量并行 | 4–8（8GB 显存） |
| `sample_count` | 单标的多次采样取平均 | 1（快）或 2–3（稳） |

### 快速验证

```shell
python tests/run_daily_predictor_test.py
```

## 本地推理服务（stock_qmt 对接）

Kronos 作为独立 conda 环境运行 HTTP 服务，直读 ``stock_qmt/storage/kline.db``，stock_qmt 侧只传 symbol 等参数。

### 启动

```shell
conda activate kronos
pip install -r requirements.txt

# 可选环境变量
# set KRONOS_KLINE_DB_PATH=d:\Project\stock_qmt\storage\kline.db
# set KRONOS_HOST=127.0.0.1
# set KRONOS_PORT=8765

python -m inference.server
```

### API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | ``/health`` | 服务与模型状态 |
| POST | ``/predict`` | 单股预测 |
| POST | ``/predict_batch`` | 多股批量预测 |

``POST /predict`` 请求示例：

```json
{
  "symbol": "600519",
  "pred_days": 5,
  "lookback": 400,
  "end_date": "2026-03-07"
}
```

## 底层 API

如需自定义时间窗口或非日线数据，可直接使用 `KronosPredictor`：

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("./pretrained/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("./pretrained/Kronos-base")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
)
```

更多示例见 `examples/` 目录；WebUI 见 `webui/README.md`。

## 微调

如需在自己的数据上微调，参考：

- Qlib 流程：`finetune/`（配置见 `finetune/config.py`）
- CSV 流程：`finetune_csv/README_CN.md`
- A 股主板日线：`finetune_ashare/README_CN.md`

## 引用

```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
}
```

## 许可证

[MIT License](./LICENSE)
