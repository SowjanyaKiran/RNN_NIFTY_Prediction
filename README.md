# RNN-based NIFTY Prediction

This repository contains a single Jupyter notebook that trains simple recurrent neural networks (SimpleRNN, LSTM, GRU) to predict the **NIFTY 50** index using historical data from Yahoo Finance.

> **Notebook**: `RNN_NIFTY_Prediction.ipynb`  
> **Ticker**: `^NSEI` (NIFTY 50, Yahoo Finance)  
> **Date Range**: From `2010-01-01` to latest available (pulled live via `yfinance`)  
> **Primary Target/Features**: By default `Close` only (univariate).  
> **Sequence Length**: 60 timesteps.  
> **Train/Test Split**: 80% / 20%  
> **Scaling**: `MinMaxScaler` on selected features.  
> **Loss/Metric**: MSE loss; MAE reported.  
> **Model Selection**: Best model chosen by lowest test RMSE on `Close`.

---

## 1) What the notebook does

1. **Downloads data** for `^NSEI` via `yfinance` (`START_DATE="2010-01-01"`, `END_DATE=None` for “up to latest”).  
2. **(Optional) Decomposes** the `Close` series (multiplicative) for a quick seasonal/trend look (uses `statsmodels`).  
3. **Preprocesses** the data:
   - Selects features from `USE_FEATURES` (default: `['Close']`).
   - Scales with `MinMaxScaler`.
   - Builds supervised sequences with a sliding window of `TIME_STEPS=60`.
4. **Splits** into train/test with `TEST_SPLIT=0.2`.
5. **Builds three model families** with Keras:
   - **SimpleRNN**
   - **LSTM**
   - **GRU**
   Each family is tried with multiple depth/width configs (e.g., `[32]`, `[64, 32]`, `[128, 64, 32]`) and a `Dropout` after each recurrent layer.
6. **Trains** all variants with:
   - `EPOCHS=20`, `BATCH_SIZE=32`, `validation_split=0.15`
   - `EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)`
7. **Evaluates** on the test set:
   - Reports **RMSE** and **MAE** (on the rescaled/original `Close`).
   - Tracks runtime for each model.
   - Picks the **best model by RMSE**.
8. **Visualizes** the last 200 test points (Actual vs. Predicted) for the best model.

---

## 2) Environment & Requirements

Install Python 3.9+ with the packages below. A quick `pip` setup:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install   yfinance   pandas   numpy   scikit-learn   matplotlib   seaborn   statsmodels   tensorflow   keras
```

> **Note**: If you have a GPU, install the appropriate TensorFlow build and drivers for faster training.

---

## 3) How to run

1. Open the notebook:
   - VS Code: open folder and the notebook; or
   - Jupyter Lab/Notebook: `jupyter lab` or `jupyter notebook`.
2. Run cells from top to bottom. The notebook will:
   - Pull the data,
   - Prepare sequences,
   - Train RNN/LSTM/GRU variants,
   - Print **RMSE/MAE** for each,
   - Plot a zoomed comparison (last 200 test samples) for the **best** model.

If you hit rate limits or network issues, re-run the data download cell after a short delay.

---

## 4) Key knobs you can tweak

- **Ticker / Date Range**
  - `TICKER = "^NSEI"`
  - `START_DATE = "2010-01-01"`
  - `END_DATE = None` (use a concrete date like `"2025-11-01"` for reproducibility)

- **Features**
  - `USE_FEATURES = ['Close']` by default.
  - You can include others (if available): `['Open','High','Low','Close','Adj Close','Volume']`.
  - When you add features, the network predicts all selected features; evaluation focuses on `Close` by index.

- **Sequence Length**
  - `TIME_STEPS = 60` (try 30/90/120 etc. to see trade-offs).

- **Train/Test Split**
  - `TEST_SPLIT = 0.2`

- **Training**
  - `EPOCHS = 20`, `BATCH_SIZE = 32`
  - Early stopping patience, architectures:
    - RNN: `[32]`, `[64, 32]`, `[128, 64, 32]`
    - Same for LSTM/GRU (tweak units and dropout).

- **Model Families**
  - Functions: `build_rnn`, `build_lstm`, `build_gru`
  - Each adds `Dropout` after each recurrent layer and a final `Dense(n_features)`.

---

## 5) Outputs & interpretation

- **Console** prints per-model **RMSE, MAE, and runtime**.  
- A **line plot** compares **Actual vs. Predicted** on the **last 200** test points for the selected best model.
- Lower **RMSE/MAE** on `Close` is better. Compare families (RNN vs LSTM vs GRU) and depths to see which generalizes best.

> This is a **next-step regression** on scaled sequences; it does **not** forecast multiple days ahead by default. For multi-step forecasting, extend the prediction loop to roll inputs forward or train seq2seq models.

---

## 6) Reproducibility tips

- Set `END_DATE` to a fixed date.
- Set NumPy/TensorFlow seeds (add `np.random.seed(42)` and the TF seed setup). Some nondeterminism can remain depending on backend/cuDNN.
- Save models/metrics to disk if you want to compare runs over time.

---

## 7) Troubleshooting

- **yfinance download issues**: Try again, or reduce requests. Check network connectivity.
- **TensorFlow errors on CPU/GPU**: Ensure you have a compatible Python/TensorFlow and (for GPU) matching CUDA/cuDNN.
- **Notebook RAM usage**: Lower `TIME_STEPS`, reduce model size, or trim the date range.

---

## 8) Roadmap / ideas to extend

- Add **walk-forward validation** and **time-series split**.
- Try **learned optimizers** or **different losses** (e.g., Huber).
- Add **feature engineering** (returns, technical indicators).
- Implement **multi-step forecasting** (direct or recursive).
- Log experiments with **TensorBoard** or **Weights & Biases**.
- Deploy the best model as an API or Streamlit app.

---

## 9) Disclaimer

This project is for **educational/research** purposes. It is **not** financial advice. Past performance does not guarantee future results. Use responsibly.

---

## 10) License

Choose a license appropriate for your sharing needs (e.g., MIT, Apache-2.0). Update this section accordingly.
