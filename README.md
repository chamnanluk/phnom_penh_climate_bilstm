# Phnom Penh Daily Temperature and Rainfall Forecasting — BiLSTM Experiments

This project forecasts daily average temperature and rainfall in Phnom Penh using NASA meteorological data.

It contains two comparable model paths:

1. **Baseline Multi-task BiLSTM (TensorFlow)**
   - Temperature regression
   - Rain / no-rain classification
   - Rainfall amount regression
2. **Hybrid SARIMA + BiLSTM (PyTorch experiment)**
   - SARIMA for linear/seasonal temperature signal
   - BiLSTM for nonlinear residual correction

## Project structure

```text
phnom_penh_climate_bilstm/
├── data/
│   └── Dataset_with_RainStatus_V5.csv
├── experiments/
│   ├── train_baseline.py
│   └── train_hybrid_sarima_bilstm.py
├── models/
├── outputs/
├── src/
│   ├── data.py
│   ├── inference/
│   ├── models/
│   └── training/
├── streamlit_app.py
├── config.py
├── requirements.txt
└── requirements-streamlit.txt
```

## Training (research workflow)

Install training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the baseline model:

```bash
python run_train.py
```

Train the hybrid experiment:

```bash
python experiments/train_hybrid_sarima_bilstm.py
```

---

## Deploy the prediction app

This app is inference-only and **does not retrain** in Streamlit.

### 1) Local setup for prediction app

Use a clean environment with deployment-focused dependencies:

```bash
python -m venv .venv-app
source .venv-app/bin/activate  # Windows: .venv-app\Scripts\activate
pip install -r requirements-streamlit.txt
```

### 2) Train/export baseline artifacts once (local or Colab)

Run baseline training once:

```bash
python run_train.py
```

This produces baseline artifacts for inference:

- `models/best_baseline.keras`
- `models/baseline_scalers.pkl`
- `models/baseline_metadata.json`

> If these files are missing, the app will show a model artifact error with guidance.

### 3) Run the Streamlit app locally

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

### 4) Input format expected by the app

Provide CSV data with at least these columns:

- `Date`
- `PRECTOTCORR`
- `WS2M`
- `T2M_RANGE`
- `T2M_MAX`
- `T2M_MIN`
- `PS`
- `ALLSKY_SFC_SW_DWN`
- `RH2M`
- `T2MDEW`

The app validates:

- required columns
- valid dates
- numeric values
- missing values
- minimum sequence length

### 5) Streamlit Community Cloud configuration

In Streamlit Community Cloud, configure:

- **Repository:** `chamnanluk/phnom_penh_climate_bilstm`
- **Branch:** `main` (or your deployment branch)
- **Main file path:** `streamlit_app.py`
- **Dependencies file:** `requirements-streamlit.txt`

### 6) Common errors and fixes

- **`Model artifact error: Metadata file not found...`**
  - Run `python run_train.py` and ensure artifact files exist under `models/`.
- **`Input validation error: Missing required columns...`**
  - Ensure CSV headers exactly match required names.
- **`Input validation error: Not enough usable rows...`**
  - Provide a longer recent history (continuous daily records).
- **TensorFlow not installed**
  - Install `requirements-streamlit.txt` before launching the app.

## Training vs prediction (important)

- **Training scripts** (`run_train.py`, `experiments/*`) are for research and model development.
- **Prediction app** (`streamlit_app.py`) is for end-user inference with already-saved artifacts.
- The app intentionally does not start model training.
