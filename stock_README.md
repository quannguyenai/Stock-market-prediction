# Stock Price Forecasting: Hybrid DLinear-Transformer

A deep learning project for stock price prediction using a hybrid architecture that combines DLinear and Transformer models with Monte Carlo simulation for uncertainty quantification.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)

## Overview

This project implements a state-of-the-art time series forecasting model for stock price prediction. The model uses a hybrid approach combining:

- **DLinear**: A simple yet effective linear model for capturing trends
- **Transformer**: For modeling complex seasonal and cyclical patterns
- **Monte Carlo Simulation**: For uncertainty quantification and confidence intervals

The project is designed for the AIO 2025 Linear Forecasting Challenge and includes both a training script and an interactive Streamlit web application.

## Model Architecture

### Hybrid DLinear-Transformer

```
Input Sequence
      |
      v
  [RevIN] ──────────────────────────────────
      |                                     |
      v                                     |
[Moving Average Decomposition]              |
      |                                     |
      ├──────────────┬──────────────┐       |
      v              v              v       |
   [Trend]      [Seasonal]                  |
      |              |                      |
      v              v                      |
[DLinear Branch] [Transformer Branch]       |
      |              |                      |
      └──────┬───────┘                      |
             v                              |
         [Addition]                         |
             |                              |
             v                              |
    [Inverse RevIN] <───────────────────────┘
             |
             v
       Predictions
```

### Components

1. **RevIN (Reversible Instance Normalization)**
   - Handles distribution shift between training and inference
   - Learnable affine parameters for adaptive normalization

2. **Trend-Seasonal Decomposition**
   - Moving average kernel for trend extraction
   - Residual as seasonal component

3. **Trend Branch (DLinear)**
   - Time-axis linear projection
   - Feature-axis linear projection
   - GELU activation

4. **Seasonal Branch (Transformer)**
   - Input embedding layer
   - Positional encoding
   - Transformer encoder-decoder architecture
   - Query tokens for autoregressive prediction

## Features

- **Multiple Sequence Lengths**: Support for 60, 120, 240, 480-day lookback windows
- **Time Series Cross-Validation**: Proper temporal validation with TimeSeriesSplit
- **Monte Carlo Simulation**: Stochastic inference with configurable number of simulations
- **Confidence Intervals**: 90% prediction intervals based on simulation distribution
- **GPU Support**: Automatic CUDA detection and utilization
- **Interactive Web App**: Streamlit application for visualization and experimentation

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-forecasting.git
cd stock-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Script

#### Basic Usage

```python
# Update the data path in CONFIG
CONFIG = {
    'data_path': 'data/FPT_train.csv',
    # ... other settings
}

# Run the script
python project6_1.py
```

#### On Kaggle

```python
CONFIG = {
    'data_path': '/kaggle/input/aio-2025-linear-forecasting-challenge/FPT_train.csv',
    # ... other settings
}
```

#### Custom Configuration

```python
CONFIG = {
    'data_path': 'your_data.csv',
    'seed': 42,
    'seq_lengths': [120, 240],      # Sequence lengths to test
    'pred_len': 100,                 # Prediction horizon
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10,
    'n_splits': 5,                   # Cross-validation folds
    'train_window_size': 365,        # Rolling window size
    'n_simulations': 50,             # Monte Carlo simulations
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
```

### Running the Streamlit App

```bash
streamlit run stock_app.py
```

Then open your browser to `http://localhost:8501`

The app provides:
- Upload custom stock data (CSV)
- Configure model parameters interactively
- Visualize historical data with candlestick charts
- Train model with progress tracking
- View forecasts with confidence intervals
- Download prediction results

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| seed | 42 | Random seed for reproducibility |
| seq_lengths | [120, 240, 480] | List of sequence lengths to train |
| pred_len | 100 | Number of days to forecast |
| batch_size | 32 | Training batch size |
| learning_rate | 0.001 | Adam optimizer learning rate |
| epochs | 100 | Maximum training epochs |
| patience | 10 | Early stopping patience |
| n_splits | 5 | Time series CV folds |
| train_window_size | 365 | Rolling training window |
| n_simulations | 50 | Monte Carlo simulation count |

## Data Format

The input CSV should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| time | datetime | Date/timestamp |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | int/float | Trading volume |

Example:
```csv
time,open,high,low,close,volume
2020-01-02,50000,51000,49500,50500,1000000
2020-01-03,50500,52000,50000,51500,1200000
...
```

## Project Structure

```
stock-forecasting/
├── project6_1.py          # Main training script
├── stock_app.py           # Streamlit web application
├── README.md              # Documentation
├── requirements.txt       # Python dependencies
└── data/
    └── FPT_train.csv      # Training data
```

## Methodology

### Data Preprocessing

1. **Log Transformation**: Apply log transform to price columns for stable gradients
2. **Volume Normalization**: Log1p transform for volume
3. **Return Calculation**: Daily percentage returns
4. **Temporal Features**: Cyclical encoding of day-of-week using sin/cos

### Feature Engineering

Input features:
- `volume_log`: Log-transformed trading volume
- `close_log`: Log-transformed closing price (target)
- `daily_return`: Daily percentage change
- `dow_sin`: Sine encoding of day-of-week
- `dow_cos`: Cosine encoding of day-of-week

### Training Strategy

1. **Time Series Cross-Validation**: 5-fold temporal split
2. **Rolling Window**: Last 365 days for training to focus on recent patterns
3. **Loss Function**: Huber Loss for robustness to outliers
4. **Optimizer**: Adam with ReduceLROnPlateau scheduler
5. **Model Selection**: Best model from last fold

### Inference

1. **Autoregressive Prediction**: Step-by-step forecasting
2. **Monte Carlo Simulation**: Add stochastic noise based on historical volatility
3. **Confidence Intervals**: 10th and 90th percentiles from simulations

## Results

Output files generated:
- `submission_120d_hybrid.csv`: Predictions using 120-day sequence
- `submission_240d_hybrid.csv`: Predictions using 240-day sequence
- `submission_480d_hybrid.csv`: Predictions using 480-day sequence

Each file contains:
```csv
id,close
1,predicted_price_day_1
2,predicted_price_day_2
...
100,predicted_price_day_100
```

## Dependencies

```
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
streamlit>=1.28.0
plotly>=5.15.0
```

## References

- [DLinear: Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
- [RevIN: Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

MIT License
