"""
Stock Price Prediction App
A Streamlit application for forecasting stock prices using Hybrid DLinear-Transformer model.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-positive {
        color: #00c853;
    }
    .metric-negative {
        color: #ff1744;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class TrendBranch(nn.Module):
    def __init__(self, seq_len, pred_len, num_features):
        super(TrendBranch, self).__init__()
        self.time_linear = nn.Linear(seq_len, pred_len)
        self.feature_linear = nn.Linear(num_features, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.time_linear(x)
        x = x.permute(0, 2, 1)
        x = self.feature_linear(x)
        return self.act(x)


class SeasonalBranch(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, d_model=64):
        super(SeasonalBranch, self).__init__()
        self.pred_len = pred_len
        self.input_embedding = nn.Linear(num_features, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, seq_len + pred_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.query_token = nn.Parameter(torch.randn(pred_len, d_model))
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, C = x.shape
        enc_in = self.input_embedding(x) + self.position_encoding[:, :L, :]
        memory = self.encoder(enc_in)
        query = self.query_token.unsqueeze(0).repeat(B, 1, 1) + self.position_encoding[:, L:L + self.pred_len, :]
        out = self.decoder(query, memory)
        return self.projection(out)


class HybridDLinearTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, moving_avg=25):
        super(HybridDLinearTransformer, self).__init__()
        self.revin = RevIN(num_features)
        self.decomposition = MovingAvg(kernel_size=moving_avg, stride=1)
        self.trend_model = TrendBranch(seq_len, pred_len, num_features)
        self.seasonal_model = SeasonalBranch(seq_len, pred_len, num_features)

    def forward(self, x, target_idx):
        x = self.revin(x, 'norm')
        trend = self.decomposition(x)
        seasonal = x - trend

        trend_out = self.trend_model(trend)
        seasonal_out = self.seasonal_model(seasonal)
        prediction = trend_out + seasonal_out

        target_mean = self.revin.mean[:, :, target_idx:target_idx + 1]
        target_std = self.revin.stdev[:, :, target_idx:target_idx + 1]

        if self.revin.affine:
            prediction = prediction - self.revin.affine_bias[target_idx]
            prediction = prediction / (self.revin.affine_weight[target_idx] + 1e-5)

        prediction = prediction * target_std + target_mean
        return prediction


# =============================================================================
# DATA PROCESSING
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, feature_cols, target_col):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = df[feature_cols].values.astype(np.float32)
        target_data = df[target_col].values.astype(np.float32)
        if target_data.ndim == 1:
            target_data = target_data.reshape(-1, 1)
        self.target = target_data

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.target[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@st.cache_data
def load_and_process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Auto-detect time column
    time_cols = [col for col in df.columns if col.lower() in ['time', 'date', 'datetime', 'timestamp']]
    if time_cols:
        df['time'] = pd.to_datetime(df[time_cols[0]])
    else:
        df['time'] = pd.to_datetime(df.iloc[:, 0])

    df = df.sort_values('time').reset_index(drop=True)

    # Log transform price columns
    cols = ['open', 'high', 'low', 'close']
    for c in cols:
        if c in df.columns:
            df[f'{c}_log'] = np.log(df[c] + 1e-8)

    # Volume log transform
    if 'volume' in df.columns:
        df['volume_log'] = np.log1p(df['volume'])

    # Daily return
    if 'close' in df.columns:
        df['daily_return'] = df['close'].pct_change().fillna(0)

    # Time features
    df['dow_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)

    return df


# =============================================================================
# TRAINING & INFERENCE
# =============================================================================

def train_model(dataset, seq_len, feature_cols, target_col, config, progress_callback=None):
    device = config['device']
    target_idx = feature_cols.index(target_col)
    tscv = TimeSeriesSplit(n_splits=config['n_splits'])
    final_model = None

    indices = np.arange(len(dataset))
    total_steps = config['n_splits'] * config['epochs']
    current_step = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
        if len(train_idx) > config['train_window_size']:
            train_idx = train_idx[-config['train_window_size']:]

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config['batch_size'], shuffle=False)

        model = HybridDLinearTransformer(
            seq_len=seq_len,
            pred_len=config['pred_len'],
            num_features=len(feature_cols)
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.HuberLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        fold_best_loss = float('inf')
        for epoch in range(config['epochs']):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x, target_idx)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x, target_idx)
                    val_loss += criterion(output, y).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < fold_best_loss:
                fold_best_loss = val_loss
                if fold == config['n_splits'] - 1:
                    final_model = model

            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)

    return final_model


def monte_carlo_inference(model, input_data, steps, feature_cols, n_sims, device):
    model.eval()
    close_idx = feature_cols.index('close_log')
    hist_close = input_data[0, :, close_idx].cpu().numpy()
    volatility = np.std(np.diff(hist_close)) if len(hist_close) > 1 else 0.015

    simulations = np.zeros((n_sims, steps))

    for i in range(n_sims):
        curr_seq = input_data.clone()
        for step in range(steps):
            with torch.no_grad():
                pred = model(curr_seq, target_idx=close_idx)
                next_val = pred[0, 0, 0].item()

                noise = np.random.normal(0, volatility)
                next_val += noise
                simulations[i, step] = next_val

                new_point = curr_seq[:, -1:, :].clone()
                new_point[0, 0, close_idx] = next_val
                curr_seq = torch.cat([curr_seq[:, 1:, :], new_point], dim=1)

    mean_pred = np.mean(simulations, axis=0)
    lower_bound = np.percentile(simulations, 10, axis=0)
    upper_bound = np.percentile(simulations, 90, axis=0)

    return mean_pred, lower_bound, upper_bound, simulations


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_historical_data(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))

    fig.update_layout(
        title='Historical Price Data',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500
    )

    return fig


def plot_forecast(df, mean_pred, lower_bound, upper_bound, pred_len):
    last_date = df['time'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_len, freq='D')

    fig = go.Figure()

    # Historical close price (last 100 days)
    hist_df = df.tail(100)
    fig.add_trace(go.Scatter(
        x=hist_df['time'],
        y=hist_df['close'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))

    # Mean prediction
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=mean_pred,
        mode='lines',
        name='Forecast (Mean)',
        line=dict(color='#2ca02c', width=2)
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence'
    ))

    fig.update_layout(
        title='Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500,
        showlegend=True
    )

    return fig


def plot_simulations(simulations, mean_pred, pred_len):
    last_date = datetime.now()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_len, freq='D')

    fig = go.Figure()

    # Plot sample simulations
    for i in range(min(20, len(simulations))):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=np.exp(simulations[i]),
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
            showlegend=False
        ))

    # Mean prediction
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=np.exp(mean_pred),
        mode='lines',
        name='Mean Forecast',
        line=dict(color='#d62728', width=3)
    ))

    fig.update_layout(
        title='Monte Carlo Simulations',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<p class="main-header">Stock Price Forecaster</p>', unsafe_allow_html=True)
    st.markdown("Hybrid DLinear-Transformer with Monte Carlo Simulation")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        uploaded_file = st.file_uploader(
            "Upload Stock Data (CSV)",
            type=['csv'],
            help="CSV with columns: time/date, open, high, low, close, volume"
        )

        st.markdown("---")

        if uploaded_file:
            st.success("File uploaded successfully!")

            st.subheader("Model Parameters")

            seq_len = st.selectbox(
                "Sequence Length (days)",
                options=[60, 120, 240, 480],
                index=1,
                help="Number of historical days to use for prediction"
            )

            pred_len = st.slider(
                "Prediction Horizon (days)",
                min_value=10,
                max_value=200,
                value=100,
                step=10
            )

            st.markdown("---")
            st.subheader("Training Parameters")

            epochs = st.slider("Epochs", 10, 200, 50, step=10)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
            n_splits = st.slider("CV Folds", 3, 10, 5)

            st.markdown("---")
            st.subheader("Monte Carlo")

            n_simulations = st.slider("Number of Simulations", 10, 200, 50, step=10)

            st.markdown("---")

            train_button = st.button("Train & Forecast", type="primary")

    # Main content
    if uploaded_file is None:
        st.info("Please upload a CSV file with stock data to get started.")

        st.markdown("""
        ### Expected Data Format

        Your CSV should contain the following columns:
        - **time** or **date**: Date/datetime column
        - **open**: Opening price
        - **high**: Highest price
        - **low**: Lowest price
        - **close**: Closing price
        - **volume**: Trading volume

        ### Model Architecture

        This application uses a **Hybrid DLinear-Transformer** model that combines:
        - **RevIN**: Reversible Instance Normalization for distribution shift handling
        - **Trend-Seasonal Decomposition**: Separates trend and seasonal components
        - **DLinear Branch**: Captures linear trends efficiently
        - **Transformer Branch**: Models complex seasonal patterns
        - **Monte Carlo Simulation**: Provides uncertainty quantification

        ### Features
        - Time series cross-validation for robust training
        - Multiple sequence length options
        - Configurable prediction horizons
        - 90% confidence intervals via Monte Carlo simulation
        """)
        return

    # Load data
    df = load_and_process_data(uploaded_file)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Training & Forecast", "Results"])

    with tab1:
        st.header("Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{(df['time'].max() - df['time'].min()).days} days")
        with col3:
            current_price = df['close'].iloc[-1]
            st.metric("Current Price", f"{current_price:,.2f}")
        with col4:
            daily_change = df['daily_return'].iloc[-1] * 100
            st.metric("Last Change", f"{daily_change:+.2f}%")

        st.markdown("---")

        # Candlestick chart
        st.subheader("Historical Price Chart")
        fig = plot_historical_data(df)
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Volume Distribution")
            fig = px.histogram(df, x='volume', nbins=50, title='Trading Volume Distribution')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Daily Returns Distribution")
            fig = px.histogram(df, x='daily_return', nbins=50, title='Daily Returns Distribution')
            st.plotly_chart(fig, use_container_width=True)

        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(10), use_container_width=True)

    with tab2:
        st.header("Model Training")

        if train_button:
            seed_everything(42)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            st.info(f"Using device: {device}")

            config = {
                'device': device,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'n_splits': n_splits,
                'train_window_size': 365,
                'n_simulations': n_simulations
            }

            features = ['volume_log', 'close_log', 'daily_return', 'dow_sin', 'dow_cos']
            target = 'close_log'

            # Check if all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                return

            dataset = TimeSeriesDataset(df, seq_len, pred_len, features, target)

            if len(dataset) < 100:
                st.error("Not enough data for training. Please provide more historical data.")
                return

            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Training model...")

            def update_progress(progress):
                progress_bar.progress(progress)

            model = train_model(dataset, seq_len, features, target, config, update_progress)

            if model is None:
                st.error("Training failed. Please try different parameters.")
                return

            status_text.text("Running Monte Carlo simulations...")

            # Inference
            last_seq = torch.FloatTensor(dataset.features[-seq_len:]).unsqueeze(0).to(device)
            mean_pred, lower_bound, upper_bound, simulations = monte_carlo_inference(
                model, last_seq, pred_len, features, n_simulations, device
            )

            # Convert from log scale
            mean_price = np.exp(mean_pred)
            lower_price = np.exp(lower_bound)
            upper_price = np.exp(upper_bound)

            # Store results in session state
            st.session_state['forecast_results'] = {
                'mean': mean_price,
                'lower': lower_price,
                'upper': upper_price,
                'simulations': simulations,
                'df': df,
                'pred_len': pred_len
            }

            status_text.text("Training complete!")
            st.success("Model trained successfully! Check the Results tab.")

        else:
            st.info("Configure parameters in the sidebar and click 'Train & Forecast' to start.")

    with tab3:
        st.header("Forecast Results")

        if 'forecast_results' in st.session_state:
            results = st.session_state['forecast_results']

            # Summary metrics
            current_price = results['df']['close'].iloc[-1]
            final_price = results['mean'][-1]
            price_change = final_price - current_price
            pct_change = (price_change / current_price) * 100

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"{current_price:,.2f}")
            with col2:
                st.metric("Forecast (End)", f"{final_price:,.2f}")
            with col3:
                delta_color = "normal" if price_change >= 0 else "inverse"
                st.metric("Expected Change", f"{price_change:+,.2f}", delta=f"{pct_change:+.2f}%")
            with col4:
                uncertainty = results['upper'][-1] - results['lower'][-1]
                st.metric("Uncertainty Range", f"{uncertainty:,.2f}")

            st.markdown("---")

            # Forecast plot
            st.subheader("Price Forecast with Confidence Interval")
            fig = plot_forecast(
                results['df'],
                results['mean'],
                results['lower'],
                results['upper'],
                results['pred_len']
            )
            st.plotly_chart(fig, use_container_width=True)

            # Monte Carlo simulations
            st.subheader("Monte Carlo Simulation Paths")
            fig = plot_simulations(results['simulations'], 
                                   np.log(results['mean']),  # Convert back to log for consistency
                                   results['pred_len'])
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            st.markdown("---")
            st.subheader("Download Forecast")

            forecast_df = pd.DataFrame({
                'day': range(1, results['pred_len'] + 1),
                'predicted_price': results['mean'],
                'lower_bound_90': results['lower'],
                'upper_bound_90': results['upper']
            })

            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name="stock_forecast.csv",
                mime="text/csv"
            )

        else:
            st.info("Train a model first to see forecast results.")


if __name__ == "__main__":
    main()
