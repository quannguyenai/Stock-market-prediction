import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIGURATION)
# =============================================================================
CONFIG = {
    'data_path': '/kaggle/input/aio-2025-linear-forecasting-challenge/FPT_train.csv',   
    'seed': 42,
    'seq_lengths': [120,240,480],    
    'pred_len': 100,                
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10,
    'n_splits': 5,
    'train_window_size': 365,
    'n_simulations': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG['seed'])
print(f"--> Đang sử dụng thiết bị: {CONFIG['device']}")

# =============================================================================
# 2. XỬ LÝ DỮ LIỆU (DATA ENGINEERING)
# =============================================================================
def load_and_process_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    cols = ['open', 'high', 'low', 'close']
    for c in cols:
        df[f'{c}_log'] = np.log(df[c] + 1e-8)
    df['volume_log'] = np.log1p(df['volume'])
    
    df['daily_return'] = df['close'].pct_change().fillna(0)
    
    dates = pd.to_datetime(df['time'])
    df['dow_sin'] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, pred_len, feature_cols, target_col):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.features = df[feature_cols].values.astype(np.float32)
        
        # --- FIX LỖI TẠI ĐÂY: Reshape target để có thêm chiều channel (N, 1) ---
        target_data = df[target_col].values.astype(np.float32)
        if target_data.ndim == 1:
            target_data = target_data.reshape(-1, 1)
        self.target = target_data
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.target[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# =============================================================================
# 3. KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)
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

    def forward(self, x, mode:str):
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
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=2, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        self.query_token = nn.Parameter(torch.randn(pred_len, d_model))
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, C = x.shape
        enc_in = self.input_embedding(x) + self.position_encoding[:, :L, :]
        memory = self.encoder(enc_in)
        query = self.query_token.unsqueeze(0).repeat(B, 1, 1) + self.position_encoding[:, L:L+self.pred_len, :]
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
        
        target_mean = self.revin.mean[:, :, target_idx:target_idx+1]
        target_std = self.revin.stdev[:, :, target_idx:target_idx+1]
        
        if self.revin.affine:
            prediction = prediction - self.revin.affine_bias[target_idx]
            prediction = prediction / (self.revin.affine_weight[target_idx] + 1e-5)
            
        prediction = prediction * target_std + target_mean
        return prediction

# =============================================================================
# 4. HUẤN LUYỆN & SUY DIỄN (TRAINING & INFERENCE)
# =============================================================================

def train_model(dataset, seq_len, feature_cols, target_col):
    target_idx = feature_cols.index(target_col)
    tscv = TimeSeriesSplit(n_splits=CONFIG['n_splits'])
    best_loss = float('inf')
    final_model = None
    
    print(f"\n>> Bắt đầu huấn luyện cho Sequence Length: {seq_len}")
    indices = np.arange(len(dataset))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
        if len(train_idx) > CONFIG['train_window_size']:
            train_idx = train_idx[-CONFIG['train_window_size']:]
            
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=CONFIG['batch_size'], shuffle=False)
        
        model = HybridDLinearTransformer(
            seq_len=seq_len, 
            pred_len=CONFIG['pred_len'], 
            num_features=len(feature_cols)
        ).to(CONFIG['device'])
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.HuberLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        fold_best_loss = float('inf')
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                optimizer.zero_grad()
                output = model(x, target_idx)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                    output = model(x, target_idx)
                    val_loss += criterion(output, y).item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < fold_best_loss:
                fold_best_loss = val_loss
                if fold == CONFIG['n_splits'] - 1:
                    final_model = model
            
            if (epoch+1) % 20 == 0:
                print(f"   Fold {fold+1} | Epoch {epoch+1} | Val Loss: {val_loss:.5f}")
    
    return final_model

def monte_carlo_inference(model, input_data, steps, feature_cols, n_sims=50):
    model.eval()
    close_idx = feature_cols.index('close_log')
    hist_close = input_data[0, :, close_idx].cpu().numpy()
    volatility = np.std(np.diff(hist_close)) if len(hist_close) > 1 else 0.015
    
    simulations = np.zeros((n_sims, steps))
    print(f"   >> Chạy {n_sims} kịch bản giả lập Monte Carlo...")
    
    for i in range(n_sims):
        curr_seq = input_data.clone()
        for step in range(steps):
            with torch.no_grad():
                pred = model(curr_seq, target_idx=close_idx)
                next_val = pred[0, 0, 0].item()
                
                noise = np.random.normal(0, volatility)
                next_val += noise
                simulations[i, step] = next_val
                
                # Cập nhật input: Tạo điểm mới và nối vào đuôi
                new_point = curr_seq[:, -1:, :].clone()
                new_point[0, 0, close_idx] = next_val 
                curr_seq = torch.cat([curr_seq[:, 1:, :], new_point], dim=1)
                
    mean_pred = np.mean(simulations, axis=0)
    lower_bound = np.percentile(simulations, 10, axis=0)
    upper_bound = np.percentile(simulations, 90, axis=0)
    
    return mean_pred, lower_bound, upper_bound

# =============================================================================
# 5. CHƯƠNG TRÌNH CHÍNH
# =============================================================================
if __name__ == "__main__":
    try:
        print("1. Đang tải và xử lý dữ liệu...")
        df = load_and_process_data(CONFIG['data_path'])
        
        features = ['volume_log', 'close_log', 'daily_return', 'dow_sin', 'dow_cos']
        target = 'close_log'
        results = {}
        
        for seq_len in CONFIG['seq_lengths']:
            dataset = TimeSeriesDataset(df, seq_len, CONFIG['pred_len'], features, target)
            model = train_model(dataset, seq_len, features, target)
            
            if model:
                last_seq = torch.FloatTensor(dataset.features[-seq_len:]).unsqueeze(0).to(CONFIG['device'])
                mean, lower, upper = monte_carlo_inference(
                    model, last_seq, steps=100, feature_cols=features, n_sims=CONFIG['n_simulations']
                )
                
                results[seq_len] = {'mean': np.exp(mean), 'lower': np.exp(lower), 'upper': np.exp(upper)}
                
                # Lưu file kết quả
                output_file = f'submission_{seq_len}d_hybrid.csv'
                sub = pd.DataFrame({'id': range(1, 101), 'close': np.exp(mean)})
                sub.to_csv(output_file, index=False)
                print(f"   [OK] Đã lưu: {output_file}")

        print("\n3. Đang vẽ biểu đồ kết quả...")
        plt.figure(figsize=(15, 7))
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        for i, (seq, res) in enumerate(results.items()):
            plt.plot(res['mean'], label=f'Model {seq}d Mean', color=colors[i], linewidth=2)
            plt.fill_between(range(100), res['lower'], res['upper'], color=colors[i], alpha=0.1, label=f'{seq}d Conf(90%)')
        
        plt.title("Dự báo giá FPT 100 ngày - Hybrid DLinear-Transformer", fontsize=14)
        plt.xlabel("Ngày tương lai")
        plt.ylabel("Giá (VNĐ)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        print("\n--> Hoàn tất!")
        
    except Exception as e:
        print(f"\n[LỖI] Đã xảy ra sự cố: {e}")
