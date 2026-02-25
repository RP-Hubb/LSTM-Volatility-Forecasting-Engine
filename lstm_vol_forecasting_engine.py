import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO

# ==============================
# Utility Functions
# ==============================
def generate_gbm(n_days=1000, mu=0.05, sigma=0.2, s0=100, seed=42):
    np.random.seed(seed)
    dt = 1/252
    returns = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), n_days)
    price = s0 * np.exp(np.cumsum(returns))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame({'date': dates, 'close': price})

def realized_volatility(returns, window):
    return returns.rolling(window).std() * np.sqrt(252)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

# ==============================
# PyTorch LSTM Model
# ==============================
class LSTMVolModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="LSTM Volatility Forecasting Engine", layout="wide")
st.title("LSTM Volatility Forecasting Engine")

with st.sidebar:
    st.header("Configuration")
    seq_length = st.slider("Sequence Length", 5, 60, 20)
    hidden_size = st.slider("Hidden Units", 8, 128, 32)
    num_layers = st.slider("LSTM Layers", 1, 3, 1)
    epochs = st.slider("Epochs", 10, 200, 50)
    lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%f")
    st.markdown("---")
    data_source = st.radio("Data Source", ["Upload CSV", "Synthetic GBM"], horizontal=True)

# ==============================
# Data Loading
# ==============================
if data_source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (date, close)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if not {'date', 'close'}.issubset(df.columns):
            st.error("CSV must have 'date' and 'close' columns.")
            st.stop()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        st.info("Please upload a CSV file.")
        st.stop()
else:
    n_days = st.slider("Synthetic GBM Days", 500, 3000, 1000)
    df = generate_gbm(n_days=n_days)

# ==============================
# Feature Engineering
# ==============================
df['log_return'] = np.log(df['close']).diff()
df['realized_vol'] = realized_volatility(df['log_return'], window=seq_length)
df = df.dropna().reset_index(drop=True)

# Prepare sequences
data_vol = df['realized_vol'].values.astype(np.float32)
X, y = create_sequences(data_vol, seq_length)
X = X[..., np.newaxis]  # shape: (samples, seq_length, 1)
y = y[..., np.newaxis]  # shape: (samples, 1)

# Train/val split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_val_t = torch.tensor(X_val).to(device)
y_val_t = torch.tensor(y_val).to(device)

# ==============================
# Model Training
# ==============================
model = LSTMVolModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)
        val_losses.append(val_loss.item())

# ==============================
# Evaluation
# ==============================
model.eval()
with torch.no_grad():
    y_pred = model(X_val_t).cpu().numpy().flatten()
    y_true = y_val.flatten()
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
errors = y_pred - y_true

# ==============================
# Plots
# ==============================
def plot_forecast(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_true, label='Actual', color='black')
    ax.plot(y_pred, label='Predicted', color='royalblue', alpha=0.7)
    ax.set_title('Predicted vs Actual Volatility')
    ax.set_xlabel('Time')
    ax.set_ylabel('Volatility')
    ax.legend()
    st.pyplot(fig)

def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_title('Training Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    st.pyplot(fig)

def plot_error_dist(errors):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(errors, bins=30, color='tomato', alpha=0.7)
    ax.set_title('Forecast Error Distribution')
    ax.set_xlabel('Error')
    st.pyplot(fig)

# ==============================
# Streamlit Layout
# ==============================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Forecast Plot")
    plot_forecast(y_true, y_pred)
    st.subheader("Training Loss Curve")
    plot_loss(train_losses, val_losses)
with col2:
    st.subheader("Metrics")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R-squared", f"{r2:.4f}")
    st.subheader("Error Distribution")
    plot_error_dist(errors)

st.caption("© 2026 LSTM Volatility Forecasting Engine. Powered by PyTorch & Streamlit.")
