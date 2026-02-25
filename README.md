## Creator/Dev: Rajput Prajjval Singh
<img width="2958" height="1504" alt="image" src="https://github.com/user-attachments/assets/1a9b52be-4dd5-4e2a-b4c8-793c31229f18" />

# LSTM Volatility Forecasting Engine

This project is a fully functional Streamlit dashboard for forecasting financial market volatility using deep learning (LSTM neural networks). It allows users to upload their own price data or generate synthetic data, perform feature engineering, train a PyTorch LSTM model, and visualize forecast results interactively.

## Features
- CSV upload or synthetic GBM price generation
- Log returns and rolling realized volatility calculation
- Sliding window sequence creation for LSTM
- Configurable PyTorch LSTM (sequence length, hidden units, layers, epochs, learning rate)
- Train/validation split, MSE loss, Adam optimizer
- Interactive Streamlit dashboard with forecast plots, metrics, and error analysis

## How to Fork
1. Click the "Fork" button on GitHub to copy this repository to your account.
2. Clone your fork locally:
   ```
   git clone https://github.com/<your-username>/<your-forked-repo>.git
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run lstm_vol_forecasting_engine.py
   ```

## Related Topics to Explore
- [LSTM (Long Short-Term Memory) Neural Networks](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [Volatility Forecasting](https://en.wikipedia.org/wiki/Volatility_(finance))
- [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [PyTorch](https://en.wikipedia.org/wiki/PyTorch)
- [Streamlit](https://streamlit.io/)
- [Financial Time Series](https://en.wikipedia.org/wiki/Time_series)
- [Rolling Statistics](https://en.wikipedia.org/wiki/Moving_average)

---

For questions or improvements, feel free to fork, star, or open issues!

