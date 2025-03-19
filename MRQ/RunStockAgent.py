import yfinance as yf
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MRQStockAgent import MRQStockAgent

def get_stock_data(symbol, start="2010-01-01", end="2024-01-01"):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(symbol, start=start, end=end)
    file_name = f"{symbol}_{start}_{end}.csv"
    data.to_csv(file_name)
    headers = ['Date','Close','High','Low','Open','Volume']

    # Load CSV while skipping the first 3 lines
    stock = pd.read_csv(file_name, skiprows=3, names=headers)
    stock["SMA_5"] = stock["Close"].rolling(window=5).mean()
    stock["SMA_20"] = stock["Close"].rolling(window=20).mean()
    stock["Return"] = stock["Close"] - stock["Open"]
    stock["Volatility"] = stock["Return"].rolling(window=5).std()
    delta = stock["Close"].diff(1)
    delta.fillna(0, inplace=True)  # Fix: Replace NaN with 0

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Ensure no NaN issues in rolling calculations
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
    stock["RSI"] = 100 - (100 / (1 + rs))
    stock = stock.iloc[20:] # calc of 20 day moving average will lead ot NaN values
    return stock

data = get_stock_data("AAPL")
FEATURES = ["Close", "SMA_5", "SMA_20", "Return", "Volatility", "RSI"]
data[FEATURES].head()
data = data[FEATURES]
data = data.astype(float)
dates = data.index
print(data.head())



# ✅ Set up training environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_shape = (len(FEATURES),)  # State dimension
action_dim = 3  # Buy, Sell, Hold

# Initialize MR.Q agent
mrq_agent = MRQStockAgent(obs_shape=obs_shape, action_dim=action_dim, device=device)

# Initialize portfolio tracking
capital = 10000
holdings = 0
portfolio_values = []
buy_dates, sell_dates = [], []

# Training loop
for i in range(len(data) - 1):
    state = data.iloc[i].values
    next_state = data.iloc[i + 1].values

    action = mrq_agent.select_action(state)

    # Compute rewards
    reward = (next_state[0] - state[0]) * (1 if action == 1 else -1)

    if action == 1 and next_state[0] > state[0]:
        reward += 2
    elif action == 2 and next_state[0] < state[0]:
        reward += 2
    elif action == 1 and next_state[0] < state[0]:
        reward -= 2
    elif action == 2 and next_state[0] > state[0]:
        reward -= 2

    # Store experience in replay buffer
    done = (i == len(data) - 2)

    # mrq_agent.replay_buffer.add(state, action, reward, next_state, done, False)

    mrq_agent.replay_buffer.add(
        torch.tensor(state, dtype=torch.float32, device=mrq_agent.device).unsqueeze(0).clone().detach(),
        # ✅ Ensure correct shape
        int(action),  # ✅ Ensure action is passed as an integer
        torch.tensor(next_state, dtype=torch.float32, device=mrq_agent.device).unsqueeze(0).clone().detach(),
        # ✅ Ensure correct shape
        float(reward),  # ✅ Pass reward as a float (not a tensor)
        bool(done),  # ✅ Convert done flag to boolean
        False  # ✅ Convert truncated flag to boolean
    )

    # Train the agent
    if mrq_agent.replay_buffer.size > 32:
        for _ in range(10):
            mrq_agent.train()

    # Portfolio value tracking
    if action == 1 and capital >= state[0]:
        holdings += 1
        capital -= state[0]
        buy_dates.append(dates[i])
    elif action == 2 and holdings > 0:
        holdings -= 1
        capital += state[0]
        sell_dates.append(dates[i])

    total_value = capital + holdings * state[0]
    portfolio_values.append(total_value)

# Final portfolio value
final_value = capital + holdings * data.iloc[-1]["Close"]
print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Total Profit: ${final_value - 10000:.2f}")


