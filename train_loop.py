import os
import time
import pandas as pd
import yfinance as yf
from stable_baselines3 import A2C
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.finrl_meta.preprocessor.preprocessors import data_split

TICKERS = ["AAPL", "TSLA", "NVDA"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
INITIAL_ACCOUNT_BALANCE = 100000

LOG_DIR = "logs"
MODEL_DIR = "model_checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_data(tickers):
    data = yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=True)
    df_list = []
    for ticker in tickers:
        d = data[ticker].copy()
        d['tic'] = ticker
        d.reset_index(inplace=True)
        df_list.append(d)
    return pd.concat(df_list)

def train_model(df, timesteps=10000):
    train, _ = data_split(df, START_DATE, END_DATE)
    env = StockTradingEnv(df=train,
                          stock_dim=len(TICKERS),
                          hmax=10,
                          initial_amount=INITIAL_ACCOUNT_BALANCE,
                          buy_cost_pct=0.001,
                          sell_cost_pct=0.001,
                          state_space=1 + 2 * len(TICKERS),
                          action_space=len(TICKERS),
                          reward_scaling=1e-4)
    model = A2C("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_model(model, df):
    _, trade = data_split(df, START_DATE, END_DATE)
    env = StockTradingEnv(df=trade,
                          stock_dim=len(TICKERS),
                          hmax=10,
                          initial_amount=INITIAL_ACCOUNT_BALANCE,
                          buy_cost_pct=0.001,
                          sell_cost_pct=0.001,
                          state_space=1 + 2 * len(TICKERS),
                          action_space=len(TICKERS),
                          reward_scaling=1e-4)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    return env.asset_memory[-1]

def train_forever():
    iteration = 0
    best_reward = 0
    df = fetch_data(TICKERS)

    while True:
        iteration += 1
        print(f"🔁 Iteracja {iteration} — trening nowego modelu...")
        model = train_model(df)
        result = evaluate_model(model, df)
        print(f"📈 Wynik: {result:.2f} USD")

        if result > best_reward:
            best_reward = result
            model_path = os.path.join(MODEL_DIR, f"best_model_iter_{iteration}.zip")
            model.save(model_path)
            print(f"💾 Zapisano nowy najlepszy model: {model_path}")

        with open(os.path.join(LOG_DIR, "training_log.txt"), "a") as log:
            log.write(f"Iteracja {iteration}, wynik: {result:.2f}\n")

        time.sleep(300)  # 5 minut przerwy

if __name__ == "__main__":
    train_forever()
