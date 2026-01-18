import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import asyncio
import uvicorn
from fastapi import FastAPI
from kalshi_python import Configuration, KalshiClient
from telegram import Bot
from telegram.error import TelegramError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
client = None
bot_obj = None
chat_id = os.getenv('CHAT_ID')
model = None

@app.get("/")
async def root():
    return {"status": "KMDW Kalshi Bot live", "kalshi_ready": client is not None}

def init_kalshi():
    global client
    kalshi_key_id = os.getenv('KALSHI_KEY_ID')
    kalshi_priv_key = os.getenv('KALSHI_PRIVATE_KEY')
    telegram_token = os.getenv('TELEGRAM_TOKEN')
    global bot_obj
    bot_obj = Bot(token=telegram_token) if telegram_token else None
    if kalshi_key_id and kalshi_priv_key:
        config = Configuration()
        config.api_key_id = kalshi_key_id
        config.private_key_pem = kalshi_priv_key
        client = KalshiClient(config)
        balance_resp = client.get_balance()
        logger.info(f"Kalshi init OK. Balance: ${balance_resp.balance / 100:.2f}")
        asyncio.create_task(bot_obj.send_message(chat_id=chat_id, text="Kalshi Bot restarted!"))

def get_kmdw_cli():
    # NOAA CSV vai METAR proxy
    url = "https://mesonet.agron.iastate.edu/request/download.phtml?network=IL_ASOS&station=MDW&data=all&start=20260101&end=today&format=csv"  # Fix URL
    try:
        resp = requests.get(url)
        df = pd.read_csv(StringIO(resp.text), skiprows=1)
        df['tmpf'] = pd.to_numeric(df['tmpf'], errors='coerce')
        df['sknt'] = pd.to_numeric(df['sknt'], errors='coerce')
        cli = df['tmpf'].tail(24).max()  # CLI proxy: recent high
        return cli
    except Exception as e:
        logger.error(f"CLI fetch error: {e}")
        return None

def fetch_kmdw_data():
    url = "https://mesonet.agron.iastate.edu/request/asos/1min.phtml?station=KMDW"
    try:
        resp = requests.get(url)
        df = pd.read_csv(StringIO(resp.text))
        return df.dropna()
    except:
        return pd.DataFrame()

def train_model():
    global model
    df = fetch_kmdw_data()
    if len(df) > 100:
        df['CLI_proxy'] = df['tmpf'].rolling(24).max().shift(-1)
        df = df.dropna()
        X = df[['tmpf', 'sknt']]
        y = df['CLI_proxy']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        joblib.dump(model, 'model.pkl')
        logger.info(f"Model accuracy: {acc:.2f}")
        return acc
    return 0.0

def predict_cli(metar):
    global model
    if model is None:
        try:
            model = joblib.load('model.pkl')
        except:
            model = None
    if model:
        pred = model.predict([[metar['tmpf'], metar['sknt']]])[0]
        return pred[0] if isinstance(pred, np.ndarray) else pred
    return 45.0  # Chicago baseline

def download_era5():
    try:
        import cdsapi
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': '2m_temperature',
                'year': ['2026', '2025'],
                'month': ['01'],
                'day': ['01/to/31'],
                'time': ['12:00'],
                'area': [41.785, -87.752, 41.785, -87.752],
                'format': 'netcdf',
            },
            'era5_kmdw.nc')
        logger.info("ERA5 downloaded!")
        return True
    except ImportError:
        logger.warning("cdsapi not installed - skip ERA5")
        return False
    except Exception as e:
        logger.error(f"ERA5 error: {e}")
        return False

def predict_outcome(ticker):
    if 'chi' in ticker:
        metar = {'tmpf': 25, 'sknt': 10}
        return predict_cli(metar) > 20  # Threshold
    return 0.6  # Baseline

def kelly_size(p, b):
    f = (p * b - 1) / (b - 1)
    return max(0, min(10, f * 100))  # Contracts

async def main_loop():
    init_kalshi()
    train_model()
    download_era5()  # ERA5 init
    while True:
        cli = get_kmdw_cli()
        if client and cli is not None:
            balance_resp = client.get_balance()
            msg = f"KMDW CLI: {cli:.1f}°F, Balance: ${balance_resp.balance / 100:.2f}"
            try:
                await bot_obj.send_message(chat_id=chat_id, text=msg)
            except:
                pass

            # Multi-market scan
            markets_resp = client.list_markets({'category': 'climate', 'status': 'open'})
            for market in markets_resp.markets[:10]:
                ticker = market.ticker
                if any(word in ticker.lower() for word in ['temperature', 'rain', 'hurricane', 'chi']):
                    yes_price = market.yes_bid
                    pred_prob = predict_outcome(ticker)
                    ev = pred_prob - yes_price / 100
                    if ev > 0.05:
                        size = kelly_size(pred_prob, yes_price)
                        if size > 0:
                            order = client.buy_order(ticker, side='yes', count=int(size), type='market')
                            trade_msg = f"Buy {ticker} {size} EV:{ev:.2%} Order:{order.order_id}"
                            await bot_obj.send_message(chat_id=chat_id, text=trade_msg)
                            logger.info(trade_msg)
        await asyncio.sleep(300)  # 5min

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
