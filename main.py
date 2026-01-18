import os
import time
import requests
import pandas as pd
import numpy as np
from telegram import Bot
from kalshi_python import Configuration, KalshiClient  # Pareizais imports!

# Debug prints
print("Keys OK:", bool(os.getenv('KALSHI_KEY_ID')))
print("Telegram OK:", bool(os.getenv('TELEGRAM_TOKEN')))

# Kalshi setup
config = Configuration()
config.api_key_id = os.getenv('KALSHI_KEY_ID')
config.private_key_pem = os.getenv('KALSHI_PRIVATE_KEY_PEM')  # PEM formātā!
client = KalshiClient(config)  # NE 'Kalshi()', bet 'KalshiClient(config)'

bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
chat_id = int(os.getenv('CHAT_ID'))

def get_kmdw_cli():
    try:
        df = pd.read_csv('ptype_meteonetwork_IL_ASOSzstation_MDWsts_2026-01-16-0000ets_2026-01-17-0000r_tdpi_400cb_1.csv')
        # Jūsu analīze šeit...
        return df
    except Exception as e:
        print(f"CSV error: {e}")
        return None

# Galvenā cilpa (piemērs)
while True:
    df = get_kmdw_cli()
    if df is not None:
        # Kalshi piemērs: balanss
        balance = client.get_balance()
        message = f"KMDW CLI data loaded. Balance: {balance}"
        bot.send_message(chat_id=chat_id, text=message)
    time.sleep(300)  # 5min
