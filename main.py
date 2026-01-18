import os
print("Keys OK:", bool(os.getenv('KALSHI_KEY_ID')))
print("Telegram OK:", bool(os.getenv('TELEGRAM_TOKEN')))
import time
import requests
import pandas as pd
import numpy as np
from kalshi_python import Kalshi
from telegram import Bot

# Auth
exchange = Kalshi()
exchange.login(key_id=os.getenv('KALSHI_KEY_ID'), 
               private_key=os.getenv('KALSHI_PRIVATE_KEY'))
bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
chat_id = int(os.getenv('CHAT_ID'))

def get_kmdw_cli():
  # Use tavu CSV file:119
  try:
    df = pd.read_csv('ptype_meteo__network_IL_ASOS__zstation_MDW__sts_2026-01-16-0000__ets_2026-01-17-0000___r_t__dpi_400___cb_1.csv')

