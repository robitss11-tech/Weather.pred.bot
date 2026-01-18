import os, time, requests, pandas as pd, numpy as np
from kalshi_python import Kalshi
from telegram import Bot

kalshi = Kalshi()
kalshi.login(key_id=os.getenv('KALSHI_API_KEY'))
bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
chat_id = os.getenv('CHAT_ID')

def get_kmdw_cli():
  url = 'https://api.mesowest.utah.edu/api/stations?stid=KMDW&units=std&token=demo&obtime=*recent*'
  df = pd.read_json(url).temp.value.dropna()
  cli_max = np.max(df[-12:].values)  # pēdējās 1h
  return round(cli_max,1)

while True:
  cli = get_kmdw_cli()
  markets = kalshi.get_markets('kxhighchi')  # Chicago high today
  for m in markets:
    if abs(m.max_price - cli) < 1:
      ev = (m.yes_bid - 0.5) * 100
      if ev > 10:
        kalshi.buy(m.ticker, 2, 'yes')  # Auto trade
        bot.send_message(chat_id, f'KMDW CLI {cli}F → {m.ticker} EV+{ev:.0f}% → Bought 2')
  time.sleep(300)  # 5min
