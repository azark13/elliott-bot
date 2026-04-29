import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHATID", "")
TOP_N = 30

# Топ-30 пар по объёму на Binance (статический список, обновлён 2026)
TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
    "DOTUSDT", "BCHUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT",
    "XLMUSDT", "ETCUSDT", "FILUSDT", "ALGOUSDT", "VETUSDT",
    "ICPUSDT", "SANDUSDT", "AXSUSDT", "THETAUSDT", "FTMUSDT",
    "EGLDUSDT", "MANAUSDT", "GRTUSDT", "ZECUSDT", "AAVEUSDT"
]

def load_pair_data(symbol, days=30):
    """Загружает 4H свечи через Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now().timestamp() - days * 86400) * 1000)
    
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": "4h",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 500
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    break
                all_data = data + all_data
                end_time = data[0][0] - 1
            else:
                break
        except:
            break
    
    if len(all_data) < 20:
        return None
    
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df

def ai_predict(df):
    if len(df) < 30:
        return None, None
    df_ai = df.copy()
    df_ai['returns'] = df_ai['Close'].pct_change()
    df_ai['high_low'] = (df_ai['High'] - df_ai['Low']) / df_ai['Close']
    df_ai['ma10'] = df_ai['Close'].rolling(10).mean()
    df_ai['dist_ma10'] = (df_ai['Close'] - df_ai['ma10']) / df_ai['Close'] * 100
    delta = df_ai['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df_ai['rsi'] = 100 - (100 / (1 + gain/(loss + 0.0001)))
    df_ai['target'] = (df_ai['Close'].shift(-1) > df_ai['Close']).astype(int)
    df_ai = df_ai.dropna()
    if len(df_ai) < 20:
        return None, None
    features = ['returns', 'high_low', 'dist_ma10', 'rsi']
    X = df_ai[features].values
    y = df_ai['target'].values
    try:
        model = xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X[:-1], y[:-1])
        pred = model.predict(X[-1:])[0]
        conf = max(model.predict_proba(X[-1:])[0])
        return pred, conf
    except:
        return None, None

def analyze_pair(symbol):
    df = load_pair_data(symbol, days=30)
    if df is None:
        return None
    
    current_price = df['Close'].iloc[-1]
    ai_pred, ai_conf = ai_predict(df)
    if ai_pred is None:
        return None
    
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                    abs(df['Low'] - df['Close'].shift(1))))
    atr = df['TR'].tail(14).mean()
    
    recent = df.iloc[-30:]
    high = recent['High'].max()
    low = recent['Low'].min()
    diff = high - low
    
    if diff <= 0 or atr <= 0:
        return None
    
    if ai_pred == 1:
        entry_price = low + diff * 0.382
        stop_price = entry_price - atr * 2
        tp1 = entry_price + atr * 3
        tp2 = entry_price + atr * 5
        action = "LONG"
    else:
        entry_price = high - diff * 0.382
        stop_price = entry_price + atr * 2
        tp1 = entry_price - atr * 3
        tp2 = entry_price - atr * 5
        action = "SHORT"
    
    risk = abs(entry_price - stop_price)
    reward1 = abs(tp1 - entry_price)
    
    if risk <= 0:
        return None
    
    rr1 = reward1 / risk
    score = 0
    if ai_conf > 0.65: score += 25
    elif ai_conf > 0.55: score += 15
    else: score += 5
    if rr1 >= 1.5: score += 15
    elif rr1 >= 1.0: score += 8
    
    pair_name = symbol.replace("USDT", "/USDT")
    
    return {
        'pair': pair_name, 'price': current_price, 'action': action,
        'ai_conf': ai_conf, 'entry': entry_price, 'stop': stop_price,
        'tp1': tp1, 'tp2': tp2, 'rr1': rr1, 'score': score,
        'symbol': symbol
    }

# Главный цикл
print(f"🚀 {datetime.now().strftime('%d.%m.%Y %H:%M')}")
print(f"📡 Binance API • {TOP_N} пар")

results = []
for symbol in TOP_SYMBOLS[:TOP_N]:
    print(f"   {symbol}...", end=" ")
    result = analyze_pair(symbol)
    if result:
        results.append(result)
        print(f"{result['action']} | {result['score']}/45 | R:R 1:{result['rr1']:.1f}")
    else:
        print("—")

results.sort(key=lambda x: x['score'], reverse=True)
top5 = results[:5]

# Сводка
message = f"📊 <b>ТОП-5 СИГНАЛОВ</b> ({len(results)}/{len(TOP_SYMBOLS[:TOP_N])})\n\n"

for i, r in enumerate(top5):
    medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i]
    pf = lambda x: f"${x:,.2f}" if x >= 1 else f"${x:.6f}"
    message += f"{medal} <b>{r['pair']}</b> | {r['score']}/45 | R:R 1:{r['rr1']:.1f}\n"
    message += f"   {'🟢 LONG' if r['action'] == 'LONG' else '🔴 SHORT'} | AI: {r['ai_conf']:.0%}\n"
    message += f"   ┌─ Вход: <b>{pf(r['entry'])}</b>\n"
    message += f"   ├─ Стоп: {pf(r['stop'])}\n"
    message += f"   ├─ TP1:  {pf(r['tp1'])}\n"
    message += f"   └─ TP2:  {pf(r['tp2'])}\n"
    message += f"   📈 <a href='https://www.tradingview.com/chart/?symbol=BINANCE:{r['symbol']}&interval=240'>TradingView</a>\n\n"

message += "<b>📋 ВСЕ ПАРЫ:</b>\n<pre>"
for r in sorted(results, key=lambda x: x['pair']):
    message += f"{r['pair']:<10} {'LONG' if r['action']=='LONG' else 'SHORT':<6} {r['score']}/45  R:R 1:{r['rr1']:.1f}\n"
for s in TOP_SYMBOLS[:TOP_N]:
    pn = s.replace("USDT", "/USDT")
    if pn not in [r['pair'] for r in results]:
        message += f"{pn:<10} —     —\n"
message += f"</pre>\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
message += "<i>Binance API • GitHub Actions • каждые 4 часа</i>"

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": False}
resp = requests.post(url, json=payload)

if resp.status_code == 200:
    print("✅ Отправлено!")
else:
    print(f"❌ Ошибка: {resp.text}")
