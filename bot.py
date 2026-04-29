import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
TOP_N = 30

# Топ-30 пар (статический список)
TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
    "DOTUSDT", "BCHUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT",
    "XLMUSDT", "ETCUSDT", "FILUSDT", "ALGOUSDT", "VETUSDT",
    "ICPUSDT", "SANDUSDT", "AXSUSDT", "THETAUSDT", "FTMUSDT",
    "EGLDUSDT", "MANAUSDT", "GRTUSDT", "ZECUSDT", "AAVEUSDT"
]

def load_pair_data_bitget(symbol, days=30):
    """
    Загружает 4H свечи через Bitget API.
    Bitget использует endpoint /api/v2/spot/market/candles
    """
    # Собираем данные по кускам (Bitget отдаёт до 200 свечей за запрос)
    all_candles = []
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now().timestamp() - days * 86400))
    
    # Bitget требует гранулярность в секундах
    granularity = "4h"  # 4 часа
    
    while start_time < end_time:
        url = "https://api.bitget.com/api/v2/spot/market/candles"
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "endTime": str(end_time),
            "limit": "200"
        }
        headers = {"Accept": "application/json"}
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "00000" and data.get("data"):
                    candles = data["data"]
                    if not candles:
                        break
                    all_candles = candles + all_candles
                    # Обновляем end_time на время самой старой свечи
                    oldest_ts = int(candles[-1][0])
                    end_time = oldest_ts - 1
                else:
                    break
            else:
                print(f"HTTP{resp.status_code}", end="")
                break
        except Exception as e:
            print(f"Err:{e}", end="")
            break
    
    if len(all_candles) < 20:
        return None
    
    # Bitget возвращает: [timestamp, open, high, low, close, volume, ...]
    df = pd.DataFrame(all_candles, columns=["ts", "o", "h", "l", "c", "v", "usd_vol", "x"])
    df = df[["ts", "o", "h", "l", "c", "v"]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Bitget timestamp в секундах, конвертируем в миллисекунды
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
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
    df = load_pair_data_bitget(symbol, days=30)
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
print(f"📡 Bitget API • {TOP_N} пар")

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
message = f"📊 <b>ТОП-5 СИГНАЛОВ</b> ({len(results)}/{TOP_N} пар)\n\n"

if top5:
    for i, r in enumerate(top5):
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i]
        pf = lambda x: f"${x:,.2f}" if x >= 1 else f"${x:.6f}"
        message += f"{medal} <b>{r['pair']}</b> | {r['score']}/45 | R:R 1:{r['rr1']:.1f}\n"
        message += f"   {'🟢 LONG' if r['action'] == 'LONG' else '🔴 SHORT'} | AI: {r['ai_conf']:.0%}\n"
        message += f"   ┌─ Вход: <b>{pf(r['entry'])}</b>\n"
        message += f"   ├─ Стоп: {pf(r['stop'])}\n"
        message += f"   ├─ TP1:  {pf(r['tp1'])}\n"
        message += f"   └─ TP2:  {pf(r['tp2'])}\n"
        message += f"   📈 <a href='https://www.tradingview.com/chart/?symbol=BITGET:{r['symbol']}&interval=240'>TradingView</a>\n\n"
else:
    message += "⚠️ Нет сигналов.\n\n"

message += "<b>📋 ВСЕ ПАРЫ:</b>\n<pre>"
for r in sorted(results, key=lambda x: x['pair']):
    message += f"{r['pair']:<10} {r['action']:<6} {r['score']}/45  R:R 1:{r['rr1']:.1f}\n"
for s in TOP_SYMBOLS[:TOP_N]:
    pn = s.replace("USDT", "/USDT")
    if pn not in [r['pair'] for r in results]:
        message += f"{pn:<10} —     —\n"
message += f"</pre>\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
message += "<i>Bitget API • GitHub Actions • каждые 4 часа</i>"

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": False}
resp = requests.post(url, json=payload)

if resp.status_code == 200:
    print(f"\n✅ Отправлено! ({len(results)} сигналов)")
else:
    print(f"\n❌ Ошибка: {resp.text}")
