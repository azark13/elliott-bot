import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from smartmoneyconcepts import smc
from datetime import datetime

TELEGRAM_TOKEN = "8717849870:AAHOuOQLXSK3TFiEJpF4n0HJCqoXWPIhet4"
CHAT_ID = "901392944"
TOP_N = 20  # уменьшено для бесплатного тарифа

STABLECOINS = ['usdt', 'usdc', 'usd1', 'dai', 'busd', 'tusd', 'fdusd', 'usdd', 'usde', 'rusd']

def is_valid_pair(coin):
    name = coin.get("id", "").lower()
    symbol = coin.get("symbol", "").lower()
    volume = coin.get("total_volume", 0)
    if symbol in STABLECOINS or name in STABLECOINS or symbol.startswith('w'):
        return False
    if volume < 50_000_000:
        return False
    return True

def get_top_pairs(n=20):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "volume_desc", "per_page": 100, "page": 1, "sparkline": False}
    headers = {"accept": "application/json", "user-agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    data = resp.json()
    pairs = {}
    for coin in data:
        if not is_valid_pair(coin):
            continue
        pairs[f"{coin['symbol'].upper()}/USDT"] = {"id": coin["id"], "volume": coin.get("total_volume", 0)}
        if len(pairs) >= n:
            break
    return pairs

def load_pair_data(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    headers = {"accept": "application/json", "user-agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or len(data) < 20:
            return None
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["volume"] = 0
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df
    except:
        return None

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

def analyze_pair(pair_name, coin_id, volume):
    df = load_pair_data(coin_id, days=30)
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
    if volume > 1e9: score += 5
    
    return {
        'pair': pair_name, 'price': current_price, 'action': action,
        'ai_conf': ai_conf, 'entry': entry_price, 'stop': stop_price,
        'tp1': tp1, 'rr1': rr1, 'score': score, 'tv_link': f"https://www.tradingview.com/chart/?symbol=BINANCE:{pair_name.replace('/USDT', 'USDT')}&interval=240"
    }

# Главный цикл
print(f"🚀 Бот запущен: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
pairs = get_top_pairs(20)

results = []
for pair_name, info in pairs.items():
    result = analyze_pair(pair_name, info["id"], info["volume"])
    if result:
        results.append(result)

results.sort(key=lambda x: x['score'], reverse=True)
top5 = results[:5]

# Telegram
message = f"📊 <b>ТОП-5 СИГНАЛОВ</b>\n\n"
for i, r in enumerate(top5):
    medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i]
    pf = lambda x: f"${x:,.2f}" if x >= 1 else f"${x:.6f}"
    message += f"{medal} <b>{r['pair']}</b> | {r['score']}/45 | R:R 1:{r['rr1']:.1f}\n"
    message += f"   {'🟢 LONG' if r['action'] == 'LONG' else '🔴 SHORT'} | Вход: {pf(r['entry'])} | Стоп: {pf(r['stop'])}\n"
    message += f"   TP1: {pf(r['tp1'])} | 📈 <a href='{r['tv_link']}'>TradingView</a>\n\n"

# Сводка всех пар
message += "<b>📋 ВСЕ ПАРЫ:</b>\n<pre>"
for r in sorted(results, key=lambda x: x['pair']):
    message += f"{r['pair']:<10} {'LONG' if r['action']=='LONG' else 'SHORT':<6} {r['score']}/45  R:R 1:{r['rr1']:.1f}\n"
for p in sorted([p for p in pairs.keys() if p not in [r['pair'] for r in results]]):
    message += f"{p:<10} —     —\n"
message += f"</pre>\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}"

requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
              json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": False})

print("✅ Отправлено!")
