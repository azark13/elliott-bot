import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

def ai_predict(df):
    if len(df) < 30:
        return None, None, "мало данных"
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
        return None, None, f"после очистки всего {len(df_ai)}"
    features = ['returns', 'high_low', 'dist_ma10', 'rsi']
    X = df_ai[features].values
    y = df_ai['target'].values
    try:
        model = xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X[:-1], y[:-1])
        pred = model.predict(X[-1:])[0]
        conf = max(model.predict_proba(X[-1:])[0])
        return pred, conf, "ок"
    except Exception as e:
        return None, None, f"ошибка модели: {e}"

def analyze_pair(df, pair_name):
    current_price = df['Close'].iloc[-1]
    ai_pred, ai_conf, ai_status = ai_predict(df)
    if ai_pred is None:
        return None, ai_status
    
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                    abs(df['Low'] - df['Close'].shift(1))))
    atr = df['TR'].tail(14).mean()
    
    recent = df.iloc[-30:]
    high = recent['High'].max()
    low = recent['Low'].min()
    diff = high - low
    
    if diff <= 0 or atr <= 0:
        return None, f"diff={diff:.2f} atr={atr:.2f}"
    
    if ai_pred == 1:
        entry = low + diff * 0.382
        stop = entry - atr * 2
        tp1 = entry + atr * 3
        tp2 = entry + atr * 5
        action = "LONG"
    else:
        entry = high - diff * 0.382
        stop = entry + atr * 2
        tp1 = entry - atr * 3
        tp2 = entry - atr * 5
        action = "SHORT"
    
    risk = abs(entry - stop)
    if risk <= 0:
        return None, f"risk={risk}"
    
    rr1 = abs(tp1 - entry) / risk
    score = 0
    if ai_conf > 0.65: score += 25
    elif ai_conf > 0.55: score += 15
    else: score += 5
    if rr1 >= 1.5: score += 15
    elif rr1 >= 1.0: score += 8
    
    return {
        'pair': pair_name, 'price': current_price, 'action': action,
        'ai_conf': ai_conf, 'entry': entry, 'stop': stop,
        'tp1': tp1, 'tp2': tp2, 'rr1': rr1, 'score': score
    }, "ок"

# Загрузка CSV
print(f"🚀 {datetime.now().strftime('%d.%m.%Y %H:%M')}")
print("📂 Читаю crypto_data.csv...")

try:
    df_all = pd.read_csv("crypto_data.csv")
    print(f"   Колонки: {list(df_all.columns)}")
    print(f"   Первая строка: {df_all.iloc[0].to_dict()}")
    
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_all.set_index('timestamp', inplace=True)
    
    symbols = df_all['symbol'].unique()
    print(f"✅ {len(df_all)} свечей, {len(symbols)} пар")
    
    results = []
    for sym in symbols[:5]:  # ТОЛЬКО 5 ДЛЯ ТЕСТА
        df_pair = df_all[df_all['symbol'] == sym].copy()
        print(f"   {sym}: {len(df_pair)} свечей, колонки: {list(df_pair.columns)}")
        
        if len(df_pair) < 20:
            print(f"      ⚠️ мало данных")
            continue
        
        # Переименовываем колонки
        col_map = {}
        for c in df_pair.columns:
            if c.lower() == 'open':
                col_map[c] = 'Open'
            elif c.lower() == 'high':
                col_map[c] = 'High'
            elif c.lower() == 'low':
                col_map[c] = 'Low'
            elif c.lower() == 'close':
                col_map[c] = 'Close'
            elif c.lower() == 'volume':
                col_map[c] = 'Volume'
        df_pair.rename(columns=col_map, inplace=True)
        
        pair_name = sym.replace("USDT", "/USDT")
        r, status = analyze_pair(df_pair, pair_name)
        if r:
            results.append(r)
            print(f"      {r['action']} | {r['score']}/45")
        else:
            print(f"      — ({status})")
    
    if results:
        print(f"\n✅ Найдено {len(results)} сигналов")
    else:
        print(f"\n⚠️ Сигналов нет")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
