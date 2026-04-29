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

def analyze_pair(df, pair_name):
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
        return None
    
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
    }

# Загрузка CSV
print(f"🚀 {datetime.now().strftime('%d.%m.%Y %H:%M')}")
print("📂 Читаю crypto_data.csv...")

try:
    df_all = pd.read_csv("crypto_data.csv")
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_all.set_index('timestamp', inplace=True)
    
    symbols = df_all['symbol'].unique()
    print(f"✅ {len(df_all)} свечей, {len(symbols)} пар")
    
    results = []
    for sym in symbols:
        df_pair = df_all[df_all['symbol'] == sym].copy()
        if len(df_pair) < 20:
            continue
        
        col_map = {}
        for c in df_pair.columns:
            if c.lower() == 'open': col_map[c] = 'Open'
            elif c.lower() == 'high': col_map[c] = 'High'
            elif c.lower() == 'low': col_map[c] = 'Low'
            elif c.lower() == 'close': col_map[c] = 'Close'
            elif c.lower() == 'volume': col_map[c] = 'Volume'
        df_pair.rename(columns=col_map, inplace=True)
        
        pair_name = sym.replace("USDT", "/USDT")
        print(f"   {pair_name}...", end=" ")
        r = analyze_pair(df_pair, pair_name)
        if r:
            results.append(r)
            print(f"{r['action']} | {r['score']}/45 | R:R 1:{r['rr1']:.1f}")
        else:
            print("—")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    top5 = results[:5]
    
    # Telegram
    message = f"📊 <b>ТОП-5 СИГНАЛОВ</b> ({len(results)}/{len(symbols)} пар)\n\n"
    
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
            sym = r['pair'].replace("/", "")
            message += f"   📈 <a href='https://www.tradingview.com/chart/?symbol=BINANCE:{sym}&interval=240'>TradingView</a>\n\n"
    else:
        message += "⚠️ Нет сигналов.\n\n"
    
    message += "<b>📋 ВСЕ ПАРЫ:</b>\n<pre>"
    for r in sorted(results, key=lambda x: x['pair']):
        message += f"{r['pair']:<10} {r['action']:<6} {r['score']}/45  R:R 1:{r['rr1']:.1f}\n"
    for s in symbols:
        pn = s.replace("USDT", "/USDT")
        if pn not in [r['pair'] for r in results]:
            message += f"{pn:<10} —     —\n"
    message += f"</pre>\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    message += "<i>CSV-данные • GitHub Actions • каждые 4 часа</i>"
    
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                  json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": False})
    print(f"\n✅ Отправлено! ({len(results)} сигналов)")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
