import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import time
import math

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

def analyze_pair(df, symbol):
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
    
    pair_name = symbol.replace("USDT", "/USDT")
    
    return {
        'pair': pair_name, 'symbol': symbol, 'price': current_price,
        'action': action, 'ai_conf': ai_conf,
        'entry': entry, 'stop': stop, 'tp1': tp1, 'tp2': tp2,
        'rr1': rr1, 'score': score, 'high': high, 'low': low
    }

def generate_tv_link(symbol, entry, stop, tp1, tp2):
    """Ссылка на TradingView с уровнями в описании."""
    pair_name = symbol.replace("USDT", "/USDT")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval=240"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": False}
    requests.post(url, json=payload, timeout=10)

def format_price(p):
    if p >= 100: return f"${p:,.0f}"
    elif p >= 1: return f"${p:,.2f}"
    elif p >= 0.01: return f"${p:,.4f}"
    else: return f"${p:,.8f}"

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
        
        col_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        df_pair.rename(columns={k:v for k,v in col_map.items() if k in df_pair.columns}, inplace=True)
        
        print(f"   {sym}...", end=" ")
        r = analyze_pair(df_pair, sym)
        if r:
            results.append(r)
            print(f"{r['action']} | {r['score']}/45 | R:R 1:{r['rr1']:.1f}")
        else:
            print("—")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # --- ОБЩАЯ СВОДКА (одно сообщение) ---
    summary = f"📊 <b>СВОДКА ПО {len(results)} ПАРАМ</b>\n\n"
    
    # Топ-3
    summary += "<b>🏆 ТОП-3:</b>\n"
    for i, r in enumerate(results[:3]):
        medal = ['🥇','🥈','🥉'][i]
        summary += f"{medal} <b>{r['pair']}</b> | {r['score']}/45 | {'🟢' if r['action']=='LONG' else '🔴'} R:R 1:{r['rr1']:.1f}\n"
    
    # Все пары таблицей
    summary += "\n<b>📋 ВСЕ ПАРЫ:</b>\n<pre>"
    for r in sorted(results, key=lambda x: x['pair']):
        summary += f"{r['pair']:<10} {r['action']:<6} {r['score']}/45  R:R 1:{r['rr1']:.1f}\n"
    summary += f"</pre>\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    summary += "<i>Сигналы по каждой паре — ниже ↓</i>"
    
    send_telegram(summary)
    print("📨 Сводка отправлена")
    
    # --- ОТДЕЛЬНЫЕ СООБЩЕНИЯ ДЛЯ КАЖДОЙ ПАРЫ ---
    for i, r in enumerate(results):
        # Задержка каждые 15 сообщений (лимит Telegram)
        if i > 0 and i % 15 == 0:
            print("   ⏳ Пауза 2 сек...")
            time.sleep(2)
        
        tv_link = generate_tv_link(r['symbol'], r['entry'], r['stop'], r['tp1'], r['tp2'])
        
        # Описание уровней для TradingView
        levels_desc = (
            f"Уровни:\n"
            f"─ Вход: {format_price(r['entry'])}\n"
            f"─ Стоп: {format_price(r['stop'])}\n"
            f"─ TP1:  {format_price(r['tp1'])}\n"
            f"─ TP2:  {format_price(r['tp2'])}\n"
            f"Диапазон: {format_price(r['low'])} → {format_price(r['high'])}"
        )
        
        stars = '⭐' * max(1, min(5, r['score'] // 9 + 1))
        
        message = (
            f"{'🟢 LONG' if r['action'] == 'LONG' else '🔴 SHORT'} | <b>{r['pair']}</b>\n"
            f"{stars} {r['score']}/45 | R:R 1:{r['rr1']:.1f} | AI: {r['ai_conf']:.0%}\n\n"
            f"<b>📍 УРОВНИ:</b>\n"
            f"┌─ Вход: <b>{format_price(r['entry'])}</b>\n"
            f"├─ Стоп: {format_price(r['stop'])}\n"
            f"├─ TP1:  {format_price(r['tp1'])}\n"
            f"└─ TP2:  {format_price(r['tp2'])}\n\n"
            f"📈 <a href='{tv_link}'>График TradingView</a>\n"
            f"<i>Добавьте уровни вручную на график ↑</i>"
        )
        
        send_telegram(message)
        print(f"   📤 {r['pair']}")
    
    print(f"\n✅ Готово! {len(results)} сигналов отправлено")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
