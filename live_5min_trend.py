# live_5min_trend.py
import asyncio
import json
from collections import deque
from datetime import datetime, timedelta
import math

import numpy as np
from river import ensemble, linear_model, preprocessing, metrics
import websockets

# ----------------- CONFIG -----------------
WEBSOCKET_URL = "wss://YOUR_PROVIDER_WEBSOCKET"  # <-- replace with your provider
SYMBOL = "AAPL"  # symbol to subscribe / filter
BAR_SECONDS = 60  # aggregate into 1-minute bars
PRED_HORIZON = timedelta(minutes=5)  # predict next 5 minutes
FEATURE_WINDOW_MINUTES = 15  # how many minutes of bars to use for features
LABEL_THRESHOLD = 0.001  # 0.1% threshold -> up / neutral / down

# online model choice
# AdaptiveRandomForest for streaming robustness (can be heavier). Use LogisticRegression if you want tiny.
model = preprocessing.StandardScaler() | ensemble.AdaptiveRandomForestClassifier(n_estimators=10, seed=42)
metric_acc = metrics.Accuracy()
metric_bal = metrics.BalancedAccuracy()  # optional

# ----------------- in-memory storage -----------------
# store finalized minute bars as dicts: {ts:datetime, open, high, low, close, volume, vwap}
bars = deque(maxlen=500)  # keep last 500 minutes (plenty)
# a single current bar being built
current_bar = None  # dict or None

# pending snapshots waiting for their label after PRED_HORIZON
# each item: {"created_at": datetime, "features_at_creation": dict, "price_at_creation": float, "target_time": datetime}
pending_snapshots = deque()

# ----------------- helpers -----------------
def start_new_bar(ts, price, size):
    return {
        "ts": ts,        # minute-aligned datetime
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": size if size is not None else 0.0,
        "vwap_numer": price * (size if size is not None else 0.0),
        "vwap_denom": (size if size is not None else 0.0)
    }

def update_bar(bar, price, size):
    bar["high"] = max(bar["high"], price)
    bar["low"] = min(bar["low"], price)
    bar["close"] = price
    if size is not None:
        bar["volume"] += size
        bar["vwap_numer"] += price * size
        bar["vwap_denom"] += size

def finalize_bar(bar):
    # compute vwap, remove numer/denom fields
    vwap = bar["vwap_numer"] / bar["vwap_denom"] if bar["vwap_denom"] > 0 else bar["close"]
    finalized = {
        "ts": bar["ts"],
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "volume": float(bar["volume"]),
        "vwap": float(vwap)
    }
    return finalized

def compute_features_from_bars(bars_deque):
    # bars_deque is ordered oldest -> newest
    arr_close = np.array([b["close"] for b in bars_deque])
    arr_vol = np.array([b["volume"] for b in bars_deque])
    arr_vwap = np.array([b["vwap"] for b in bars_deque])

    n = len(arr_close)
    if n == 0:
        return {}

    last = float(arr_close[-1])
    features = {}
    # basic features
    features["n_minutes"] = n
    features["last_price"] = last
    # returns: last vs 1,3,5,10 minutes if available
    features["ret_1m"] = float((arr_close[-1] / arr_close[-2]) - 1.0) if n >= 2 else 0.0
    features["ret_3m"] = float((arr_close[-1] / arr_close[-4]) - 1.0) if n >= 4 else 0.0
    features["ret_5m"] = float((arr_close[-1] / arr_close[-6]) - 1.0) if n >= 6 else 0.0
    features["ret_10m"] = float((arr_close[-1] / arr_close[-11]) - 1.0) if n >= 11 else 0.0

    # moving averages and gaps
    def ma(k):
        if n >= k:
            return float(np.mean(arr_close[-k:]))
        else:
            return float(np.mean(arr_close))
    features["ma_3"] = ma(3)
    features["ma_5"] = ma(5)
    features["ma_10"] = ma(10)
    features["ma_gap_3_10"] = features["ma_3"] - features["ma_10"]

    # volatility (std)
    features["std_3"] = float(np.std(arr_close[-3:])) if n>=3 else 0.0
    features["std_10"] = float(np.std(arr_close[-10:])) if n>=10 else float(np.std(arr_close))

    # volume features
    features["vol_1m"] = float(arr_vol[-1])
    features["vol_avg_5m"] = float(np.mean(arr_vol[-5:])) if n>=5 else float(np.mean(arr_vol))
    features["vol_ratio_1m_avg5m"] = (features["vol_1m"] / (features["vol_avg_5m"] + 1e-9))

    # vwap drift
    features["vwap_last"] = float(arr_vwap[-1])
    features["vwap_gap_price"] = features["vwap_last"] - last

    # momentum sign counts over small window
    ups = 0
    downs = 0
    for i in range(1, min(n,6)):
        if arr_close[-i] > arr_close[-i-1]:
            ups += 1
        elif arr_close[-i] < arr_close[-i-1]:
            downs += 1
    features["momentum_up_count_5"] = ups
    features["momentum_down_count_5"] = downs

    return features

def make_label(price_at_creation, future_price, threshold=LABEL_THRESHOLD):
    ret = (future_price / price_at_creation) - 1.0
    if ret > threshold:
        return "up"
    if ret < -threshold:
        return "down"
    return "neutral"

# ----------------- parse provider messages -----------------
def parse_message(msg_text):
    # Generic parser: must be adapted to your provider's JSON structure.
    # Expect fields: symbol, price, size(optional), timestamp(optional).
    try:
        m = json.loads(msg_text)
    except Exception:
        return None
    # Example generic trade message patterns - modify as needed:
    # { "type":"trade", "symbol":"AAPL", "price":123.45, "size":100, "t":"2025-09-24T09:30:01.123Z" }
    sym = m.get("symbol") or m.get("s") or m.get("sym")
    if not sym:
        return None
    price = m.get("price") or m.get("p")
    size = m.get("size") or m.get("v") or m.get("volume")
    ts = m.get("t") or m.get("timestamp") or m.get("time")
    if price is None:
        return None
    # parse timestamp -> datetime (UTC)
    if ts is None:
        dt = datetime.utcnow()
    else:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            # if it's epoch ms
            try:
                ms = int(ts)
                dt = datetime.utcfromtimestamp(ms/1000.0)
            except Exception:
                dt = datetime.utcnow()
    return {"symbol": sym, "price": float(price), "size": float(size) if size is not None else None, "ts": dt}

# ----------------- main websocket consumer -----------------
async def run():
    global current_bar
    print("Connecting to", WEBSOCKET_URL)
    async with websockets.connect(WEBSOCKET_URL, ping_interval=20) as ws:
        # If provider needs auth or subscribe, send here. Example:
        # await ws.send(json.dumps({"action":"subscribe","symbols":[SYMBOL]}))
        # --- event loop for incoming ticks ---
        async for raw in ws:
            parsed = parse_message(raw)
            if not parsed:
                continue
            if parsed["symbol"] != SYMBOL:
                continue

            tick_time = parsed["ts"]
            price = parsed["price"]
            size = parsed["size"]

            # align to minute start
            minute_ts = tick_time.replace(second=0, microsecond=0)

            if (current_bar is None) or (minute_ts != current_bar["ts"]):
                # finalize previous bar (if any)
                if current_bar is not None:
                    finalized = finalize_bar(current_bar)
                    bars.append(finalized)
                    # when a new bar finalizes, check pending snapshots for labels
                    now_bar_time = finalized["ts"]
                    # label pending snapshots whose target_time <= now_bar_time + BAR_SECONDS (use bar timestamp)
                    while pending_snapshots and pending_snapshots[0]["target_time"] <= now_bar_time:
                        snapshot = pending_snapshots.popleft()
                        price_then = snapshot["price_at_creation"]
                        future_price = finalized["close"]  # label with the close of the bar at target time
                        y = make_label(price_then, future_price)
                        x = snapshot["features_at_creation"]
                        # update model (learn)
                        try:
                            model.learn_one(x, y)
                        except Exception as e:
                            print("Learn error:", e)
                        # update metrics
                        y_pred = model.predict_one(x)
                        metric_acc.update(y, y_pred)
                        metric_bal.update(y, y_pred)
                        print(f"[{snapshot['created_at'].isoformat()}] Trained on label={y}; pred(before)={y_pred}; acc={metric_acc.get():.4f}")

                # start a fresh bar
                current_bar = start_new_bar(minute_ts, price, size)
            else:
                # update current bar
                update_bar(current_bar, price, size)

            # After updating bar, compute features from existing bars + current in-progress bar.
            # For feature stability use bars + current projected final close = latest price
            window_bars = list(bars)[- (FEATURE_WINDOW_MINUTES - 1 ):] if len(bars) >= (FEATURE_WINDOW_MINUTES - 1) else list(bars)
            # include an imaginary bar representing the current minute using current close=price and volume from current bar
            simulated_current = {
                "ts": current_bar["ts"],
                "open": current_bar["open"],
                "high": current_bar["high"],
                "low": current_bar["low"],
                "close": current_bar["close"],
                "volume": current_bar["volume"],
                "vwap": (current_bar["vwap_numer"] / current_bar["vwap_denom"]) if current_bar["vwap_denom"]>0 else current_bar["close"]
            }
            window_bars = window_bars + [simulated_current]
            # keep only the latest FEATURE_WINDOW_MINUTES
            if len(window_bars) > FEATURE_WINDOW_MINUTES:
                window_bars = window_bars[-FEATURE_WINDOW_MINUTES:]

            feats = compute_features_from_bars(window_bars)

            # Make prediction for next 5 minutes
            pred = model.predict_one(feats)
            try:
                proba = model.predict_proba_one(feats)
            except Exception:
                proba = None

            # Create a snapshot for later labeling: target_time = current minute_ts + PRED_HORIZON
            snapshot = {
                "created_at": minute_ts,
                "features_at_creation": feats,
                "price_at_creation": float(price),
                "target_time": minute_ts + PRED_HORIZON
            }
            pending_snapshots.append(snapshot)

            # Keep pending bounded (say we don't need more than 500 pending)
            while len(pending_snapshots) > 500:
                pending_snapshots.popleft()

            # Logging
            print(f"[{tick_time.isoformat()}] price={price:.4f} pred={pred} proba={proba} features_n={len(window_bars)} pending={len(pending_snapshots)} acc={metric_acc.get():.4f}")

# ----------------- run -----------------
if __name__ == "__main__":
    asyncio.run(run())