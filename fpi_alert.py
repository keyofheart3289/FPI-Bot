#!/usr/bin/env python3
"""
FPI v3 Alert Bot — GitHub Actions + Telegram
Replicate FPI v3 Hold Score, detect signal edge, push Telegram alert
"""
import os, json, requests
import numpy as np
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOLS = ["ETH-USDT-SWAP", "BTC-USDT-SWAP", "SOL-USDT-SWAP"]
TIMEFRAME = "1H"   # OKX dùng chữ hoa: 1H, 5m, 15m, 4H

# FPI params — khớp với indicator
Z_LEN, WICK_LEN, TDR_LEN = 20, 10, 14
WZ, WW, WT = 0.40, 0.35, 0.25
LVL_SQZ, LVL_ENTRY = 85, 70
SLOPE_LEN, SLOPE_MIN = 3, 4.0
ATR_FAST, ATR_SLOW, REGIME_MIN = 14, 50, 0.75

STATE_FILE = "state.json"

# ── DATA FETCH ────────────────────────────────────────────────────────
def fetch_candles(symbol: str, interval: str, limit: int = 300):
    # OKX public API — không cần key, không bị chặn
    url = "https://www.okx.com/api/v5/market/candles"
    r = requests.get(url, params={
        "instId": symbol,
        "bar": interval,
        "limit": str(limit)
    }, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    # OKX trả ngược (mới nhất trước) → đảo lại
    data = list(reversed(data))
    return {
        "o": np.array([float(c[1]) for c in data]),
        "h": np.array([float(c[2]) for c in data]),
        "l": np.array([float(c[3]) for c in data]),
        "c": np.array([float(c[4]) for c in data]),
        "v": np.array([float(c[5]) for c in data]),
        "t": np.array([int(c[0]) for c in data]),
    }

# ── INDICATORS ────────────────────────────────────────────────────────
def sma(arr, n):
    out = np.full_like(arr, np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1:i + 1].mean()
    return out

def stdev(arr, n):
    out = np.full_like(arr, np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1:i + 1].std(ddof=0)   # Pine dùng population std
    return out

def wilder_atr(h, l, c, n):
    """RMA(TR, n) — Wilder's ATR giống Pine Script"""
    tr = np.maximum(h - l,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = np.full_like(tr, np.nan)
    atr[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr

def calc_fpi(d):
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    n = len(c)

    # Z-Score
    meanC = sma(c, Z_LEN)
    stdC  = stdev(c, Z_LEN)
    zRaw  = np.where(stdC > 0, (c - meanC) / stdC, 0.0)
    zL    = np.clip( zRaw / 2.5 * 100, 0, 100)
    zS    = np.clip(-zRaw / 2.5 * 100, 0, 100)

    # Wick
    cRange = h - l
    upWick = h - np.maximum(o, c)
    loWick = np.minimum(o, c) - l
    wickL  = np.clip(sma(np.where(cRange > 0, upWick / cRange, 0.0), WICK_LEN) * 250, 0, 100)
    wickS  = np.clip(sma(np.where(cRange > 0, loWick / cRange, 0.0), WICK_LEN) * 250, 0, 100)

    # TDR
    bSize  = np.abs(c - o)
    bodyMA = sma(bSize, TDR_LEN)
    bNorm  = np.where(bodyMA > 0, np.minimum(1.0, bSize / bodyMA), 0.0)
    tdrL   = np.full(n, np.nan)
    tdrS   = np.full(n, np.nan)
    for i in range(TDR_LEN - 1, n):
        bull = np.sum(c[i - TDR_LEN + 1:i + 1] > o[i - TDR_LEN + 1:i + 1])
        bear = TDR_LEN - bull
        tdrL[i] = min(100, (bull / TDR_LEN) * bNorm[i] * 130)
        tdrS[i] = min(100, (bear / TDR_LEN) * bNorm[i] * 130)

    fpiL = WZ * zL + WW * wickL + WT * tdrL
    fpiS = WZ * zS + WW * wickS + WT * tdrS

    # ATR + regime
    atrF  = wilder_atr(h, l, c, ATR_FAST)
    atrSl = wilder_atr(h, l, c, ATR_SLOW)
    atrR  = np.where(atrSl > 0, atrF / atrSl, 1.0)
    ok    = atrR >= REGIME_MIN

    # Slope
    slopeL = np.full(n, 0.0)
    slopeS = np.full(n, 0.0)
    for i in range(SLOPE_LEN, n):
        slopeL[i] = fpiL[i] - fpiL[i - SLOPE_LEN]
        slopeS[i] = fpiS[i] - fpiS[i - SLOPE_LEN]

    # Momentum
    roc  = np.zeros(n)
    for i in range(5, n):
        roc[i] = c[i] - c[i - 5]
    momL = roc > 0
    momS = roc < 0

    # Signals — giống hệt Pine Script
    sqzL   = (fpiL >= LVL_SQZ)   & ok & momL
    sqzS   = (fpiS >= LVL_SQZ)   & ok & momS
    earlyL = (fpiL >= LVL_ENTRY) & (fpiL < LVL_SQZ) & (slopeL >= SLOPE_MIN) & ok
    earlyS = (fpiS >= LVL_ENTRY) & (fpiS < LVL_SQZ) & (slopeS >= SLOPE_MIN) & ok
    exhL   = (fpiL >= LVL_SQZ)   & ~momL
    exhS   = (fpiS >= LVL_SQZ)   & ~momS

    return dict(fpiL=fpiL, fpiS=fpiS,
                sqzL=sqzL, sqzS=sqzS,
                earlyL=earlyL, earlyS=earlyS,
                exhL=exhL, exhS=exhS,
                atrF=atrF, slopeL=slopeL, slopeS=slopeS, zRaw=zRaw)

# ── STATE (dùng cache GitHub Actions) ────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# ── TELEGRAM ──────────────────────────────────────────────────────────
def send(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID,
                              "text": msg, "parse_mode": "HTML"}, timeout=10).raise_for_status()

# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    state = load_state()

    for sym in SYMBOLS:
        try:
            d = fetch_candles(sym, TIMEFRAME, 300)
        except Exception as e:
            print(f"[{sym}] fetch error: {e}")
            continue

        sig = calc_fpi(d)

        # i_cur = candle cuối đã đóng ([-2])
        # i_prv = candle trước đó ([-3])
        # → edge detection không cần state ngoài, nhưng dùng state để tránh
        #   alert 2 lần trong cùng 1 candle (nếu cron chạy trong khi candle chưa đổi)
        i_cur, i_prv = -2, -3
        cur_ts  = str(d["t"][i_cur])
        sym_state = state.get(sym, {})

        if cur_ts == sym_state.get("last_ts"):
            print(f"[{sym}] already alerted this candle, skip")
            continue

        # Detect edge: false → true trên candle đóng
        new_sqzL   = bool(sig["sqzL"][i_cur]   and not sig["sqzL"][i_prv]   and not sig["exhL"][i_cur])
        new_sqzS   = bool(sig["sqzS"][i_cur]   and not sig["sqzS"][i_prv]   and not sig["exhS"][i_cur])
        new_earlyL = bool(sig["earlyL"][i_cur] and not sig["earlyL"][i_prv] and not sig["exhL"][i_cur])
        new_earlyS = bool(sig["earlyS"][i_cur] and not sig["earlyS"][i_prv] and not sig["exhS"][i_cur])
        new_exhL   = bool(sig["exhL"][i_cur]   and not sig["exhL"][i_prv])
        new_exhS   = bool(sig["exhS"][i_cur]   and not sig["exhS"][i_prv])

        price = float(d["c"][i_cur])
        atr   = float(sig["atrF"][i_cur])
        ts    = datetime.fromtimestamp(d["t"][i_cur] / 1000, tz=timezone.utc).strftime("%H:%M UTC")

        msg = None

        if new_sqzL:
            sl = round(price - atr * 1.0, 6)
            tp = round(price + atr * 1.5, 6)
            msg = (f"🔴 <b>FPI SQUEEZE LONG — {sym}</b>\n"
                   f"FPI = {sig['fpiL'][i_cur]:.1f}  |  ATR = {atr:.4f}\n"
                   f"Price <code>{price:.4f}</code>\n"
                   f"SL <code>{sl:.4f}</code>  TP <code>{tp:.4f}</code>  (R:R 1.5)\n"
                   f"⏰ {ts}")

        elif new_sqzS:
            sl = round(price + atr * 1.0, 6)
            tp = round(price - atr * 1.5, 6)
            msg = (f"🔵 <b>FPI SQUEEZE SHORT — {sym}</b>\n"
                   f"FPI = {sig['fpiS'][i_cur]:.1f}  |  ATR = {atr:.4f}\n"
                   f"Price <code>{price:.4f}</code>\n"
                   f"SL <code>{sl:.4f}</code>  TP <code>{tp:.4f}</code>  (R:R 1.5)\n"
                   f"⏰ {ts}")

        elif new_earlyL:
            msg = (f"🟠 <b>FPI Early LONG — {sym}</b>\n"
                   f"FPI = {sig['fpiL'][i_cur]:.1f}  Slope = +{sig['slopeL'][i_cur]:.1f}\n"
                   f"Price <code>{price:.4f}</code> | ⏰ {ts}\n"
                   f"⚠️ Early — chờ FPI qua 85 để confirm")

        elif new_earlyS:
            msg = (f"🟠 <b>FPI Early SHORT — {sym}</b>\n"
                   f"FPI = {sig['fpiS'][i_cur]:.1f}  Slope = +{sig['slopeS'][i_cur]:.1f}\n"
                   f"Price <code>{price:.4f}</code> | ⏰ {ts}\n"
                   f"⚠️ Early — chờ FPI qua 85 để confirm")

        elif new_exhL:
            msg = f"⚠️ <b>EXHAUSTION Long — {sym}</b> | Price <code>{price:.4f}</code> | ⏰ {ts}"

        elif new_exhS:
            msg = f"⚠️ <b>EXHAUSTION Short — {sym}</b> | Price <code>{price:.4f}</code> | ⏰ {ts}"

        if msg:
            try:
                send(msg)
                state[sym] = {"last_ts": cur_ts}
                print(f"[{sym}] ✓ Alert sent")
            except Exception as e:
                print(f"[{sym}] Telegram error: {e}")
        else:
            print(f"[{sym}] No signal. FPIl={sig['fpiL'][-2]:.1f} FPIs={sig['fpiS'][-2]:.1f}")

    save_state(state)

if __name__ == "__main__":
    main()
