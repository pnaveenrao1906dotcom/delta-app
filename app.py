import time, requests, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import ndtr

np.seterr(all="ignore")
st.set_page_config(layout="wide", page_title="Delta Options Terminal")

RES_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}
PERP = {"BTC": "BTCUSD", "ETH": "ETHUSD", "XAUT": "XAUTUSD", "XAU": "XAUTUSD"}
TZ = "Asia/Kolkata"


# ---------------- API ----------------
@st.cache_data(ttl=300, show_spinner=False)
def get_products(base):
    rows, after = [], None
    while True:
        pr = dict(contract_types="call_options,put_options", states="live", page_size=500)
        if after:
            pr["after"] = after
        try:
            r = requests.get(f"{base}/v2/products", params=pr, timeout=20).json()
        except Exception:
            break
        for p in r.get("result", []):
            try:
                rows.append(dict(symbol=p["symbol"], type=p["contract_type"],
                                 strike=float(p["strike_price"]), expiry=p["settlement_time"],
                                 under=p["underlying_asset"]["symbol"]))
            except Exception:
                continue
        after = (r.get("meta") or {}).get("after")
        if not after:
            break
    return pd.DataFrame(rows)


@st.cache_data(ttl=30, show_spinner=False)
def candles(base, symbol, res, start, end):
    rows, s, step = [], start, RES_SEC[res] * 1500
    while s < end:
        e = min(s + step, end)
        try:
            r = requests.get(f"{base}/v2/history/candles",
                             params=dict(symbol=symbol, resolution=res, start=s, end=e), timeout=20).json()
            rows += r.get("result", []) or []
        except Exception:
            pass
        s = e
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df.reindex(columns=["open", "high", "low", "close", "volume"]).astype(float)


def leg(base, sym, res, start, end, source):
    tr = candles(base, sym, res, start, end)
    if source == "Mark":
        mk = candles(base, "MARK:" + sym, res, start, end)
        if not mk.empty:
            mk["volume"] = tr["volume"].reindex(mk.index).fillna(0) if not tr.empty else 0.0
            tr = mk
    oi = candles(base, "OI:" + sym, res, start, end)
    return tr, (oi["close"] if not oi.empty else pd.Series(dtype=float))


# ---------------- Black-Scholes (IV + Greeks) ----------------
def bs(S, K, T, sig, call):
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / sq
    d2 = d1 - sq
    if call:
        return S * ndtr(d1) - K * ndtr(d2)
    return K * ndtr(-d2) - S * ndtr(-d1)


def implied(price, S, K, T, call):
    lo, hi = np.full(len(price), 0.01), np.full(len(price), 6.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        v = bs(S, K, T, mid, call)
        hi = np.where(v > price, mid, hi)
        lo = np.where(v > price, lo, mid)
    iv = (lo + hi) / 2
    return np.where((T > 1e-7) & (price > 0), iv, np.nan)


def greeks(S, K, T, sig, call):
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / sq
    pdf = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)
    delta = ndtr(d1) if call else ndtr(d1) - 1
    return dict(delta=delta, gamma=pdf / (S * sq),
                vega=S * pdf * np.sqrt(T) / 100, theta=-S * pdf * sig / (2 * np.sqrt(T)) / 365)


def leg_greeks(price, S, K, expiry, call):
    T = np.maximum((expiry - price.index).total_seconds().values / (365 * 86400), 1e-8)
    p = price.values.astype(float)
    s = S.reindex(price.index).ffill().values.astype(float)
    iv = implied(p, s, K, T, call)
    g = greeks(s, K, T, iv, call)
    g["iv"] = iv * 100
    return pd.DataFrame(g, index=price.index)


# ---------------- Indicators ----------------
def rsi(s, n):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def adx(h, l, c, di_len, adx_len):
    up, dn = h.diff(), -l.diff()
    pl = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mi = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / di_len, adjust=False).mean()
    pdi = 100 * pl.ewm(alpha=1 / di_len, adjust=False).mean() / atr
    mdi = 100 * mi.ewm(alpha=1 / di_len, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return pdi, mdi, dx.ewm(alpha=1 / adx_len, adjust=False).mean()


def obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()


def ma(s, typ, n):
    return s.rolling(n).mean() if typ == "SMA" else s.ewm(span=n, adjust=False).mean()


# ---------------- Chart builder ----------------
def make_fig(df, panels, title, P):
    n = len(panels)
    rows = 1 + n
    heights = ([0.5] + [0.5 / n] * n) if n else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.015, row_heights=heights)
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
                                 name=title, increasing_line_color="#26a69a",
                                 decreasing_line_color="#ef5350"), 1, 1)
    for typ, ln, col in P["mas"]:
        fig.add_trace(go.Scatter(x=df.index, y=ma(df.close, typ, int(ln)), name=f"{typ}{ln}",
                                 line=dict(width=1.3, color=col)), 1, 1)
    if P["vwap"] and df.volume.sum() > 0:
        tp = (df.high + df.low + df.close) / 3
        day = df.index.date
        pv = (tp * df.volume).groupby(day).cumsum()
        vv = df.volume.groupby(day).cumsum()
        fig.add_trace(go.Scatter(x=df.index, y=pv / vv, name="VWAP",
                                 line=dict(color="#ffeb3b", width=1.5)), 1, 1)

    def line(r, y, name, color="#2962ff"):
        fig.add_trace(go.Scatter(x=df.index, y=y, name=name, line=dict(color=color, width=1.4)), r, 1)

    for i, pn in enumerate(panels, start=2):
        if pn == "OBV":
            o = obv(df.close, df.volume)
            line(i, o, "OBV")
            line(i, ma(o, "SMA", int(P["obv_ma"])), "OBV MA", "#ff9800")
        elif pn == "RSI":
            line(i, rsi(df.close, int(P["rsi_len"])), "RSI", "#b388ff")
            for lv in (30, 70):
                fig.add_hline(y=lv, line_dash="dot", line_color="gray", row=i, col=1)
        elif pn == "ADX / DI":
            pdi, mdi, ax = adx(df.high, df.low, df.close, int(P["di_len"]), int(P["adx_len"]))
            line(i, ax, "ADX", "#ffffff")
            line(i, pdi, "+DI", "#26a69a")
            line(i, mdi, "-DI", "#ef5350")
        elif pn == "Std Dev":
            line(i, df.close.rolling(int(P["sd_len"])).std(), f"StdDev{P['sd_len']}", "#00bcd4")
        elif pn == "CRS" and "crs" in df.columns:
            line(i, df["crs"], "CRS ratio", "#4caf50")
            line(i, ma(df["crs"], "SMA", int(P["crs_ma"])), "CRS MA", "#ff5252")
        elif pn == "OI" and "oi" in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df["oi"], name="OI", marker_color="#5c6bc0"), i, 1)
        elif pn == "Volume":
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="#607d8b"), i, 1)
        elif pn in df.columns:  # delta, gamma, theta, vega, iv
            line(i, df[pn], pn.upper(), "#ffa726")
        fig.update_yaxes(title_text=pn, row=i, col=1)

    fig.update_layout(template="plotly_dark", height=520 + 190 * n, hovermode="x unified",
                      xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikethickness=1, side="right")
    return fig


# ---------------- Sidebar ----------------
sb = st.sidebar
sb.title("⚙️ Settings")
server = sb.radio("Delta server", ["India", "Global"])
base = "https://api.india.delta.exchange" if server == "India" else "https://api.delta.exchange"

prods = get_products(base)
if prods.empty:
    st.error("Products nahi aaye - server (India/Global) badal kar dekhein.")
    st.stop()

coin = sb.selectbox("Coin", sorted(prods.under.unique()))
sub = prods[prods.under == coin]
expiry = sb.selectbox("Expiry", sorted(sub.expiry.unique()))
sub = sub[sub.expiry == expiry]
calls = sub[sub.type == "call_options"].sort_values("strike")
puts = sub[sub.type == "put_options"].sort_values("strike")
if calls.empty or puts.empty:
    st.error("Is expiry me call ya put available nahi hai, dusri expiry chuniye.")
    st.stop()

ck = sb.selectbox("Call strike", calls.strike.tolist(), index=len(calls) // 2)
pk = sb.selectbox("Put strike (same = straddle)", puts.strike.tolist(), index=len(puts) // 2)
res = sb.selectbox("Timeframe", list(RES_SEC), index=1)
days = sb.slider("Din (history)", 1, 30, 3)
source = sb.radio("Option price source", ["Mark", "Traded"])

sb.subheader("Moving averages")
mas = []
for k, (t, n, c) in enumerate([("EMA", 20, "#ffb300"), ("SMA", 50, "#29b6f6")], 1):
    a, b = sb.columns(2)
    typ = a.selectbox(f"MA{k}", ["SMA", "EMA"], index=["SMA", "EMA"].index(t), key=f"t{k}")
    ln = b.number_input("len", 1, 500, n, key=f"n{k}")
    mas.append((typ, ln, c))

sb.subheader("Indicator inputs")
P = dict(mas=mas, vwap=sb.checkbox("VWAP", True),
         rsi_len=sb.number_input("RSI length", 2, 100, 14),
         obv_ma=sb.number_input("OBV MA length", 2, 200, 20),
         di_len=sb.number_input("ADX DI length", 2, 100, 14),
         adx_len=sb.number_input("ADX smoothing", 2, 100, 14),
         sd_len=sb.number_input("StdDev length", 2, 200, 20),
         crs_ma=sb.number_input("CRS MA length", 2, 200, 20))
comp = sb.selectbox("CRS compare with",
                    ["Dusra chart (underlying <-> strangle)", "BTCUSD", "ETHUSD", "XAUTUSD", "Custom"])
custom = sb.text_input("Custom symbol (e.g. C-BTC-90000-261030)") if comp == "Custom" else ""
auto = sb.checkbox("Auto refresh (30s)", False)

# ---------------- Data ----------------
end = int(time.time()) // 60 * 60
start = end - days * 86400
csym = calls[calls.strike == ck].symbol.iloc[0]
psym = puts[puts.strike == pk].symbol.iloc[0]
usym = PERP.get(coin, coin + "USD")
exp_ts = pd.to_datetime(expiry, utc=True)

with st.spinner("Delta se data la raha hu..."):
    c_df, c_oi = leg(base, csym, res, start, end, source)
    p_df, p_oi = leg(base, psym, res, start, end, source)
    u = candles(base, usym, res, start, end)

if c_df.empty or p_df.empty or u.empty:
    st.error(f"Data nahi mila. Call: {csym} | Put: {psym} | Underlying: {usym}. "
             "Symbol ya server check karein.")
    st.stop()

idx = c_df.index.union(p_df.index)
C = c_df.reindex(idx).ffill()
Pu = p_df.reindex(idx).ffill()
S = pd.DataFrame({"open": C.open + Pu.open, "high": C.high + Pu.high, "low": C.low + Pu.low,
                  "close": C.close + Pu.close,
                  "volume": c_df.volume.reindex(idx).fillna(0) + p_df.volume.reindex(idx).fillna(0)})
S = S.dropna()
if S.empty:
    st.error("Strangle ka data khali hai, timeframe/din badal kar dekhein.")
    st.stop()

gc = leg_greeks(C.close.loc[S.index], u.close, ck, exp_ts, True)
gp = leg_greeks(Pu.close.loc[S.index], u.close, pk, exp_ts, False)
for g in ("delta", "gamma", "theta", "vega"):
    S[g] = gc[g] + gp[g]  # net (long strangle)
S["iv"] = (gc["iv"] + gp["iv"]) / 2
if not c_oi.empty and not p_oi.empty:
    S["oi"] = c_oi.reindex(S.index).ffill() + p_oi.reindex(S.index).ffill()

U = u.copy()
U["iv"] = S["iv"].reindex(U.index).ffill()
oiu = candles(base, "OI:" + usym, res, start, end)
if not oiu.empty:
    U["oi"] = oiu["close"].reindex(U.index).ffill()

if comp.startswith("Dusra"):
    rs_, ru_ = U["close"], S["close"]  # strangle vs underlying | underlying vs strangle
else:
    d = candles(base, custom if comp == "Custom" else comp, res, start, end)
    rs_ = ru_ = d["close"] if not d.empty else None
if rs_ is not None:
    S["crs"] = S["close"] / rs_.reindex(S.index).ffill()
    U["crs"] = U["close"] / ru_.reindex(U.index).ffill()

# ---------------- UI ----------------
st.title(f"{coin} | {csym} + {psym}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Strangle premium", f"{S.close.iloc[-1]:.1f}")
m2.metric("Underlying", f"{U.close.iloc[-1]:.2f}")
m3.metric("Avg IV", f"{S.iv.iloc[-1]:.1f}%")
m4.metric("Net delta", f"{S.delta.iloc[-1]:.3f}")

OPT = ["Volume", "OBV", "OI", "delta", "theta", "gamma", "vega", "iv", "RSI", "ADX / DI", "Std Dev", "CRS"]
OPT_U = ["Volume", "OBV", "OI", "iv", "RSI", "ADX / DI", "Std Dev", "CRS"]
t1, t2 = st.tabs(["📈 Strangle / Straddle", f"🪙 {coin} Underlying"])
with t1:
    pn1 = st.multiselect("Panels", OPT, default=["OBV", "OI", "iv", "RSI"], key="p1")
    st.plotly_chart(make_fig(S, pn1, "Strangle", P), use_container_width=True)
with t2:
    pn2 = st.multiselect("Panels", OPT_U, default=["OBV", "OI", "RSI"], key="p2")
    st.plotly_chart(make_fig(U, pn2, coin, P), use_container_width=True)

if auto:
    time.sleep(30)
    st.rerun()
