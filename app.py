import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Dip Buy Guide",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Mobile-first base */
html, body, [data-testid="stAppViewContainer"] {
    max-width: 100%;
}
[data-testid="stAppViewContainer"] > .main {
    padding: 1rem 1rem 3rem;
}

/* Decline display */
.decline-block {
    text-align: center;
    padding: 1.5rem 1rem;
    border-radius: 14px;
    margin: 1rem 0;
}
.decline-pct {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -2px;
}
.decline-label {
    font-size: 0.95rem;
    opacity: 0.7;
    margin-top: 0.3rem;
}
.today-change {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.6rem;
    opacity: 0.9;
}

/* Judgment box */
.judgment-block {
    text-align: center;
    padding: 1.5rem 1rem;
    border-radius: 14px;
    margin: 1rem 0;
}
.judgment-text {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.4;
}
.judgment-amount {
    font-size: 2.2rem;
    font-weight: 900;
    margin-top: 0.6rem;
    color: #00cc88;
}

/* Data table */
.info-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}
.info-table td {
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.info-table td:last-child {
    text-align: right;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

INDICES = {
    "S&P 500  (^GSPC)": "^GSPC",
    "NASDAQ 100  (^NDX)": "^NDX",
}

# Watch-only tickers: chart + price only, no dip judgment
WATCH = {
    "SpaceX (SPCX)": "SPCX",
}

ALL_OPTIONS = {**INDICES, **WATCH}

SECTORS = {
    "SOXX": "半導体",
    "XLK": "テクノロジー",
    "XLF": "金融",
    "XLE": "エネルギー",
    "XLV": "ヘルスケア",
    "XLY": "一般消費財",
    "XLP": "生活必需品",
}

SECTOR_PERIODS = {"1週間": 7, "1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180}


@st.cache_data(ttl=300)
def fetch_data(ticker: str, months: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    # yfinance may return MultiIndex columns for some versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=60)
def fetch_quote(ticker: str) -> dict:
    """Real-time-ish quote via fast_info."""
    result = {"last_price": None, "prev_close": None}
    try:
        fi = yf.Ticker(ticker).fast_info
        for key in ("last_price",):
            try:
                v = fi[key]
                if v is not None and v == v:
                    result["last_price"] = float(v)
            except Exception:
                pass
        for key in ("previous_close", "regular_market_previous_close"):
            try:
                v = fi[key]
                if v is not None and v == v:
                    result["prev_close"] = float(v)
                    break
            except Exception:
                pass
    except Exception:
        pass
    return result


@st.cache_data(ttl=300)
def fetch_sector_return(ticker: str, period_days: int) -> float | None:
    """Returns percentage change over period_days. None on any failure."""
    try:
        end = datetime.now()
        start = end - timedelta(days=period_days + 5)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].squeeze()
        if not isinstance(close, pd.Series):
            close = pd.Series([close], index=[df.index[-1]])
        close = close.dropna()
        target_start = datetime.now() - timedelta(days=period_days)
        eligible = close[close.index >= pd.Timestamp(target_start.date())]
        if len(eligible) < 1 or len(close) < 2:
            return None
        start_price = float(close.iloc[0])
        end_price = float(eligible.iloc[-1])
        if start_price == 0:
            return None
        return (end_price - start_price) / start_price * 100
    except Exception:
        return None



def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


@st.cache_data(ttl=86400)
def train_ai_model() -> dict | None:
    """
    Train LogisticRegression on 5Y of NDX to predict 20-day forward returns.
    Returns a model bundle dict, or None on failure.
    Time-based 80/20 split; last 20 trading days excluded to prevent data leakage.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score

        end = datetime.now()
        start = end - timedelta(days=5 * 365 + 60)
        df = yf.download("^NDX", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns:
            return None

        close = df["Close"].squeeze().dropna()
        if len(close) < 300:
            return None

        # ── Feature engineering (mirrors app's live calculations) ──
        rolling_high = close.rolling(252, min_periods=50).max()
        decline      = ((rolling_high - close) / rolling_high * 100).clip(lower=0)

        ma25     = close.rolling(25).mean()
        ma75     = close.rolling(75).mean()
        ma25_dev = (close - ma25) / ma25 * 100
        ma75_dev = (close - ma75) / ma75 * 100
        ma_trend = (ma25 > ma75).astype(float)

        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, float("inf"))
        rsi_s    = 100 - (100 / (1 + rs))

        # ── Labels: 20-day forward return ──────────────
        fwd_ret = close.shift(-20) / close - 1
        label   = (fwd_ret > 0).astype(int)

        feat_cols = ["decline", "ma25_dev", "ma75_dev", "ma_trend", "rsi"]
        data = pd.DataFrame({
            "decline": decline, "ma25_dev": ma25_dev, "ma75_dev": ma75_dev,
            "ma_trend": ma_trend, "rsi": rsi_s, "label": label,
        }).dropna()

        # Exclude last 20 trading days — future return is not yet observable
        data = data.iloc[:-20]
        if len(data) < 150:
            return None

        # ── Chronological 80 / 20 split ────────────────
        split    = int(len(data) * 0.8)
        X_train  = data.iloc[:split][feat_cols].values
        y_train  = data.iloc[:split]["label"].values
        X_test   = data.iloc[split:][feat_cols].values
        y_test   = data.iloc[split:]["label"].values

        scaler     = StandardScaler()
        X_train_s  = scaler.fit_transform(X_train)
        X_test_s   = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_s, y_train)

        test_acc = accuracy_score(y_test, model.predict(X_test_s))
        return {
            "model": model, "scaler": scaler,
            "test_acc": test_acc, "feat_cols": feat_cols,
            "n_train": len(X_train), "n_test": len(X_test),
        }
    except Exception:
        return None


def compute_composite(
    decline_pct: float,
    is_below_high: bool,
    combined_ma: pd.DataFrame,
    golden_crosses: pd.DatetimeIndex,
    dead_crosses: pd.DatetimeIndex,
    sector_1m: dict,
) -> dict:
    """Return composite buy-signal score (0–10) and breakdown."""
    OFFENSIVE = {"SOXX", "XLK", "XLY"}
    DEFENSIVE = {"XLP", "XLV"}

    # ── Decline score (0–5) ──────────────────────────────
    if not is_below_high:
        d_score = 0
    elif decline_pct >= 20:
        d_score = 5
    elif decline_pct >= 15:
        d_score = 4
    elif decline_pct >= 10:
        d_score = 3
    elif decline_pct >= 5:
        d_score = 2
    else:
        d_score = 1

    # ── Cross score (0–3) ────────────────────────────────
    lookback = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=30)
    recent_gc = any(d >= lookback for d in golden_crosses)
    recent_dc = any(d >= lookback for d in dead_crosses)

    if recent_gc:
        c_score = 3
        c_label = "GC形成（直近30日）"
    elif recent_dc:
        c_score = 0
        c_label = "DC形成（直近30日）"
    elif len(combined_ma) >= 1 and combined_ma["s"].iloc[-1] > combined_ma["l"].iloc[-1]:
        c_score = 2
        c_label = "MA25 > MA75（上昇トレンド）"
    else:
        c_score = 1
        c_label = "MA25 < MA75（下降トレンド）"

    # ── Sector score (0–2) ───────────────────────────────
    if sector_1m:
        sorted_desc = sorted(sector_1m.items(), key=lambda x: x[1], reverse=True)
        top3 = [lbl.split()[0] for lbl, _ in sorted_desc[:3]]
        def_count = sum(1 for t in top3 if t in DEFENSIVE)
        off_count = sum(1 for t in top3 if t in OFFENSIVE)
        neg_ratio = sum(1 for v in sector_1m.values() if v < 0) / len(sector_1m)
        if def_count >= 2 or neg_ratio >= 0.6:
            s_score = 2
            s_label = "リスクオフ（逆張り好機の傾向）"
        elif off_count >= 2:
            s_score = 0
            s_label = "リスクオン（高値圏の可能性）"
        else:
            s_score = 1
            s_label = "混在"
    else:
        s_score = 1
        s_label = "データなし"

    total = d_score + c_score + s_score

    if total >= 8:
        verdict = "3つのシグナルが揃ってきた局面の傾向です。"
        color, bg = "#00cc88", "rgba(0,200,136,0.09)"
    elif total >= 5:
        verdict = "一部シグナルが出ている傾向です。慎重に判断を。"
        color, bg = "#ffb400", "rgba(255,180,0,0.09)"
    else:
        verdict = "まだシグナルは弱い傾向です。様子見の局面かもしれません。"
        color, bg = "#888888", "rgba(255,255,255,0.04)"

    return {
        "total": total, "color": color, "bg": bg, "verdict": verdict,
        "d_score": d_score, "c_score": c_score, "s_score": s_score,
        "c_label": c_label, "s_label": s_label,
    }


def market_mood(returns: dict, period_label: str) -> dict:
    """Rule-based market mood from sector ETF returns."""
    OFFENSIVE = {"SOXX", "XLK", "XLY"}
    DEFENSIVE = {"XLP", "XLV"}
    overheat_th = {"1週間": 5.0, "1ヶ月": 15.0, "3ヶ月": 25.0, "6ヶ月": 40.0}.get(period_label, 15.0)

    def ticker_of(label: str) -> str:
        return label.split()[0]

    values = list(returns.values())
    total = len(values)
    pos_count = sum(1 for v in values if v > 0)
    neg_count = sum(1 for v in values if v < 0)

    sorted_desc = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    top_ticker = ticker_of(sorted_desc[0][0]) if sorted_desc else ""
    top_name = SECTORS.get(top_ticker, top_ticker)
    top_value = sorted_desc[0][1] if sorted_desc else 0.0
    top3_tickers = [ticker_of(label) for label, _ in sorted_desc[:3]]
    off_in_top3 = sum(1 for t in top3_tickers if t in OFFENSIVE)
    def_in_top3 = sum(1 for t in top3_tickers if t in DEFENSIVE)

    if neg_count == total:
        icon, bg, border = "🔴", "rgba(255,60,60,0.09)", "#ff3c3c"
        text = "全セクターがマイナス圏で、地合いは全面安の傾向です。"
    elif neg_count >= total * 0.6:
        icon, bg, border = "🟠", "rgba(255,140,0,0.09)", "#ff8c00"
        text = "マイナスのセクターが多く、全体的に地合いが軟化している傾向です。"
    elif def_in_top3 >= 2:
        icon, bg, border = "🟡", "rgba(255,204,51,0.09)", "#ffcc33"
        text = "生活必需品・ヘルスケアなど守りのセクターが上位の傾向で、資金が安全圏に向かいやすい地合いです。"
    elif off_in_top3 >= 2 and pos_count >= total * 0.5:
        icon, bg, border = "🟢", "rgba(0,200,136,0.09)", "#00cc88"
        text = "半導体・テクノロジーなど攻めのセクターが上位の傾向で、リスクオン（強気）のムードが感じられます。"
    elif pos_count >= total * 0.6:
        icon, bg, border = "🔵", "rgba(91,156,246,0.09)", "#5b9cf6"
        text = "プラスのセクターが過半数で、全体としてやや強い地合いの傾向です。"
    else:
        icon, bg, border = "⚪", "rgba(255,255,255,0.04)", "rgba(255,255,255,0.2)"
        text = "強弱が混在しており、セクターによって方向感が分かれる地合いの傾向です。"

    if top_value >= overheat_th:
        text += f"　なお{top_name}が{top_value:+.1f}%と突出した動きで、過熱感には注意が必要かもしれません。"

    return {"icon": icon, "text": text, "bg": bg, "border": border}


def cross_period_summary(returns_1m: dict, returns_6m: dict) -> dict | None:
    """Compare 1M vs 6M sector returns to surface trend-change signals."""
    if len(returns_1m) < 2 or len(returns_6m) < 2:
        return None

    OFFENSIVE = {"SOXX", "XLK", "XLY"}
    DEFENSIVE = {"XLP", "XLV"}

    def ticker_of(label: str) -> str:
        return label.split()[0]

    ranked_1m = sorted(returns_1m.items(), key=lambda x: x[1], reverse=True)
    ranked_6m = sorted(returns_6m.items(), key=lambda x: x[1], reverse=True)
    top3_1m = [ticker_of(lbl) for lbl, _ in ranked_1m[:3]]
    top3_6m = [ticker_of(lbl) for lbl, _ in ranked_6m[:3]]
    off_1m = sum(1 for t in top3_1m if t in OFFENSIVE)
    def_1m = sum(1 for t in top3_1m if t in DEFENSIVE)
    off_6m = sum(1 for t in top3_6m if t in OFFENSIVE)
    def_6m = sum(1 for t in top3_6m if t in DEFENSIVE)
    top_1m = ticker_of(ranked_1m[0][0])
    top_6m = ticker_of(ranked_6m[0][0])
    name_1m = SECTORS.get(top_1m, top_1m)
    name_6m = SECTORS.get(top_6m, top_6m)
    tickers_1m_list = [ticker_of(lbl) for lbl, _ in ranked_1m]
    rank_6m_in_1m = (tickers_1m_list.index(top_6m) + 1
                     if top_6m in tickers_1m_list else None)

    if def_1m > def_6m and def_1m >= 1:
        return {"icon": "⚠️",
                "text": ("短期（1ヶ月）で守りのセクターが長期（6ヶ月）よりも上位に浮上している傾向です。"
                         "資金が守りに回り始め、地合い悪化の初期サインかもしれません。"),
                "bg": "rgba(255,204,51,0.07)", "border": "#ffcc33"}

    if off_1m > off_6m and off_1m >= 2:
        return {"icon": "📈",
                "text": ("短期（1ヶ月）で攻めのセクターが長期（6ヶ月）よりも上位に浮上している傾向です。"
                         "リスクオンが加速してきた可能性が感じられます。"),
                "bg": "rgba(0,200,136,0.07)", "border": "#00cc88"}

    if top_1m == top_6m:
        return {"icon": "➡️",
                "text": (f"短期・長期ともに{name_1m}がトップの傾向で、現在の流れが継続している可能性があります。"
                         "セクターローテーションの兆候は現時点では目立ちません。"),
                "bg": "rgba(91,156,246,0.07)", "border": "#5b9cf6"}

    if rank_6m_in_1m is not None and rank_6m_in_1m > 3:
        position = "下位" if rank_6m_in_1m > len(tickers_1m_list) // 2 else "中位"
        return {"icon": "🔄",
                "text": (f"長期（6ヶ月）でトップだった{name_6m}が、短期（1ヶ月）では{position}に後退している傾向です。"
                         "長期の主役が足元で失速し、潮目が変わりかけの可能性も考えられます。"),
                "bg": "rgba(255,140,0,0.07)", "border": "#ff8c00"}

    return {"icon": "🔄",
            "text": (f"長期（6ヶ月）のトップは{name_6m}でしたが、"
                     f"短期（1ヶ月）では{name_1m}が強さを見せている傾向です。"
                     "セクター間のローテーションが起きている可能性があります。"),
            "bg": "rgba(255,140,0,0.07)", "border": "#ff8c00"}


def get_judgment(decline_pct: float, funds: float) -> dict:
    if decline_pct >= 20:
        return {"level": "🔴 全力買い局面", "detail": "残りを投入する局面",
                "amount": funds, "bg": "rgba(255, 60, 60, 0.15)", "border": "#ff3c3c"}
    elif decline_pct >= 10:
        return {"level": "🟡 半分投入の局面", "detail": "待機資金の半分を投入する局面",
                "amount": funds * 0.5, "bg": "rgba(255, 180, 0, 0.12)", "border": "#ffb400"}
    else:
        return {"level": "🟢 待機", "detail": "まだ様子見。高値からの下落が10%未満。",
                "amount": 0, "bg": "rgba(0, 200, 136, 0.1)", "border": "#00cc88"}


# ── Header ──────────────────────────────────────────
st.title("📉 Dip Buy Guide")

tab1, tab2 = st.tabs(["📉 ディップ判定", "🔥 セクター比較"])

# ════════════════════════════════════════════════════
# TAB 1 — Dip judgment + composite score + news
# ════════════════════════════════════════════════════
with tab1:
    # ── Controls ────────────────────────────────────────
    selected_label = st.radio("銘柄", list(ALL_OPTIONS.keys()), horizontal=True)
    ticker = ALL_OPTIONS[selected_label]
    watch_mode = selected_label in WATCH

    period_months = st.slider(
        "表示期間" if watch_mode else "高値を測る期間",
        min_value=3, max_value=36, value=12, step=3, format="%dヶ月",
    )

    # ── Fetch ───────────────────────────────────────────
    with st.spinner("データ取得中..."):
        df = fetch_data(ticker, period_months)
        # 1M sector data for composite score (cached; cheap on repeat renders)
        sector_1m: dict[str, float] = {}
        if not watch_mode:
            for _etf, _name in SECTORS.items():
                _r = fetch_sector_return(_etf, 30)
                if _r is not None:
                    sector_1m[f"{_etf}  {_name}"] = _r

    # ── Build close series ──────────────────────────────
    close = pd.Series(dtype="float64")
    if not df.empty and "Close" in df.columns:
        close = df["Close"].squeeze()
        if not isinstance(close, pd.Series):
            close = pd.Series([close], index=[df.index[-1]])
        close = close.dropna()

    if close.empty:
        if watch_mode:
            st.markdown(
                f"""
                <div class="decline-block" style="background:rgba(255,255,255,0.04); border: 1.5px solid rgba(255,255,255,0.15);">
                    <div class="decline-label" style="opacity:0.6;">👀 ウォッチ中</div>
                    <div class="decline-pct" style="color:#888; font-size:2.2rem;">— —</div>
                    <div class="decline-label">データ取得不可（{ticker}）</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("この銘柄は現在 yfinance から価格データを取得できません。時間をおいて再度お試しください。")
        else:
            st.error("データの取得に失敗しました。しばらく待ってから再試行してください。")
        st.stop()

    # ── Price calculations ──────────────────────────────
    quote = fetch_quote(ticker)
    hist_last = float(close.iloc[-1])
    current_price = quote["last_price"] if quote["last_price"] is not None else hist_last

    if quote["prev_close"] is not None:
        prev_close = quote["prev_close"]
    elif len(close) >= 2:
        prev_close = float(close.iloc[-2])
    else:
        prev_close = current_price

    today_change_pct = (current_price - prev_close) / prev_close * 100 if prev_close else 0.0

    last_date = close.index[-1]
    is_today = hasattr(last_date, "date") and last_date.date() == datetime.now().date()
    if is_today:
        close.iloc[-1] = current_price
    else:
        close.loc[pd.Timestamp(datetime.now().date())] = current_price
    current_date = close.index[-1]

    today_color = "#00cc88" if today_change_pct >= 0 else "#ff3c3c"
    today_sign = "+" if today_change_pct >= 0 else ""
    today_html = (
        f'<div class="today-change" style="color:{today_color};">'
        f'今日: {today_sign}{today_change_pct:.1f}%</div>'
    )

    # ── Dip-specific calculations (needed for composite) ──
    if not watch_mode:
        recent_high = float(close.max())
        recent_high_date = close.idxmax()
        decline_pct = abs((current_price - recent_high) / recent_high * 100)
        is_below_high = current_price < recent_high

    # ── Technical indicators ─────────────────────────────
    ma_short = close.rolling(25).mean()
    ma_long = close.rolling(75).mean()
    rsi = compute_rsi(close)

    rsi_clean = rsi.dropna()
    combined_ma = pd.DataFrame({"s": ma_short, "l": ma_long}).dropna()
    if len(combined_ma) >= 2:
        diff = combined_ma["s"] - combined_ma["l"]
        sign_prev = diff.shift(1)
        golden_crosses = diff.index[(sign_prev < 0) & (diff >= 0)]
        dead_crosses = diff.index[(sign_prev > 0) & (diff <= 0)]
    else:
        golden_crosses = pd.DatetimeIndex([])
        dead_crosses = pd.DatetimeIndex([])

    # ── Composite score ──────────────────────────────────
    if not watch_mode:
        sc = compute_composite(
            decline_pct if is_below_high else 0,
            is_below_high,
            combined_ma,
            golden_crosses,
            dead_crosses,
            sector_1m,
        )
        filled = "█" * sc["total"]
        empty = "░" * (10 - sc["total"])
        d_str = f"{decline_pct:.1f}%" if is_below_high else "高値圏"
        st.markdown(
            f"""
            <div style="padding:1.1rem 1.3rem; border-radius:14px;
                        background:{sc['bg']}; border:1.5px solid {sc['color']}55;
                        margin-bottom:0.5rem;">
                <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.3rem;">
                    <span style="font-size:0.82rem; opacity:0.6;">🎯 総合スコア</span>
                    <span style="font-size:2.6rem; font-weight:900; color:{sc['color']}; line-height:1;">{sc['total']}</span>
                    <span style="font-size:0.95rem; opacity:0.45;">/ 10</span>
                </div>
                <div style="font-family:monospace; font-size:1.05rem; color:{sc['color']}; margin-bottom:0.5rem; letter-spacing:1px;">{filled}{empty}</div>
                <div style="font-size:0.88rem; opacity:0.85; margin-bottom:0.6rem;">{sc['verdict']}</div>
                <div style="font-size:0.8rem; opacity:0.55; line-height:1.7;">
                    ▼ 下落率 {d_str} &nbsp;→&nbsp; {sc['d_score']}/5<br>
                    ↗ クロス {sc['c_label']} &nbsp;→&nbsp; {sc['c_score']}/3<br>
                    🌐 地合い {sc['s_label']} &nbsp;→&nbsp; {sc['s_score']}/2
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── AI judgment ──────────────────────────────────────
        ai_bundle = train_ai_model()
        if ai_bundle is not None:
            ma25_last = (float(ma_short.dropna().iloc[-1])
                         if ma_short.dropna().shape[0] > 0 else None)
            ma75_last = (float(ma_long.dropna().iloc[-1])
                         if ma_long.dropna().shape[0] > 0 else None)
            rsi_last  = (float(rsi_clean.iloc[-1])
                         if len(rsi_clean) > 0 else None)

            if all(v is not None for v in [ma25_last, ma75_last, rsi_last]):
                x_now = pd.DataFrame([[
                    decline_pct if is_below_high else 0.0,
                    (current_price - ma25_last) / ma25_last * 100,
                    (current_price - ma75_last) / ma75_last * 100,
                    1.0 if ma25_last > ma75_last else 0.0,
                    rsi_last,
                ]], columns=ai_bundle["feat_cols"])

                x_scaled   = ai_bundle["scaler"].transform(x_now.values)
                pred       = int(ai_bundle["model"].predict(x_scaled)[0])
                proba      = ai_bundle["model"].predict_proba(x_scaled)[0]
                confidence = float(proba[pred])

                ai_label = "強気" if pred == 1 else "弱気"
                ai_icon  = "📈"   if pred == 1 else "📉"
                ai_color = "#00cc88" if pred == 1 else "#ff3c3c"
                ai_bg    = "rgba(0,200,136,0.07)" if pred == 1 else "rgba(255,60,60,0.07)"

                # 4-pattern comment: AI sentiment × MA cross state
                _cross_known = (ma25_last is not None and ma75_last is not None)
                if _cross_known:
                    _golden = ma25_last > ma75_last
                    _combo_map = {
                        (1, True):  "🚀 順張り良好。トレンドに乗れている",
                        (1, False): "🤔 強気だが足元は調整中。様子見も一手",
                        (0, True):  "⚡ 反転の初動かも。底打ちの可能性。逆張り妙味",
                        (0, False): "⚠️ 下落トレンド継続。落ちるナイフに注意。慎重に",
                    }
                    combo_comment = _combo_map[(pred, _golden)]
                    combo_html = (
                        f'<div style="font-size:0.85rem; margin-top:0.3rem;">'
                        f"{combo_comment}</div>"
                    )
                else:
                    combo_html = ""

                st.markdown(
                    f"""
                    <div style="padding:0.8rem 1.1rem; border-radius:12px;
                                background:{ai_bg}; border:1.5px solid {ai_color}44;
                                margin-bottom:0.5rem;">
                        <div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;">
                            <span style="font-size:0.82rem; opacity:0.6;">🤖 AI地合い判定</span>
                            <span style="font-size:1.1rem; font-weight:700; color:{ai_color};">{ai_icon} {ai_label}</span>
                            <span style="font-size:0.9rem; opacity:0.75;">確信度 {confidence*100:.0f}%</span>
                        </div>
                        {combo_html}
                        <div style="font-size:0.75rem; opacity:0.42; margin-top:0.35rem; line-height:1.5;">
                            テスト正答率 {ai_bundle['test_acc']*100:.0f}%（{ai_bundle['n_test']}日分で検証）　※参考情報です。投資判断の根拠にしないでください。
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Decline / watch display ──────────────────────────
    if watch_mode:
        st.markdown(
            f"""
            <div class="decline-block" style="background:rgba(91,156,246,0.08); border: 1.5px solid rgba(91,156,246,0.25);">
                <div class="decline-label" style="opacity:0.6;">👀 ウォッチ中</div>
                <div class="decline-pct" style="color:#5b9cf6; font-size:3rem;">{current_price:,.1f}</div>
                <div class="decline-label">現在値</div>
                {today_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if not is_below_high:
            decline_color = "#00cc88"
            decline_bg = "rgba(0, 200, 136, 0.08)"
            decline_str = f"+{decline_pct:.1f}%"
            decline_caption = "▲ 高値圏（下落なし）"
        else:
            if decline_pct >= 20:
                decline_color, decline_bg = "#ff3b30", "rgba(255, 59, 48, 0.12)"
                zone_label = "全力買うゾーン"
            elif decline_pct >= 10:
                decline_color, decline_bg = "#ff8c1a", "rgba(255, 140, 26, 0.10)"
                zone_label = "半分買うゾーン"
            else:
                decline_color, decline_bg = "#ffcc33", "rgba(255, 204, 51, 0.08)"
                zone_label = "まだ静観"
            decline_str = f"{decline_pct:.1f}%"
            decline_caption = f"▼ 高値から下落中 · {zone_label}"

        st.markdown(
            f"""
            <div class="decline-block" style="background:{decline_bg}; border: 1.5px solid {decline_color}55;">
                <div class="decline-pct" style="color:{decline_color};">{decline_str}</div>
                <div class="decline-label" style="color:{decline_color}; opacity:0.95; font-weight:600;">{decline_caption}</div>
                {today_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Chart ────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, mode="lines", name="終値",
        line=dict(color="#5b9cf6", width=2),
        hovertemplate="%{x|%Y/%m/%d}<br>%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    if not watch_mode:
        fig.add_trace(go.Scatter(
            x=[recent_high_date], y=[recent_high],
            mode="markers+text",
            marker=dict(color="#ffb400", size=10, symbol="triangle-up"),
            text=[f"  高値 {recent_high:,.0f}"],
            textposition="top right", textfont=dict(color="#ffb400", size=11),
            name="直近高値",
            hovertemplate="%{x|%Y/%m/%d}<br>%{y:,.0f}<extra>直近高値</extra>",
        ), row=1, col=1)

    if ma_short.dropna().shape[0] > 0:
        ma25 = ma_short.dropna()
        fig.add_trace(go.Scatter(
            x=ma25.index, y=ma25.values, mode="lines", name="MA25",
            line=dict(color="#ff9f43", width=1.5),
            hovertemplate="%{x|%Y/%m/%d}<br>MA25: %{y:,.0f}<extra></extra>",
        ), row=1, col=1)

    if ma_long.dropna().shape[0] > 0:
        ma75 = ma_long.dropna()
        fig.add_trace(go.Scatter(
            x=ma75.index, y=ma75.values, mode="lines", name="MA75",
            line=dict(color="#a29bfe", width=1.5),
            hovertemplate="%{x|%Y/%m/%d}<br>MA75: %{y:,.0f}<extra></extra>",
        ), row=1, col=1)

    if len(golden_crosses) > 0:
        gc_y = ma_short.loc[golden_crosses]
        fig.add_trace(go.Scatter(
            x=golden_crosses, y=gc_y.values, mode="markers", name="GC",
            marker=dict(color="#00cc88", size=11, symbol="triangle-up"),
            hovertemplate="%{x|%Y/%m/%d}<br>ゴールデンクロス<extra></extra>",
        ), row=1, col=1)

    if len(dead_crosses) > 0:
        dc_y = ma_short.loc[dead_crosses]
        fig.add_trace(go.Scatter(
            x=dead_crosses, y=dc_y.values, mode="markers", name="DC",
            marker=dict(color="#ff3c3c", size=11, symbol="triangle-down"),
            hovertemplate="%{x|%Y/%m/%d}<br>デッドクロス<extra></extra>",
        ), row=1, col=1)

    fig.add_hline(
        y=current_price, line_dash="dot", line_color="rgba(255,255,255,0.3)",
        annotation_text=f"現在 {current_price:,.0f}",
        annotation_position="bottom right",
        annotation_font_color="rgba(255,255,255,0.6)",
        row=1, col=1,
    )

    rsi_clean = rsi.dropna()
    if len(rsi_clean) > 0:
        fig.add_trace(go.Scatter(
            x=rsi_clean.index, y=rsi_clean.values, mode="lines", name="RSI(14)",
            line=dict(color="#74b9ff", width=1.5),
            hovertemplate="%{x|%Y/%m/%d}<br>RSI: %{y:.1f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,60,60,0.08)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,136,0.08)", line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,60,60,0.55)", row=2, col=1,
                      annotation_text="70", annotation_position="right",
                      annotation_font_color="rgba(255,60,60,0.8)", annotation_font_size=10)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,136,0.55)", row=2, col=1,
                      annotation_text="30", annotation_position="right",
                      annotation_font_color="rgba(0,200,136,0.8)", annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,31,46,0.8)",
        margin=dict(l=0, r=0, t=10, b=10), height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis2=dict(showgrid=False, zeroline=False),
        yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                    range=[0, 100], tickvals=[0, 30, 50, 70, 100]),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="RSI", title_font_size=10, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    cur_date_str = (current_date.strftime("%Y/%m/%d")
                    if hasattr(current_date, "strftime") else str(current_date))

    # ── Post-chart content ───────────────────────────────
    if watch_mode:
        st.info("👀 ウォッチ専用銘柄です。ディップ判定（買い増し目安）は表示しません。")
        with st.expander("詳細", expanded=True):
            rows = [
                ("現在値", f"{current_price:,.2f}　({cur_date_str})"),
                ("前日比", f"{today_sign}{today_change_pct:.2f}%"),
                ("表示期間", f"過去{period_months}ヶ月"),
            ]
            rows_html = "\n".join(
                f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
            )
            st.markdown(
                f'<table class="info-table"><tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("#### 待機資金")
        if "waiting_funds" not in st.session_state:
            st.session_state.waiting_funds = 1_000_000
        waiting_funds = st.number_input(
            "合計額（円）", min_value=0, value=st.session_state.waiting_funds,
            step=100_000, format="%d", key="waiting_funds",
        )

        judgment = get_judgment(decline_pct if is_below_high else 0, waiting_funds)
        amount_html = ""
        if judgment["amount"] > 0:
            amount_html = f'<div class="judgment-amount">¥{judgment["amount"]:,.0f}</div>'
        st.markdown(
            f"""
            <div class="judgment-block" style="background:{judgment['bg']}; border: 1.5px solid {judgment['border']}44;">
                <div class="judgment-text">{judgment['level']}</div>
                <div style="opacity:0.75; font-size:0.9rem; margin-top:0.4rem;">{judgment['detail']}</div>
                {amount_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("判定の根拠", expanded=True):
            high_date_str = (recent_high_date.strftime("%Y/%m/%d")
                             if hasattr(recent_high_date, "strftime") else str(recent_high_date))
            rows = [
                ("直近高値", f"{recent_high:,.0f}　({high_date_str})"),
                ("現在値", f"{current_price:,.0f}　({cur_date_str})"),
                ("下落率", f"{decline_pct:.2f}%" if is_below_high else "高値圏（下落なし）"),
                ("前日比", f"{today_sign}{today_change_pct:.2f}%"),
                ("測定期間", f"過去{period_months}ヶ月"),
                ("待機資金", f"¥{waiting_funds:,.0f}"),
            ]
            rows_html = "\n".join(
                f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
            )
            st.markdown(
                f'<table class="info-table"><tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True,
            )



# ════════════════════════════════════════════════════
# TAB 2 — Sector comparison
# ════════════════════════════════════════════════════
with tab2:
    period_label = st.radio(
        "騰落率の期間", list(SECTOR_PERIODS.keys()), index=1,
        horizontal=True, key="sector_period",
    )
    period_days = SECTOR_PERIODS[period_label]

    with st.spinner("セクターデータ取得中..."):
        returns: dict[str, float] = {}
        skipped: list[str] = []
        for etf, name in SECTORS.items():
            ret = fetch_sector_return(etf, period_days)
            if ret is None:
                skipped.append(etf)
            else:
                returns[f"{etf}  {name}"] = ret

        returns_1m: dict[str, float] = {}
        returns_6m: dict[str, float] = {}
        for etf, name in SECTORS.items():
            key = f"{etf}  {name}"
            r1 = returns[key] if period_days == 30 and key in returns else fetch_sector_return(etf, 30)
            if r1 is not None:
                returns_1m[key] = r1
            r6 = returns[key] if period_days == 180 and key in returns else fetch_sector_return(etf, 180)
            if r6 is not None:
                returns_6m[key] = r6

    if skipped:
        st.caption(f"データ取得不可のためスキップ: {', '.join(skipped)}")

    if not returns:
        st.warning("セクターデータを取得できませんでした。しばらく待ってから再試行してください。")
    else:
        sorted_items = sorted(returns.items(), key=lambda x: x[1])
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = ["#00cc88" if v >= 0 else "#ff3c3c" for v in values]

        mood = market_mood(returns, period_label)
        st.markdown(
            f"""
            <div style="padding:0.85rem 1.1rem; border-radius:12px;
                        background:{mood['bg']}; border:1.5px solid {mood['border']}55;
                        margin-bottom:0.5rem; font-size:0.92rem; line-height:1.65;">
                <span style="font-size:1.05rem;">{mood['icon']}</span>&nbsp;{mood['text']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_sector = go.Figure(go.Bar(
            x=values, y=labels, orientation="h", marker_color=colors,
            text=[f"{v:+.1f}%" for v in values],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:+.2f}%<extra></extra>",
        ))

        x_abs_max = max(abs(v) for v in values) if values else 1
        x_padding = x_abs_max * 0.25

        fig_sector.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,31,46,0.8)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(280, len(labels) * 44),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       zeroline=True, zerolinecolor="rgba(255,255,255,0.25)",
                       ticksuffix="%",
                       range=[-(x_abs_max + x_padding), x_abs_max + x_padding]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_sector, use_container_width=True)
        st.caption(f"期間: 直近{period_label} の騰落率（強い順）")

        summary = cross_period_summary(returns_1m, returns_6m)
        if summary:
            st.markdown("**📊 期間横断の総評**")
            st.markdown(
                f"""
                <div style="padding:0.85rem 1.1rem; border-radius:12px;
                            background:{summary['bg']}; border:1.5px solid {summary['border']}55;
                            margin-top:0.25rem; font-size:0.92rem; line-height:1.65;">
                    <span style="font-size:1.05rem;">{summary['icon']}</span>&nbsp;{summary['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )
