import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
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
    """Real-time-ish quote via fast_info. Daily download lags by a session or
    more, so this gives the live last price and the true previous close."""
    result = {"last_price": None, "prev_close": None}
    try:
        fi = yf.Ticker(ticker).fast_info
        for key in ("last_price",):
            try:
                v = fi[key]
                if v is not None and v == v:  # not NaN
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


def get_judgment(decline_pct: float, funds: float) -> dict:
    if decline_pct >= 20:
        return {
            "level": "🔴 全力買い局面",
            "detail": "残りを投入する局面",
            "amount": funds,
            "bg": "rgba(255, 60, 60, 0.15)",
            "border": "#ff3c3c",
        }
    elif decline_pct >= 10:
        return {
            "level": "🟡 半分投入の局面",
            "detail": "待機資金の半分を投入する局面",
            "amount": funds * 0.5,
            "bg": "rgba(255, 180, 0, 0.12)",
            "border": "#ffb400",
        }
    else:
        return {
            "level": "🟢 待機",
            "detail": "まだ様子見。高値からの下落が10%未満。",
            "amount": 0,
            "bg": "rgba(0, 200, 136, 0.1)",
            "border": "#00cc88",
        }


# ── Header ──────────────────────────────────────────
st.title("📉 Dip Buy Guide")

# ── Index selector ──────────────────────────────────
selected_label = st.radio("銘柄", list(ALL_OPTIONS.keys()), horizontal=True)
ticker = ALL_OPTIONS[selected_label]
watch_mode = selected_label in WATCH

# ── Period slider ────────────────────────────────────
period_months = st.slider(
    "表示期間" if watch_mode else "高値を測る期間",
    min_value=3,
    max_value=36,
    value=12,
    step=3,
    format="%dヶ月",
)

# ── Fetch ────────────────────────────────────────────
with st.spinner("データ取得中..."):
    df = fetch_data(ticker, period_months)

# Build a clean Close series. yfinance can return a trailing NaN row for the
# current (still-forming) session; dropna() prevents NaN current_price/decline.
close = pd.Series(dtype="float64")
if not df.empty and "Close" in df.columns:
    close = df["Close"].squeeze()
    if not isinstance(close, pd.Series):  # single-row frame squeezes to scalar
        close = pd.Series([close], index=[df.index[-1]])
    close = close.dropna()

# ── No usable data ───────────────────────────────────
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

# ── Common calculations ──────────────────────────────
# Daily download lags (its last bar can be a session or more old), so prefer
# the live fast_info quote for current price and the true previous close.
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

# Reflect the live price on the chart series so the tip and the recent-high
# calculation match the real market instead of a stale daily bar.
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

# ── Main display ─────────────────────────────────────
if watch_mode:
    # Watch mode: show current price + today's change only
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
    # Dip mode: compute decline from recent high
    recent_high = float(close.max())
    recent_high_date = close.idxmax()
    decline_pct = abs((current_price - recent_high) / recent_high * 100)
    is_below_high = current_price < recent_high

    if not is_below_high:
        decline_color = "#00cc88"
        decline_bg = "rgba(0, 200, 136, 0.08)"
        decline_str = f"+{decline_pct:.1f}%"
        decline_caption = "▲ 高値圏（下落なし）"
    else:
        # Heat ramp: deeper decline → hotter color (buy zone is near)
        if decline_pct >= 20:
            decline_color = "#ff3b30"  # red
            decline_bg = "rgba(255, 59, 48, 0.12)"
            zone_label = "全力買うゾーン"
        elif decline_pct >= 10:
            decline_color = "#ff8c1a"  # deep orange
            decline_bg = "rgba(255, 140, 26, 0.10)"
            zone_label = "半分買うゾーン"
        else:
            decline_color = "#ffcc33"  # yellow-orange
            decline_bg = "rgba(255, 204, 51, 0.08)"
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

# ── Chart ─────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=close.index,
    y=close.values,
    mode="lines",
    name="終値",
    line=dict(color="#5b9cf6", width=2),
    hovertemplate="%{x|%Y/%m/%d}<br>%{y:,.0f}<extra></extra>",
))

# High marker (dip mode only)
if not watch_mode:
    fig.add_trace(go.Scatter(
        x=[recent_high_date],
        y=[recent_high],
        mode="markers+text",
        marker=dict(color="#ffb400", size=10, symbol="triangle-up"),
        text=[f"  高値 {recent_high:,.0f}"],
        textposition="top right",
        textfont=dict(color="#ffb400", size=11),
        name="直近高値",
        hovertemplate="%{x|%Y/%m/%d}<br>%{y:,.0f}<extra>直近高値</extra>",
    ))

# Current price line
fig.add_hline(
    y=current_price,
    line_dash="dot",
    line_color="rgba(255,255,255,0.3)",
    annotation_text=f"現在 {current_price:,.0f}",
    annotation_position="bottom right",
    annotation_font_color="rgba(255,255,255,0.6)",
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,31,46,0.8)",
    margin=dict(l=0, r=0, t=10, b=10),
    height=260,
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

cur_date_str = current_date.strftime("%Y/%m/%d") if hasattr(current_date, "strftime") else str(current_date)

if watch_mode:
    # ── Watch mode: info only, no judgment ───────────
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
    # ── Funds input ──────────────────────────────────
    st.markdown("#### 待機資金")
    if "waiting_funds" not in st.session_state:
        st.session_state.waiting_funds = 1_000_000

    waiting_funds = st.number_input(
        "合計額（円）",
        min_value=0,
        value=st.session_state.waiting_funds,
        step=100_000,
        format="%d",
        key="waiting_funds",
    )

    # ── Judgment ─────────────────────────────────────
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

    # ── Data summary ─────────────────────────────────
    with st.expander("判定の根拠", expanded=True):
        high_date_str = recent_high_date.strftime("%Y/%m/%d") if hasattr(recent_high_date, "strftime") else str(recent_high_date)
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
