import os
import sqlite3
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from dateutil.relativedelta import relativedelta

def resolve_db_path():
    override = os.getenv("STOCKRECORDS_DB_PATH")
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    app_dir = Path(__file__).resolve().parent
    bundled_data_dir = app_dir / "data"
    try:
        bundled_data_dir.mkdir(parents=True, exist_ok=True)
        probe = bundled_data_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return str(bundled_data_dir / "portfolio.db")
    except OSError:
        pass

    fallback_dir = Path.home() / ".stockrecords"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return str(fallback_dir / "portfolio.db")


DB_PATH = resolve_db_path()


@st.cache_resource(show_spinner=False)
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    init_db(conn)
    return conn


def init_db(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,
            quantity REAL NOT NULL, price REAL NOT NULL, fees REAL DEFAULT 0,
            sector TEXT, note TEXT
        );
        CREATE TABLE IF NOT EXISTS targets (
            ticker TEXT PRIMARY KEY, name TEXT, sector TEXT,
            target_price REAL, stop_price REAL, bs_point TEXT, thesis TEXT
        );
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT PRIMARY KEY, price REAL NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL, category TEXT NOT NULL, ticker TEXT,
            title TEXT NOT NULL, impact TEXT
        );
        CREATE TABLE IF NOT EXISTS watch_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_type TEXT NOT NULL,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            underlying TEXT,
            expiry TEXT,
            strike REAL,
            option_type TEXT,
            note TEXT,
            created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_watch_items_symbol ON watch_items(symbol);
        CREATE TABLE IF NOT EXISTS option_strategy_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strategy TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            stock_shares INTEGER NOT NULL DEFAULT 0,
            note TEXT,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    ensure_column(conn, "targets", "research_md", "TEXT")
    conn.commit()
    if conn.execute("SELECT COUNT(1) FROM events").fetchone()[0] == 0:
        today = datetime.today().date()
        defaults = [
            (today + relativedelta(days=7), "macro", None, "US CPI release", "Watch inflation and rate expectations."),
            (today + relativedelta(days=21), "macro", None, "FOMC meeting", "Watch rates and guidance."),
            (today + relativedelta(months=1), "stock", "AAPL", "Apple earnings", "Watch guidance and services growth."),
            (today + relativedelta(months=1, days=10), "stock", "TSLA", "Tesla results window", "Watch deliveries and margins."),
        ]
        for row in defaults:
            conn.execute(
                "INSERT INTO events (event_date, category, ticker, title, impact) VALUES (?, ?, ?, ?, ?)",
                (row[0].isoformat(), row[1], row[2], row[3], row[4]),
            )
        conn.commit()


def ensure_column(conn, table_name, column_name, column_type):
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()


def load_transactions(conn):
    return pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC, id DESC", conn, parse_dates=["date"])


def load_targets(conn):
    return pd.read_sql_query("SELECT * FROM targets ORDER BY ticker", conn)


def load_prices(conn):
    return pd.read_sql_query("SELECT * FROM prices ORDER BY ticker", conn, parse_dates=["updated_at"])


def load_events(conn):
    return pd.read_sql_query("SELECT * FROM events ORDER BY event_date", conn, parse_dates=["event_date"])


def load_watch_items(conn):
    return pd.read_sql_query(
        "SELECT * FROM watch_items ORDER BY asset_type, market, symbol, underlying, expiry, strike, option_type, id",
        conn,
        parse_dates=["created_at", "expiry"],
    )


def load_option_strategy_groups(conn):
    return pd.read_sql_query(
        "SELECT * FROM option_strategy_groups ORDER BY updated_at DESC, id DESC",
        conn,
        parse_dates=["created_at", "updated_at", "expiration"],
    )


def save_transaction(conn, data):
    conn.execute(
        "INSERT INTO transactions (date, ticker, side, quantity, price, fees, sector, note) VALUES (:date, :ticker, :side, :quantity, :price, :fees, :sector, :note)",
        data,
    )
    conn.commit()


def save_target(conn, data):
    conn.execute(
        """
        INSERT INTO targets (ticker, name, sector, target_price, stop_price, bs_point, thesis, research_md)
        VALUES (:ticker, :name, :sector, :target_price, :stop_price, :bs_point, :thesis, :research_md)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name, sector=excluded.sector, target_price=excluded.target_price,
            stop_price=excluded.stop_price, bs_point=excluded.bs_point, thesis=excluded.thesis,
            research_md=excluded.research_md
        """,
        data,
    )
    conn.commit()


def save_price(conn, ticker, price):
    conn.execute(
        """
        INSERT INTO prices (ticker, price, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET price=excluded.price, updated_at=excluded.updated_at
        """,
        (ticker.upper(), price, datetime.utcnow().isoformat()),
    )
    conn.commit()


def save_event(conn, data):
    conn.execute(
        "INSERT INTO events (event_date, category, ticker, title, impact) VALUES (:event_date, :category, :ticker, :title, :impact)",
        data,
    )
    conn.commit()


def save_watch_item(conn, data):
    conn.execute(
        """
        INSERT INTO watch_items (
            asset_type, market, symbol, name, underlying, expiry, strike, option_type, note, created_at
        ) VALUES (
            :asset_type, :market, :symbol, :name, :underlying, :expiry, :strike, :option_type, :note, :created_at
        )
        ON CONFLICT(symbol) DO UPDATE SET
            name=excluded.name,
            note=excluded.note
        """,
        data,
    )
    conn.commit()


def delete_watch_item(conn, item_id):
    conn.execute("DELETE FROM watch_items WHERE id = ?", (item_id,))
    conn.commit()


def save_option_strategy_group(conn, data):
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO option_strategy_groups (
            name, ticker, expiration, strategy, contracts, stock_shares, note, config_json, created_at, updated_at
        ) VALUES (
            :name, :ticker, :expiration, :strategy, :contracts, :stock_shares, :note, :config_json, :created_at, :updated_at
        )
        ON CONFLICT(name) DO UPDATE SET
            ticker=excluded.ticker,
            expiration=excluded.expiration,
            strategy=excluded.strategy,
            contracts=excluded.contracts,
            stock_shares=excluded.stock_shares,
            note=excluded.note,
            config_json=excluded.config_json,
            updated_at=excluded.updated_at
        """,
        {
            **data,
            "created_at": now,
            "updated_at": now,
        },
    )
    conn.commit()


def delete_option_strategy_group(conn, group_id):
    conn.execute("DELETE FROM option_strategy_groups WHERE id = ?", (group_id,))
    conn.commit()


def compute_positions(trades):
    if trades.empty:
        return pd.DataFrame(columns=["ticker", "shares", "avg_cost", "invested", "realized", "sector"])
    rows = []
    for ticker, df in trades.groupby("ticker"):
        df = df.sort_values("date")
        shares = 0.0
        cost = 0.0
        realized = 0.0
        sector = df["sector"].dropna().mode().iloc[0] if df["sector"].notna().any() else ""
        for _, row in df.iterrows():
            qty = float(row["quantity"])
            price = float(row["price"])
            fees = float(row["fees"] or 0)
            if str(row["side"]).lower() == "buy":
                cost += qty * price + fees
                shares += qty
            elif str(row["side"]).lower() == "sell" and shares > 0:
                avg_cost = cost / shares
                realized += (price - avg_cost) * qty - fees
                cost -= avg_cost * qty
                shares -= qty
        rows.append(
            {"ticker": ticker, "shares": shares, "avg_cost": cost / shares if shares else 0, "invested": cost, "realized": realized, "sector": sector}
        )
    return pd.DataFrame(rows).sort_values("ticker")


def merge_market_values(positions, prices):
    if positions.empty:
        return positions
    out = positions.copy()
    lookup = dict(zip(prices["ticker"], prices["price"])) if not prices.empty else {}
    out["last_price"] = out["ticker"].map(lookup)
    out["market_value"] = out["shares"] * out["last_price"]
    out["unrealized"] = (out["last_price"] - out["avg_cost"]) * out["shares"]
    out["pnl_pct"] = ((out["last_price"] - out["avg_cost"]) / out["avg_cost"] * 100).where(out["avg_cost"] > 0)
    return out


def format_number(val):
    if pd.isna(val):
        return "-"
    if abs(val) >= 1e9:
        return f"{val / 1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"{val / 1e6:.2f}M"
    return f"{val:,.2f}"


def normalize_watch_symbol(symbol, market):
    base = (symbol or "").strip().upper()
    if not base:
        return ""
    if market == "CN":
        digits = "".join(ch for ch in base if ch.isdigit())
        if len(digits) != 6:
            return base
        suffix = ".SS" if digits.startswith(("5", "6", "9")) else ".SZ"
        return f"{digits}{suffix}"
    if market == "HK":
        digits = "".join(ch for ch in base if ch.isdigit())
        if digits:
            return f"{digits.zfill(4)}.HK"
        return base if base.endswith(".HK") else f"{base}.HK"
    return base


def normalize_text(value):
    return value.strip().upper() if value else None


def safe_float(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_latest_price_frame(ticker):
    intraday = ticker.history(period="1d", interval="1m", auto_adjust=False, prepost=True)
    if not intraday.empty:
        return intraday, True
    daily = ticker.history(period="5d", auto_adjust=False)
    return daily, False


@st.cache_data(show_spinner=False, ttl=30)
def fetch_equity_quote(symbol):
    ticker = yf.Ticker(symbol)
    hist, is_intraday = get_latest_price_frame(ticker)
    if hist.empty:
        raise ValueError(f"Unable to fetch quote for {symbol}")
    price = safe_float(hist["Close"].iloc[-1])
    prev_close = None
    price_hint = None
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info:
        for key in ("lastPrice", "regularMarketPrice", "last_price"):
            try:
                price_hint = safe_float(fast_info.get(key))
            except Exception:
                price_hint = None
            if price_hint:
                break
        for key in ("previousClose", "regularMarketPreviousClose", "previous_close"):
            try:
                prev_close = safe_float(fast_info.get(key))
            except Exception:
                prev_close = None
            if prev_close:
                break
    if price_hint:
        price = price_hint
    try:
        info = ticker.info
    except Exception:
        info = {}
    if prev_close in (None, 0):
        for key in ("regularMarketPreviousClose", "previousClose"):
            prev_close = safe_float(info.get(key))
            if prev_close not in (None, 0):
                break
    if prev_close in (None, 0):
        daily_hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if len(daily_hist) >= 2:
            prev_close = safe_float(daily_hist["Close"].iloc[-2])
        elif len(hist) > 1:
            prev_close = safe_float(hist["Close"].iloc[-2])
        else:
            prev_close = price
    name = info.get("shortName") or info.get("longName") or symbol
    currency = info.get("currency") or ("HKD" if symbol.endswith(".HK") else "CNY" if symbol.endswith((".SS", ".SZ")) else "USD")
    exchange = info.get("exchange") or info.get("fullExchangeName")
    return {
        "symbol": symbol,
        "name": name,
        "price": price,
        "prev_close": prev_close,
        "change": None if price is None or prev_close in (None, 0) else price - prev_close,
        "change_pct": None if price is None or prev_close in (None, 0) else (price - prev_close) / prev_close * 100,
        "currency": currency,
        "exchange": exchange,
        "quote_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "1m" if is_intraday else "daily",
    }


@st.cache_data(show_spinner=False, ttl=30)
def fetch_option_quote(underlying, expiration, strike, option_type):
    chain = yf.Ticker(underlying).option_chain(expiration)
    frame = chain.calls if option_type == "call" else chain.puts
    if frame.empty:
        raise ValueError("No option contracts returned")
    frame = frame.copy()
    frame["strike_diff"] = (frame["strike"] - float(strike)).abs()
    contract = frame.sort_values(["strike_diff", "openInterest"], ascending=[True, False]).iloc[0]
    bid = safe_float(contract.get("bid"))
    ask = safe_float(contract.get("ask"))
    last_price = safe_float(contract.get("lastPrice"))
    if bid and ask:
        mark = (bid + ask) / 2
    else:
        mark = last_price
    underlying_quote = fetch_equity_quote(underlying)
    return {
        "symbol": contract.get("contractSymbol"),
        "name": f"{underlying} {expiration} {option_type.upper()} {float(contract['strike']):.2f}",
        "price": mark,
        "prev_close": safe_float(contract.get("lastPrice")),
        "change": None if mark is None or last_price is None else mark - last_price,
        "change_pct": None if mark is None or last_price in (None, 0) else (mark - last_price) / last_price * 100,
        "bid": bid,
        "ask": ask,
        "last": last_price,
        "volume": int(contract.get("volume") or 0),
        "open_interest": int(contract.get("openInterest") or 0),
        "implied_volatility": safe_float(contract.get("impliedVolatility")),
        "underlying_price": underlying_quote["price"],
        "currency": underlying_quote["currency"],
        "exchange": "US Options",
        "quote_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "option-chain",
        "matched_strike": float(contract["strike"]),
    }


def fetch_watch_quote(item):
    asset_type = item["asset_type"]
    if asset_type == "option":
        return fetch_option_quote(item["underlying"], item["expiry"], item["strike"], item["option_type"])
    return fetch_equity_quote(item["symbol"])


@st.cache_data(show_spinner=False, ttl=900)
def fetch_stock_snapshot(ticker):
    symbol = ticker.upper().strip()
    stock = yf.Ticker(symbol)
    hist = stock.history(period="5d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"Unable to fetch price data for {symbol}")
    last_close = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
    return {
        "ticker": symbol,
        "price": last_close,
        "change_pct": ((last_close - prev_close) / prev_close * 100) if prev_close else None,
        "expirations": list(stock.options or []),
    }


@st.cache_data(show_spinner=False, ttl=900)
def fetch_option_chain(ticker, expiration):
    chain = yf.Ticker(ticker.upper().strip()).option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    for frame in (calls, puts):
        if frame.empty:
            continue
        frame["mid"] = np.where((frame["bid"] > 0) & (frame["ask"] > 0), (frame["bid"] + frame["ask"]) / 2, frame["lastPrice"])
        frame["openInterest"] = frame["openInterest"].fillna(0).astype(int)
        frame.reset_index(drop=True, inplace=True)
        frame["display"] = frame.apply(
            lambda row: f"K={row['strike']:.2f} | mid={row['mid']:.2f} | bid={row['bid']:.2f} ask={row['ask']:.2f} | IV={row['impliedVolatility']:.1%} | OI={row['openInterest']}",
            axis=1,
        )
    return calls, puts


def make_price_grid(spot):
    return np.linspace(max(spot * 0.5, 0.01), max(spot * 1.5, spot + 1), 141)


def payoff_long_call(prices, strike, premium, qty):
    return (np.maximum(prices - strike, 0) - premium) * qty * 100


def payoff_long_put(prices, strike, premium, qty):
    return (np.maximum(strike - prices, 0) - premium) * qty * 100


def payoff_short_call(prices, strike, premium, qty):
    return (premium - np.maximum(prices - strike, 0)) * qty * 100


def payoff_short_put(prices, strike, premium, qty):
    return (premium - np.maximum(strike - prices, 0)) * qty * 100


def estimate_breakevens(prices, payoff):
    roots = []
    for idx in range(1, len(prices)):
        y1, y2 = payoff[idx - 1], payoff[idx]
        x1, x2 = prices[idx - 1], prices[idx]
        if y1 == 0:
            roots.append(x1)
        elif y2 == 0:
            roots.append(x2)
        elif np.sign(y1) != np.sign(y2):
            roots.append(x1 - y1 * (x2 - x1) / (y2 - y1))
    deduped = []
    for root in roots:
        if not deduped or abs(root - deduped[-1]) > 0.05:
            deduped.append(root)
    return ", ".join(f"{root:.2f}" for root in deduped) if deduped else "-"


def choose_contract(df, label, default_idx=0):
    if df.empty:
        return None
    idx = min(max(default_idx, 0), len(df) - 1)
    selected = st.selectbox(label, options=list(range(len(df))), index=idx, format_func=lambda i: df.iloc[i]["display"])
    row = df.iloc[selected]
    return {"strike": float(row["strike"]), "premium": float(row["mid"])}


def strategy_payload(strategy, ticker, expiration, contracts, stock_shares, spot_price, long_call_leg=None, short_call_leg=None, long_put_leg=None, short_put_leg=None):
    return {
        "strategy": strategy,
        "ticker": ticker,
        "expiration": expiration,
        "contracts": int(contracts),
        "stock_shares": int(stock_shares),
        "spot_price": safe_float(spot_price),
        "legs": {
            "long_call_leg": long_call_leg,
            "short_call_leg": short_call_leg,
            "long_put_leg": long_put_leg,
            "short_put_leg": short_put_leg,
        },
    }


def render_strategy_curve(payload, title_prefix=""):
    prices = make_price_grid(payload["spot_price"])
    curve, metrics = build_strategy_curve(
        payload["strategy"],
        prices,
        payload["spot_price"],
        payload["contracts"],
        payload["legs"].get("long_call_leg"),
        payload["legs"].get("short_call_leg"),
        payload["legs"].get("long_put_leg"),
        payload["legs"].get("short_put_leg"),
        payload["stock_shares"],
    )
    st.subheader(f"{title_prefix}收益曲线")
    st.line_chart(curve.set_index("Underlying Price")["P/L at Expiry"], height=380)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最大收益", metrics["max_profit"])
    m2.metric("最大亏损", metrics["max_loss"])
    m3.metric("盈亏平衡点", metrics["breakeven"])
    m4.metric("当前价位盈亏", metrics["spot_pl"])
    st.caption(metrics["description"])
    focus = curve.iloc[[0, len(curve) // 4, len(curve) // 2, len(curve) * 3 // 4, len(curve) - 1]]
    st.dataframe(focus.style.format({"Underlying Price": "{:.2f}", "P/L at Expiry": "{:.2f}"}), use_container_width=True)


def render_research_card(row):
    title = f"{row['ticker']}"
    if row.get("name"):
        title = f"{title} | {row['name']}"
    with st.container(border=True):
        st.subheader(title)
        meta_cols = st.columns(4)
        meta_cols[0].metric("目标价", "-" if pd.isna(row.get("target_price")) else f"{float(row['target_price']):.2f}")
        meta_cols[1].metric("止损价", "-" if pd.isna(row.get("stop_price")) else f"{float(row['stop_price']):.2f}")
        meta_cols[2].write(f"行业：{row.get('sector') or '-'}")
        meta_cols[3].write(f"关键点位：{row.get('bs_point') or '-'}")
        if row.get("thesis"):
            st.markdown("**投资逻辑**")
            st.write(row["thesis"])
        st.markdown("**调研内容**")
        research_body = row.get("research_md") or "_暂无调研内容_"
        st.markdown(research_body, unsafe_allow_html=True)


def build_strategy_curve(strategy, prices, spot_price, contracts, long_call_leg=None, short_call_leg=None, long_put_leg=None, short_put_leg=None, stock_shares=100):
    payoff = np.zeros_like(prices, dtype=float)
    desc = []
    max_profit = None
    max_loss = None
    if strategy == "Long Call":
        payoff += payoff_long_call(prices, long_call_leg["strike"], long_call_leg["premium"], contracts)
        max_profit, max_loss = "Unlimited", format_number(-long_call_leg["premium"] * contracts * 100)
        desc.append(f"Long Call {long_call_leg['strike']:.2f}")
    elif strategy == "Long Put":
        payoff += payoff_long_put(prices, long_put_leg["strike"], long_put_leg["premium"], contracts)
        max_profit = format_number((long_put_leg["strike"] - long_put_leg["premium"]) * contracts * 100)
        max_loss = format_number(-long_put_leg["premium"] * contracts * 100)
        desc.append(f"Long Put {long_put_leg['strike']:.2f}")
    elif strategy == "Covered Call":
        payoff += (prices - spot_price) * stock_shares
        payoff += payoff_short_call(prices, short_call_leg["strike"], short_call_leg["premium"], contracts)
        max_profit = format_number((short_call_leg["strike"] - spot_price + short_call_leg["premium"]) * contracts * 100)
        max_loss = format_number((-spot_price + short_call_leg["premium"]) * contracts * 100)
        desc += [f"Long Stock {stock_shares}", f"Short Call {short_call_leg['strike']:.2f}"]
    elif strategy == "Protective Put":
        payoff += (prices - spot_price) * stock_shares
        payoff += payoff_long_put(prices, long_put_leg["strike"], long_put_leg["premium"], contracts)
        max_profit = "Unlimited"
        max_loss = format_number((long_put_leg["strike"] - spot_price - long_put_leg["premium"]) * contracts * 100)
        desc += [f"Long Stock {stock_shares}", f"Long Put {long_put_leg['strike']:.2f}"]
    elif strategy == "Bull Call Spread":
        payoff += payoff_long_call(prices, long_call_leg["strike"], long_call_leg["premium"], contracts)
        payoff += payoff_short_call(prices, short_call_leg["strike"], short_call_leg["premium"], contracts)
        debit = long_call_leg["premium"] - short_call_leg["premium"]
        width = short_call_leg["strike"] - long_call_leg["strike"]
        max_profit, max_loss = format_number((width - debit) * contracts * 100), format_number(-debit * contracts * 100)
        desc += [f"Long Call {long_call_leg['strike']:.2f}", f"Short Call {short_call_leg['strike']:.2f}"]
    elif strategy == "Bull Put Spread":
        payoff += payoff_short_put(prices, short_put_leg["strike"], short_put_leg["premium"], contracts)
        payoff += payoff_long_put(prices, long_put_leg["strike"], long_put_leg["premium"], contracts)
        net_credit = short_put_leg["premium"] - long_put_leg["premium"]
        width = short_put_leg["strike"] - long_put_leg["strike"]
        max_profit, max_loss = format_number(net_credit * contracts * 100), format_number((net_credit - width) * contracts * 100)
        desc += [f"Short Put {short_put_leg['strike']:.2f}", f"Long Lower Put {long_put_leg['strike']:.2f}"]
    elif strategy == "Bear Put Spread":
        payoff += payoff_long_put(prices, long_put_leg["strike"], long_put_leg["premium"], contracts)
        payoff += payoff_short_put(prices, short_put_leg["strike"], short_put_leg["premium"], contracts)
        debit = long_put_leg["premium"] - short_put_leg["premium"]
        width = long_put_leg["strike"] - short_put_leg["strike"]
        max_profit, max_loss = format_number((width - debit) * contracts * 100), format_number(-debit * contracts * 100)
        desc += [f"Long Put {long_put_leg['strike']:.2f}", f"Short Put {short_put_leg['strike']:.2f}"]
    elif strategy == "Long Straddle":
        payoff += payoff_long_call(prices, long_call_leg["strike"], long_call_leg["premium"], contracts)
        payoff += payoff_long_put(prices, long_put_leg["strike"], long_put_leg["premium"], contracts)
        total_debit = long_call_leg["premium"] + long_put_leg["premium"]
        max_profit, max_loss = "Unlimited", format_number(-total_debit * contracts * 100)
        desc += [f"Long ATM Call {long_call_leg['strike']:.2f}", f"Long ATM Put {long_put_leg['strike']:.2f}"]
    elif strategy == "Iron Condor":
        payoff += payoff_short_put(prices, short_put_leg["strike"], short_put_leg["premium"], contracts)
        payoff += payoff_long_put(prices, long_put_leg["strike"], long_put_leg["premium"], contracts)
        payoff += payoff_short_call(prices, short_call_leg["strike"], short_call_leg["premium"], contracts)
        payoff += payoff_long_call(prices, long_call_leg["strike"], long_call_leg["premium"], contracts)
        net_credit = short_put_leg["premium"] - long_put_leg["premium"] + short_call_leg["premium"] - long_call_leg["premium"]
        width = max(short_put_leg["strike"] - long_put_leg["strike"], long_call_leg["strike"] - short_call_leg["strike"])
        max_profit, max_loss = format_number(net_credit * contracts * 100), format_number((net_credit - width) * contracts * 100)
        desc += [f"Short Put {short_put_leg['strike']:.2f}", f"Long Put {long_put_leg['strike']:.2f}", f"Short Call {short_call_leg['strike']:.2f}", f"Long Call {long_call_leg['strike']:.2f}"]
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    curve = pd.DataFrame({"Underlying Price": prices, "P/L at Expiry": payoff})
    return curve, {"max_profit": max_profit, "max_loss": max_loss, "breakeven": estimate_breakevens(prices, payoff), "spot_pl": format_number(float(np.interp(spot_price, prices, payoff))), "description": " | ".join(desc)}


def page_dashboard(conn):
    st.header("投资组合总览")
    positions = merge_market_values(compute_positions(load_transactions(conn)), load_prices(conn))
    if positions.empty:
        st.info("还没有交易记录，先添加第一笔交易吧。")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("投入成本", format_number(positions["invested"].sum()))
    c2.metric("持仓市值", format_number(positions["market_value"].sum(min_count=1)))
    c3.metric("浮动盈亏", format_number(positions["unrealized"].sum(min_count=1)))
    c4.metric("已实现盈亏", format_number(positions["realized"].sum()))
    st.subheader("持仓明细")
    table = positions[["ticker", "sector", "shares", "avg_cost", "last_price", "market_value", "unrealized", "pnl_pct", "realized"]].rename(columns={"ticker": "代码", "sector": "行业", "shares": "持仓数量", "avg_cost": "持仓成本", "last_price": "最新价", "market_value": "持仓市值", "unrealized": "浮动盈亏", "pnl_pct": "浮动收益率", "realized": "已实现盈亏"})
    st.dataframe(table.style.format({"持仓数量": "{:.2f}", "持仓成本": "{:.2f}", "最新价": "{:.2f}", "持仓市值": "{:.2f}", "浮动盈亏": "{:.2f}", "浮动收益率": "{:.2f}%", "已实现盈亏": "{:.2f}"}), use_container_width=True)
    with st.form("update_price"):
        ticker = st.selectbox("代码", positions["ticker"].tolist())
        price = st.number_input("最新价", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("保存价格")
    if submitted:
        save_price(conn, ticker, price)
        st.success(f"已保存 {ticker} 的最新价。")
        st.rerun()
    st.subheader("行业分布")
    st.bar_chart(positions.groupby("sector", dropna=False)["market_value"].sum().sort_values(ascending=False), height=260)


def page_transactions(conn):
    st.header("交易记录")
    with st.form("trade_form"):
        cols = st.columns(3)
        trade_date = cols[0].date_input("日期", value=date.today())
        ticker = cols[1].text_input("代码", placeholder="AAPL")
        side = cols[2].selectbox("方向", ["Buy", "Sell"], format_func=lambda x: "买入" if x == "Buy" else "卖出")
        qcol, pcol, fcol = st.columns(3)
        qty = qcol.number_input("数量", min_value=0.0, step=1.0)
        price = pcol.number_input("成交价", min_value=0.0, step=0.01)
        fees = fcol.number_input("费用", min_value=0.0, step=0.01, value=0.0)
        sector = st.text_input("行业")
        note = st.text_area("备注", height=80)
        submitted = st.form_submit_button("保存")
    if submitted:
        if not ticker or qty <= 0 or price <= 0:
            st.error("代码、数量和成交价不能为空。")
        else:
            save_transaction(conn, {"date": trade_date.isoformat(), "ticker": ticker.upper(), "side": side.lower(), "quantity": qty, "price": price, "fees": fees, "sector": sector, "note": note})
            st.success("交易已保存。")
            st.rerun()
    trades = load_transactions(conn)
    if not trades.empty:
        st.subheader("历史记录")
        st.dataframe(trades, use_container_width=True)


def page_targets(conn):
    st.header("个股研究")
    with st.form("target_form"):
        cols = st.columns(3)
        ticker = cols[0].text_input("代码", placeholder="TSLA")
        name = cols[1].text_input("名称")
        sector = cols[2].text_input("行业")
        tcol, scol = st.columns(2)
        target_price = tcol.number_input("目标价", min_value=0.0, step=0.1)
        stop_price = scol.number_input("止损价", min_value=0.0, step=0.1)
        bs_point = st.text_input("关键买卖点位")
        thesis = st.text_area("投资逻辑 / 催化剂", height=100)
        research_md = st.text_area(
            "调研内容（支持 Markdown / 简单 HTML）",
            height=220,
            placeholder="例如：\n## 结论\n- 核心观点\n- 风险点\n\n> 可粘贴 Markdown 内容",
        )
        submitted = st.form_submit_button("保存 / 更新")
    if submitted:
        if not ticker:
            st.error("代码不能为空。")
        else:
            save_target(
                conn,
                {
                    "ticker": ticker.upper(),
                    "name": name,
                    "sector": sector,
                    "target_price": target_price or None,
                    "stop_price": stop_price or None,
                    "bs_point": bs_point,
                    "thesis": thesis,
                    "research_md": research_md,
                },
            )
            st.success("个股研究已保存。")
            st.rerun()
    targets = load_targets(conn)
    if targets.empty:
        st.info("还没有任何个股研究，先新增一条吧。")
        return
    st.subheader("研究总表")
    summary = targets[["ticker", "name", "sector", "target_price", "stop_price", "bs_point"]].rename(
        columns={"ticker": "代码", "name": "名称", "sector": "行业", "target_price": "目标价", "stop_price": "止损价", "bs_point": "关键点位"}
    )
    st.dataframe(summary, use_container_width=True)
    st.subheader("研究展示")
    for _, row in targets.iterrows():
        label = f"{row['ticker']} | {row['name'] or '未命名'}"
        with st.expander(label, expanded=False):
            render_research_card(row.to_dict())
    with st.expander("Markdown 预览说明"):
        st.markdown(
            """
支持常见 Markdown 语法：
- `#`、`##` 标题
- `-` 列表
- `**加粗**`
- 表格、引用、代码块

也支持简单 HTML 标签，例如 `<br>`、`<span>`、`<div>`。
            """
        )


def page_realtime_watchlist(conn):
    st.header("实时自选")
    st.caption("支持 A 股、港股、美股与美股期权。股票代码示例：600519、0700、AAPL；期权按标的+到期日+行权价添加。")

    with st.expander("新增自选标的", expanded=True):
        asset_type = st.radio("资产类型", ["stock", "option"], horizontal=True, format_func=lambda x: "股票 / ETF" if x == "stock" else "期权")
        if asset_type == "stock":
            col1, col2, col3 = st.columns(3)
            market = col1.selectbox("市场", ["CN", "HK", "US"], format_func=lambda x: {"CN": "A 股", "HK": "港股", "US": "美股"}[x])
            raw_symbol = col2.text_input("代码", placeholder="600519 / 0700 / AAPL")
            name = col3.text_input("备注名称", placeholder="贵州茅台 / 腾讯 / Apple")
            note = st.text_area("备注", height=70)
            if st.button("添加股票自选", type="primary"):
                symbol = normalize_watch_symbol(raw_symbol, market)
                if not symbol:
                    st.error("请输入有效代码。")
                else:
                    save_watch_item(
                        conn,
                        {
                            "asset_type": "stock",
                            "market": market,
                            "symbol": symbol,
                            "name": name,
                            "underlying": None,
                            "expiry": None,
                            "strike": None,
                            "option_type": None,
                            "note": note,
                            "created_at": datetime.utcnow().isoformat(),
                        },
                    )
                    st.success(f"已添加 {symbol}")
                    st.rerun()
        else:
            col1, col2, col3, col4 = st.columns(4)
            underlying_raw = col1.text_input("标的代码", placeholder="AAPL")
            expiry = col2.date_input("到期日", value=date.today() + timedelta(days=30))
            option_type = col3.selectbox("期权类型", ["call", "put"], format_func=lambda x: "Call" if x == "call" else "Put")
            strike = col4.number_input("行权价", min_value=0.01, step=0.5)
            option_name = st.text_input("备注名称", placeholder="AAPL 近月看涨")
            note = st.text_area("备注", height=70, key="option_note")
            if st.button("添加期权自选", type="primary"):
                underlying = normalize_watch_symbol(underlying_raw, "US")
                if not underlying or strike <= 0:
                    st.error("请填写有效的美股标的和行权价。")
                else:
                    save_watch_item(
                        conn,
                        {
                            "asset_type": "option",
                            "market": "US",
                            "symbol": f"{underlying}-{expiry.isoformat()}-{option_type.upper()}-{strike:.2f}",
                            "name": option_name,
                            "underlying": underlying,
                            "expiry": expiry.isoformat(),
                            "strike": strike,
                            "option_type": option_type,
                            "note": note,
                            "created_at": datetime.utcnow().isoformat(),
                        },
                    )
                    st.success("已添加期权自选")
                    st.rerun()

    items = load_watch_items(conn)
    if items.empty:
        st.info("还没有自选标的，先在上方添加一个。")
        return

    toolbar_left, toolbar_mid, toolbar_right = st.columns([1, 1, 3])
    refresh_now = toolbar_left.button("立即刷新")
    auto_refresh = toolbar_mid.checkbox("15 秒自动刷新", value=False)
    if refresh_now:
        st.cache_data.clear()
        st.rerun()
    if auto_refresh:
        components.html(
            """
            <script>
            setTimeout(function() {
              window.parent.location.reload();
            }, 15000);
            </script>
            """,
            height=0,
        )

    rows = []
    errors = []
    for _, item in items.iterrows():
        try:
            quote = fetch_watch_quote(item)
            rows.append(
                {
                    "ID": int(item["id"]),
                    "类型": "期权" if item["asset_type"] == "option" else "股票",
                    "市场": {"CN": "A 股", "HK": "港股", "US": "美股"}[item["market"]] if item["asset_type"] != "option" else "美股期权",
                    "代码": item["symbol"],
                    "名称": item["name"] or quote.get("name") or "",
                    "最新价": quote.get("price"),
                    "涨跌额": quote.get("change"),
                    "涨跌幅%": quote.get("change_pct"),
                    "币种": quote.get("currency"),
                    "交易所": quote.get("exchange"),
                    "更新时间": quote.get("quote_time"),
                    "备注": item["note"] or "",
                    "细节": (
                        f"标的 {item['underlying']} | 到期 {pd.to_datetime(item['expiry']).date()} | "
                        f"{str(item['option_type']).upper()} | 行权价 {float(item['strike']):.2f} | "
                        f"Bid {quote.get('bid') or 0:.2f} / Ask {quote.get('ask') or 0:.2f} | "
                        f"OI {quote.get('open_interest', 0)} | IV {quote.get('implied_volatility') or 0:.2%}"
                    )
                    if item["asset_type"] == "option"
                    else f"数据源 {quote.get('source')}",
                }
            )
        except Exception as exc:
            errors.append(f"{item['symbol']}: {exc}")

    if rows:
        quote_df = pd.DataFrame(rows)
        st.subheader("行情列表")
        st.dataframe(
            quote_df.style.format({"最新价": "{:.2f}", "涨跌额": "{:+.2f}", "涨跌幅%": "{:+.2f}%"}),
            use_container_width=True,
            height=420,
        )
    else:
        st.warning("暂时没有成功拉到任何行情。")

    if errors:
        st.warning("部分标的拉取失败：")
        for err in errors:
            st.write(f"- {err}")

    delete_options = {f"{row['ID']} | {row['代码']} | {row['名称']}": int(row["ID"]) for row in rows}
    if delete_options:
        delete_col1, delete_col2 = st.columns([3, 1])
        delete_choice = delete_col1.selectbox("删除自选", options=list(delete_options.keys()))
        if delete_col2.button("删除所选"):
            delete_watch_item(conn, delete_options[delete_choice])
            st.success("已删除自选标的")
            st.rerun()


def page_events(conn):
    st.header("事件日历")
    today = datetime.today().date()
    upcoming = load_events(conn)
    upcoming = upcoming[(upcoming["event_date"].dt.date >= today) & (upcoming["event_date"].dt.date <= today + relativedelta(months=6))]
    if upcoming.empty:
        st.info("未来半年还没有事件，下面可以自行新增。")
    else:
        st.subheader("时间线")
        st.dataframe(upcoming.rename(columns={"event_date": "日期", "category": "类别", "ticker": "代码", "title": "事件", "impact": "影响"}), use_container_width=True)
    with st.form("event_form"):
        event_date = st.date_input("日期", value=today + timedelta(days=7))
        category = st.selectbox("类别", ["macro", "stock"], format_func=lambda x: "宏观" if x == "macro" else "个股")
        ticker = st.text_input("代码")
        title = st.text_input("事件标题")
        impact = st.text_area("影响说明", height=80)
        submitted = st.form_submit_button("保存事件")
    if submitted:
        if not title:
            st.error("事件标题不能为空。")
        else:
            save_event(conn, {"event_date": event_date.isoformat(), "category": category, "ticker": ticker.upper() if ticker else None, "title": title, "impact": impact})
            st.success("事件已保存。")
            st.rerun()


def page_options(conn):
    st.header("期权策略")
    st.caption("支持期权策略到期收益曲线查看，并可将策略组持久化保存到本地数据库。")
    ticker = st.text_input("标的代码", value="AAPL", help="示例：AAPL、TSLA、SPY").upper().strip()
    if not ticker:
        st.info("请输入标的代码后继续。")
        return
    try:
        snapshot = fetch_stock_snapshot(ticker)
    except Exception as exc:
        st.error(f"获取价格失败：{exc}")
        st.info("请尝试流动性更好的美股期权标的，例如 AAPL、TSLA、SPY、QQQ。")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("标的现价", f"{snapshot['price']:.2f}")
    c2.metric("近期涨跌幅", "-" if snapshot["change_pct"] is None else f"{snapshot['change_pct']:+.2f}%")
    c3.metric("可选到期日", str(len(snapshot["expirations"])))
    if not snapshot["expirations"]:
        st.warning("该标的没有返回可用到期日。")
        return
    row = st.columns(3)
    expiration = row[0].selectbox("到期日", snapshot["expirations"])
    strategy = row[1].selectbox("策略", ["Long Call", "Long Put", "Covered Call", "Protective Put", "Bull Call Spread", "Bull Put Spread", "Bear Put Spread", "Long Straddle", "Iron Condor"])
    contracts = int(row[2].number_input("合约数", min_value=1, max_value=20, value=1, step=1))
    try:
        calls, puts = fetch_option_chain(ticker, expiration)
    except Exception as exc:
        st.error(f"获取期权链失败：{exc}")
        return
    if calls.empty and puts.empty:
        st.warning("该到期日没有返回可用期权。")
        return
    call_center = int((calls["strike"] - snapshot["price"]).abs().idxmin()) if not calls.empty else 0
    put_center = int((puts["strike"] - snapshot["price"]).abs().idxmin()) if not puts.empty else 0
    prices = make_price_grid(snapshot["price"])
    long_call_leg = short_call_leg = long_put_leg = short_put_leg = None
    stock_shares = contracts * 100
    st.subheader("腿部选择")
    if strategy == "Long Call":
        long_call_leg = choose_contract(calls, "买入 Call", call_center)
    elif strategy == "Long Put":
        long_put_leg = choose_contract(puts, "买入 Put", put_center)
    elif strategy == "Covered Call":
        stock_shares = int(st.number_input("正股持仓股数", min_value=100, value=contracts * 100, step=100))
        contracts = max(1, stock_shares // 100)
        short_call_leg = choose_contract(calls, "卖出 Call", call_center)
    elif strategy == "Protective Put":
        stock_shares = int(st.number_input("正股持仓股数", min_value=100, value=contracts * 100, step=100))
        contracts = max(1, stock_shares // 100)
        long_put_leg = choose_contract(puts, "买入 Put", put_center)
    elif strategy == "Bull Call Spread":
        left, right = st.columns(2)
        with left:
            long_call_leg = choose_contract(calls, "买入低行权价 Call", max(call_center - 1, 0))
        with right:
            short_call_leg = choose_contract(calls, "卖出高行权价 Call", min(call_center + 1, len(calls) - 1))
    elif strategy == "Bull Put Spread":
        left, right = st.columns(2)
        with left:
            short_put_leg = choose_contract(puts, "卖出高行权价 Put", min(put_center + 1, len(puts) - 1))
        with right:
            long_put_leg = choose_contract(puts, "买入低行权价 Put", max(put_center - 1, 0))
    elif strategy == "Bear Put Spread":
        left, right = st.columns(2)
        with left:
            long_put_leg = choose_contract(puts, "买入高行权价 Put", min(put_center + 1, len(puts) - 1))
        with right:
            short_put_leg = choose_contract(puts, "卖出低行权价 Put", max(put_center - 1, 0))
    elif strategy == "Long Straddle":
        left, right = st.columns(2)
        with left:
            long_call_leg = choose_contract(calls, "平值 Call", call_center)
        with right:
            long_put_leg = choose_contract(puts, "平值 Put", put_center)
    elif strategy == "Iron Condor":
        left, right = st.columns(2)
        with left:
            short_put_leg = choose_contract(puts, "卖出 Put", max(put_center - 1, 0))
            long_put_leg = choose_contract(puts, "买入更低行权价 Put", max(put_center - 3, 0))
        with right:
            short_call_leg = choose_contract(calls, "卖出 Call", min(call_center + 1, len(calls) - 1))
            long_call_leg = choose_contract(calls, "买入更高行权价 Call", min(call_center + 3, len(calls) - 1))
    try:
        current_payload = strategy_payload(
            strategy,
            ticker,
            expiration,
            contracts,
            stock_shares,
            snapshot["price"],
            long_call_leg,
            short_call_leg,
            long_put_leg,
            short_put_leg,
        )
        curve, metrics = build_strategy_curve(strategy, prices, snapshot["price"], contracts, long_call_leg, short_call_leg, long_put_leg, short_put_leg, stock_shares)
    except Exception as exc:
        st.error(f"策略计算失败：{exc}")
        st.info("请检查当前价差组合里多空腿的行权价顺序是否正确。")
        return
    st.subheader("到期收益曲线")
    st.line_chart(curve.set_index("Underlying Price")["P/L at Expiry"], height=380)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Profit", metrics["max_profit"])
    m2.metric("Max Loss", metrics["max_loss"])
    m3.metric("Breakeven", metrics["breakeven"])
    m4.metric("P/L at Spot", metrics["spot_pl"])
    st.caption(metrics["description"])
    st.subheader("关键价格点")
    focus = curve.iloc[[0, len(curve) // 4, len(curve) // 2, len(curve) * 3 // 4, len(curve) - 1]]
    st.dataframe(focus.style.format({"Underlying Price": "{:.2f}", "P/L at Expiry": "{:.2f}"}), use_container_width=True)

    st.subheader("保存策略组")
    default_name = f"{ticker} {expiration} {strategy}"
    save_col1, save_col2 = st.columns([2, 3])
    strategy_name = save_col1.text_input("策略组名称", value=default_name, key="strategy_group_name")
    strategy_note = save_col2.text_input("备注", key="strategy_group_note")
    if st.button("保存当前策略组", type="primary"):
        if not strategy_name.strip():
            st.error("策略组名称不能为空。")
        else:
            save_option_strategy_group(
                conn,
                {
                    "name": strategy_name.strip(),
                    "ticker": ticker,
                    "expiration": expiration,
                    "strategy": strategy,
                    "contracts": int(contracts),
                    "stock_shares": int(stock_shares),
                    "note": strategy_note.strip(),
                    "config_json": json.dumps(current_payload, ensure_ascii=False),
                },
            )
            st.success("策略组已保存。")
            st.rerun()

    saved_groups = load_option_strategy_groups(conn)
    st.subheader("已保存策略组")
    if saved_groups.empty:
        st.info("还没有已保存的期权策略组。")
    else:
        display_options = {
            f"{row['name']} | {row['ticker']} | {row['strategy']} | {pd.to_datetime(row['expiration']).date()}": int(row["id"])
            for _, row in saved_groups.iterrows()
        }
        selected_label = st.selectbox("选择策略组", options=list(display_options.keys()))
        selected_group = saved_groups[saved_groups["id"] == display_options[selected_label]].iloc[0]
        saved_payload = json.loads(selected_group["config_json"])
        saved_spot = saved_payload.get("spot_price")
        try:
            latest_snapshot = fetch_stock_snapshot(saved_payload["ticker"])
            if latest_snapshot.get("price"):
                saved_payload["spot_price"] = latest_snapshot["price"]
        except Exception:
            saved_payload["spot_price"] = saved_spot
        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("标的", saved_payload["ticker"])
        meta2.metric("到期日", str(saved_payload["expiration"]))
        meta3.metric("策略", saved_payload["strategy"])
        meta4.metric("合约数", str(saved_payload["contracts"]))
        if selected_group["note"]:
            st.caption(selected_group["note"])
        render_strategy_curve(saved_payload, title_prefix="已保存策略组")
        delete_col1, delete_col2 = st.columns([3, 1])
        delete_col1.caption(f"最近更新：{selected_group['updated_at']}")
        if delete_col2.button("删除该策略组"):
            delete_option_strategy_group(conn, int(selected_group["id"]))
            st.success("策略组已删除。")
            st.rerun()

    st.subheader("期权链参考")
    left, right = st.columns(2)
    if not calls.empty:
        left.dataframe(calls[["strike", "bid", "ask", "lastPrice", "mid", "impliedVolatility", "openInterest"]].rename(columns={"strike": "Call行权价", "bid": "买价", "ask": "卖价", "lastPrice": "最新成交", "mid": "中间价", "impliedVolatility": "隐含波动率", "openInterest": "未平仓量"}).style.format({"Call行权价": "{:.2f}", "买价": "{:.2f}", "卖价": "{:.2f}", "最新成交": "{:.2f}", "中间价": "{:.2f}", "隐含波动率": "{:.2%}"}), use_container_width=True, height=320)
    if not puts.empty:
        right.dataframe(puts[["strike", "bid", "ask", "lastPrice", "mid", "impliedVolatility", "openInterest"]].rename(columns={"strike": "Put行权价", "bid": "买价", "ask": "卖价", "lastPrice": "最新成交", "mid": "中间价", "impliedVolatility": "隐含波动率", "openInterest": "未平仓量"}).style.format({"Put行权价": "{:.2f}", "买价": "{:.2f}", "卖价": "{:.2f}", "最新成交": "{:.2f}", "中间价": "{:.2f}", "隐含波动率": "{:.2%}"}), use_container_width=True, height=320)


def main():
    st.set_page_config(page_title="本地投资记录", layout="wide")
    conn = get_conn()
    menu = {
        "投资组合": page_dashboard,
        "交易记录": page_transactions,
        "个股研究": page_targets,
        "实时自选": page_realtime_watchlist,
        "事件日历": page_events,
        "期权策略": page_options,
    }
    choice = st.sidebar.radio("菜单", list(menu.keys()))
    st.sidebar.caption(f"数据持久化保存在 {DB_PATH}")
    menu[choice](conn)


if __name__ == "__main__":
    main()
