import os
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "portfolio.db")


@st.cache_resource(show_spinner=False)
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
        """
    )
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


def load_transactions(conn):
    return pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC, id DESC", conn, parse_dates=["date"])


def load_targets(conn):
    return pd.read_sql_query("SELECT * FROM targets ORDER BY ticker", conn)


def load_prices(conn):
    return pd.read_sql_query("SELECT * FROM prices ORDER BY ticker", conn, parse_dates=["updated_at"])


def load_events(conn):
    return pd.read_sql_query("SELECT * FROM events ORDER BY event_date", conn, parse_dates=["event_date"])


def save_transaction(conn, data):
    conn.execute(
        "INSERT INTO transactions (date, ticker, side, quantity, price, fees, sector, note) VALUES (:date, :ticker, :side, :quantity, :price, :fees, :sector, :note)",
        data,
    )
    conn.commit()


def save_target(conn, data):
    conn.execute(
        """
        INSERT INTO targets (ticker, name, sector, target_price, stop_price, bs_point, thesis)
        VALUES (:ticker, :name, :sector, :target_price, :stop_price, :bs_point, :thesis)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name, sector=excluded.sector, target_price=excluded.target_price,
            stop_price=excluded.stop_price, bs_point=excluded.bs_point, thesis=excluded.thesis
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
    st.header("Portfolio Dashboard")
    positions = merge_market_values(compute_positions(load_transactions(conn)), load_prices(conn))
    if positions.empty:
        st.info("No trades yet. Add your first transaction to start tracking.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invested Cost", format_number(positions["invested"].sum()))
    c2.metric("Market Value", format_number(positions["market_value"].sum(min_count=1)))
    c3.metric("Unrealized P/L", format_number(positions["unrealized"].sum(min_count=1)))
    c4.metric("Realized P/L", format_number(positions["realized"].sum()))
    st.subheader("Positions")
    table = positions[["ticker", "sector", "shares", "avg_cost", "last_price", "market_value", "unrealized", "pnl_pct", "realized"]].rename(columns={"ticker": "Ticker", "sector": "Sector", "shares": "Shares", "avg_cost": "Avg Cost", "last_price": "Last Price", "market_value": "Market Value", "unrealized": "Unrealized", "pnl_pct": "Unrealized %", "realized": "Realized"})
    st.dataframe(table.style.format({"Shares": "{:.2f}", "Avg Cost": "{:.2f}", "Last Price": "{:.2f}", "Market Value": "{:.2f}", "Unrealized": "{:.2f}", "Unrealized %": "{:.2f}%", "Realized": "{:.2f}"}), use_container_width=True)
    with st.form("update_price"):
        ticker = st.selectbox("Ticker", positions["ticker"].tolist())
        price = st.number_input("Last Price", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("Save Price")
    if submitted:
        save_price(conn, ticker, price)
        st.success(f"Saved latest price for {ticker}.")
        st.rerun()
    st.subheader("Sector Mix")
    st.bar_chart(positions.groupby("sector", dropna=False)["market_value"].sum().sort_values(ascending=False), height=260)


def page_transactions(conn):
    st.header("Transactions")
    with st.form("trade_form"):
        cols = st.columns(3)
        trade_date = cols[0].date_input("Date", value=date.today())
        ticker = cols[1].text_input("Ticker", placeholder="AAPL")
        side = cols[2].selectbox("Side", ["Buy", "Sell"])
        qcol, pcol, fcol = st.columns(3)
        qty = qcol.number_input("Quantity", min_value=0.0, step=1.0)
        price = pcol.number_input("Price", min_value=0.0, step=0.01)
        fees = fcol.number_input("Fees", min_value=0.0, step=0.01, value=0.0)
        sector = st.text_input("Sector")
        note = st.text_area("Note", height=80)
        submitted = st.form_submit_button("Save")
    if submitted:
        if not ticker or qty <= 0 or price <= 0:
            st.error("Ticker, quantity, and price are required.")
        else:
            save_transaction(conn, {"date": trade_date.isoformat(), "ticker": ticker.upper(), "side": side.lower(), "quantity": qty, "price": price, "fees": fees, "sector": sector, "note": note})
            st.success("Transaction saved.")
            st.rerun()
    trades = load_transactions(conn)
    if not trades.empty:
        st.subheader("History")
        st.dataframe(trades, use_container_width=True)


def page_targets(conn):
    st.header("Watchlist and Plan")
    with st.form("target_form"):
        cols = st.columns(3)
        ticker = cols[0].text_input("Ticker", placeholder="TSLA")
        name = cols[1].text_input("Name")
        sector = cols[2].text_input("Sector")
        tcol, scol = st.columns(2)
        target_price = tcol.number_input("Target Price", min_value=0.0, step=0.1)
        stop_price = scol.number_input("Stop Price", min_value=0.0, step=0.1)
        bs_point = st.text_input("Key Buy/Sell Levels")
        thesis = st.text_area("Thesis / Catalysts", height=100)
        submitted = st.form_submit_button("Save / Update")
    if submitted:
        if not ticker:
            st.error("Ticker is required.")
        else:
            save_target(conn, {"ticker": ticker.upper(), "name": name, "sector": sector, "target_price": target_price or None, "stop_price": stop_price or None, "bs_point": bs_point, "thesis": thesis})
            st.success("Watchlist item saved.")
            st.rerun()
    targets = load_targets(conn)
    if not targets.empty:
        st.subheader("Current Watchlist")
        st.dataframe(targets, use_container_width=True)


def page_events(conn):
    st.header("Upcoming Events")
    today = datetime.today().date()
    upcoming = load_events(conn)
    upcoming = upcoming[(upcoming["event_date"].dt.date >= today) & (upcoming["event_date"].dt.date <= today + relativedelta(months=6))]
    if upcoming.empty:
        st.info("No upcoming events yet. Add your own below.")
    else:
        st.subheader("Timeline")
        st.dataframe(upcoming.rename(columns={"event_date": "Date", "category": "Category", "ticker": "Ticker", "title": "Event", "impact": "Impact"}), use_container_width=True)
    with st.form("event_form"):
        event_date = st.date_input("Date", value=today + timedelta(days=7))
        category = st.selectbox("Category", ["macro", "stock"])
        ticker = st.text_input("Ticker")
        title = st.text_input("Title")
        impact = st.text_area("Impact", height=80)
        submitted = st.form_submit_button("Save Event")
    if submitted:
        if not title:
            st.error("Event title is required.")
        else:
            save_event(conn, {"event_date": event_date.isoformat(), "category": category, "ticker": ticker.upper() if ticker else None, "title": title, "impact": impact})
            st.success("Event saved.")
            st.rerun()


def page_options():
    st.header("Options Payoff Curves")
    st.caption("Load stock and option-chain data and inspect expiry payoff curves across strategies.")
    ticker = st.text_input("Ticker", value="AAPL", help="Examples: AAPL, TSLA, SPY").upper().strip()
    if not ticker:
        st.info("Enter a ticker to continue.")
        return
    try:
        snapshot = fetch_stock_snapshot(ticker)
    except Exception as exc:
        st.error(f"Price fetch failed: {exc}")
        st.info("Try a liquid US options ticker such as AAPL, TSLA, SPY, or QQQ.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"{snapshot['price']:.2f}")
    c2.metric("Recent Change", "-" if snapshot["change_pct"] is None else f"{snapshot['change_pct']:+.2f}%")
    c3.metric("Expirations", str(len(snapshot["expirations"])))
    if not snapshot["expirations"]:
        st.warning("No option expirations were returned for this ticker.")
        return
    row = st.columns(3)
    expiration = row[0].selectbox("Expiration", snapshot["expirations"])
    strategy = row[1].selectbox("Strategy", ["Long Call", "Long Put", "Covered Call", "Protective Put", "Bull Call Spread", "Bull Put Spread", "Bear Put Spread", "Long Straddle", "Iron Condor"])
    contracts = int(row[2].number_input("Contracts", min_value=1, max_value=20, value=1, step=1))
    try:
        calls, puts = fetch_option_chain(ticker, expiration)
    except Exception as exc:
        st.error(f"Option chain fetch failed: {exc}")
        return
    if calls.empty and puts.empty:
        st.warning("No calls or puts were returned for that expiration.")
        return
    call_center = int((calls["strike"] - snapshot["price"]).abs().idxmin()) if not calls.empty else 0
    put_center = int((puts["strike"] - snapshot["price"]).abs().idxmin()) if not puts.empty else 0
    prices = make_price_grid(snapshot["price"])
    long_call_leg = short_call_leg = long_put_leg = short_put_leg = None
    stock_shares = contracts * 100
    st.subheader("Leg Selection")
    if strategy == "Long Call":
        long_call_leg = choose_contract(calls, "Long Call Leg", call_center)
    elif strategy == "Long Put":
        long_put_leg = choose_contract(puts, "Long Put Leg", put_center)
    elif strategy == "Covered Call":
        stock_shares = int(st.number_input("Stock Shares", min_value=100, value=contracts * 100, step=100))
        contracts = max(1, stock_shares // 100)
        short_call_leg = choose_contract(calls, "Short Call Leg", call_center)
    elif strategy == "Protective Put":
        stock_shares = int(st.number_input("Stock Shares", min_value=100, value=contracts * 100, step=100))
        contracts = max(1, stock_shares // 100)
        long_put_leg = choose_contract(puts, "Long Put Leg", put_center)
    elif strategy == "Bull Call Spread":
        left, right = st.columns(2)
        with left:
            long_call_leg = choose_contract(calls, "Long Lower Strike Call", max(call_center - 1, 0))
        with right:
            short_call_leg = choose_contract(calls, "Short Higher Strike Call", min(call_center + 1, len(calls) - 1))
    elif strategy == "Bull Put Spread":
        left, right = st.columns(2)
        with left:
            short_put_leg = choose_contract(puts, "Short Higher Strike Put", min(put_center + 1, len(puts) - 1))
        with right:
            long_put_leg = choose_contract(puts, "Long Lower Strike Put", max(put_center - 1, 0))
    elif strategy == "Bear Put Spread":
        left, right = st.columns(2)
        with left:
            long_put_leg = choose_contract(puts, "Long Higher Strike Put", min(put_center + 1, len(puts) - 1))
        with right:
            short_put_leg = choose_contract(puts, "Short Lower Strike Put", max(put_center - 1, 0))
    elif strategy == "Long Straddle":
        left, right = st.columns(2)
        with left:
            long_call_leg = choose_contract(calls, "ATM Call", call_center)
        with right:
            long_put_leg = choose_contract(puts, "ATM Put", put_center)
    elif strategy == "Iron Condor":
        left, right = st.columns(2)
        with left:
            short_put_leg = choose_contract(puts, "Short Put", max(put_center - 1, 0))
            long_put_leg = choose_contract(puts, "Long Lower Put", max(put_center - 3, 0))
        with right:
            short_call_leg = choose_contract(calls, "Short Call", min(call_center + 1, len(calls) - 1))
            long_call_leg = choose_contract(calls, "Long Higher Call", min(call_center + 3, len(calls) - 1))
    try:
        curve, metrics = build_strategy_curve(strategy, prices, snapshot["price"], contracts, long_call_leg, short_call_leg, long_put_leg, short_put_leg, stock_shares)
    except Exception as exc:
        st.error(f"Strategy calculation failed: {exc}")
        st.info("Check that your long and short strikes are arranged correctly for the selected spread.")
        return
    st.subheader("Payoff Chart at Expiry")
    st.line_chart(curve.set_index("Underlying Price")["P/L at Expiry"], height=380)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Profit", metrics["max_profit"])
    m2.metric("Max Loss", metrics["max_loss"])
    m3.metric("Breakeven", metrics["breakeven"])
    m4.metric("P/L at Spot", metrics["spot_pl"])
    st.caption(metrics["description"])
    st.subheader("Selected Price Points")
    focus = curve.iloc[[0, len(curve) // 4, len(curve) // 2, len(curve) * 3 // 4, len(curve) - 1]]
    st.dataframe(focus.style.format({"Underlying Price": "{:.2f}", "P/L at Expiry": "{:.2f}"}), use_container_width=True)
    st.subheader("Option Chain Reference")
    left, right = st.columns(2)
    if not calls.empty:
        left.dataframe(calls[["strike", "bid", "ask", "lastPrice", "mid", "impliedVolatility", "openInterest"]].rename(columns={"strike": "Call Strike", "lastPrice": "Last", "impliedVolatility": "IV", "openInterest": "OI"}).style.format({"Call Strike": "{:.2f}", "bid": "{:.2f}", "ask": "{:.2f}", "Last": "{:.2f}", "mid": "{:.2f}", "IV": "{:.2%}"}), use_container_width=True, height=320)
    if not puts.empty:
        right.dataframe(puts[["strike", "bid", "ask", "lastPrice", "mid", "impliedVolatility", "openInterest"]].rename(columns={"strike": "Put Strike", "lastPrice": "Last", "impliedVolatility": "IV", "openInterest": "OI"}).style.format({"Put Strike": "{:.2f}", "bid": "{:.2f}", "ask": "{:.2f}", "Last": "{:.2f}", "mid": "{:.2f}", "IV": "{:.2%}"}), use_container_width=True, height=320)


def main():
    st.set_page_config(page_title="Local Portfolio Tracker", layout="wide")
    conn = get_conn()
    menu = {"Dashboard": page_dashboard, "Transactions": page_transactions, "Watchlist": page_targets, "Events": page_events, "Options": lambda _conn: page_options()}
    choice = st.sidebar.radio("Menu", list(menu.keys()))
    st.sidebar.caption("Data is stored in data/portfolio.db")
    menu[choice](conn)


if __name__ == "__main__":
    main()
