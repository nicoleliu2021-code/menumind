import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px

st.set_page_config(page_title="MenuMind — AI Menu Optimizer", page_icon="🍽️", layout="wide")
st.title("🍽️ MenuMind — AI Menu & Pricing Optimizer (MVP)")

st.caption("Upload a simple sales CSV to forecast demand and get price recommendations. No installs needed.")

# ---- Sidebar: sample data & help ----
with st.sidebar:
    st.header("📄 Data format")
    st.write("CSV with columns:")
    st.code("date,item_name,price,cost,qty_sold", language="text")
    st.write("Example row:")
    st.code("2025-09-01,Pasta,12,6,90", language="text")
    demo = st.checkbox("Use demo data", value=True)
    st.markdown("---")
    st.write("Tip: Start with 6–12 weeks of daily data per item for better forecasts.")

# ---- Load data ----
def load_demo():
    return pd.read_csv("data/sample_sales.csv", parse_dates=["date"])

uploaded = st.file_uploader("Upload your sales CSV", type=["csv"])

if demo and not uploaded:
    df = load_demo()
elif uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    st.info("Upload a CSV or toggle **Use demo data** in the sidebar.")
    st.stop()

# Basic checks
required_cols = {"date","item_name","price","cost","qty_sold"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"CSV must include columns: {sorted(required_cols)}")
    st.stop()

# Clean
df = df.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["item_name","date"])
df["revenue"] = df["price"] * df["qty_sold"]
df["profit"]  = (df["price"] - df["cost"]) * df["qty_sold"]

# ---- Top KPIs ----
total_rev = df["revenue"].sum()
total_profit = df["profit"].sum()
avg_margin_pct = np.where(df["revenue"].sum()>0, df["profit"].sum()/df["revenue"].sum()*100, 0)

k1, k2, k3 = st.columns(3)
k1.metric("Total Revenue (range)", f"${total_rev:,.0f}")
k2.metric("Total Profit (range)", f"${total_profit:,.0f}")
k3.metric("Avg Margin %", f"{avg_margin_pct:.1f}%")

# ---- Item selection ----
items = sorted(df["item_name"].unique())
item = st.selectbox("Choose a menu item", items)

sub = df[df["item_name"] == item].copy()
base_price = sub["price"].mean()
base_cost  = sub["cost"].mean()

# ---- Forecasting (daily) ----
daily = (
    sub.groupby("date", as_index=False)
       .agg(qty_sold=("qty_sold","sum"))
)

def forecast_next_7(daily_df):
    # Need at least ~14 points; else fallback to moving average
    if len(daily_df) < 14:
        mean_qty = max(1, daily_df["qty_sold"].mean())
        last_date = daily_df["date"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1,8)]
        return pd.DataFrame({"date": future_dates, "forecast_qty": [mean_qty]*7, "method":"avg"})
    # Use Holt-Winters (additive trend, no seasonality for simplicity)
    s = daily_df.set_index("date")["qty_sold"].asfreq("D").fillna(0)
    model = ExponentialSmoothing(s, trend="add", seasonal=None)
    fit = model.fit(optimized=True)
    fcast = fit.forecast(7)
    out = fcast.reset_index()
    out.columns = ["date","forecast_qty"]
    out["method"] = "holt-winters"
    return out

f7 = forecast_next_7(daily)

# ---- Simple price recommendation (elasticity sweep) ----
def recommend_price(base_p, cost, forecast_qty, elasticity=-1.1):
    """
    elasticity ~ -1.1 means: +1% price -> -1.1% demand
    We sweep ±10% around the current price and pick max profit.
    """
    if base_p <= cost:
        base_p = cost + 0.01
    candidate_prices = [round(base_p*(1 + k/100), 2) for k in range(-10, 11, 2)]
    results = []
    base_qty = max(1, forecast_qty)
    for p in candidate_prices:
        demand_factor = (p/base_p)**elasticity
        q = max(0, base_qty * demand_factor)
        profit = (p - cost) * q
        results.append((p, q, profit))
    best = max(results, key=lambda x: x[2])
    return {
        "suggested_price": best[0],
        "predicted_qty": best[1],
        "predicted_profit": best[2],
        "grid": results
    }

# Use the average of next 7 days as “near-term demand”
avg_next_qty = f7["forecast_qty"].mean() if len(f7) else max(1, sub["qty_sold"].tail(7).mean())
rec = recommend_price(base_price, base_cost, avg_next_qty)

# ---- Show recommendations ----
c1, c2, c3 = st.columns(3)
c1.metric("Current Avg Price", f"${base_price:.2f}")
c2.metric("Suggested Price", f"${rec['suggested_price']:.2f}")
c3.metric("Predicted Profit (index)", f"{rec['predicted_profit']:.2f}")

st.caption("Note: Profit is an index for comparison (per-day basis with forecasted quantity).")

# ---- Charts ----
left, right = st.columns([2,1])

with left:
    hist = px.bar(sub, x="date", y="qty_sold", title=f"Historical Daily Sales — {item}")
    st.plotly_chart(hist, use_container_width=True)

    f7_chart = px.line(f7, x="date", y="forecast_qty", title="7-Day Forecast (qty)")
    st.plotly_chart(f7_chart, use_container_width=True)

with right:
    grid_df = pd.DataFrame(rec["grid"], columns=["price","qty","profit_index"])
    price_chart = px.line(grid_df, x="price", y="profit_index", markers=True,
                          title="Profit vs Price (elasticity sweep)")
    st.plotly_chart(price_chart, use_container_width=True)

# ---- Table: top opportunities today (simple heuristic) ----
st.subheader("📈 Simple Opportunity Scan (today)")
today = df["date"].max()
today_items = (
    df[df["date"] >= today - pd.Timedelta(days=14)]
    .groupby("item_name", as_index=False)
    .agg(avg_price=("price","mean"),
         avg_cost=("cost","mean"),
         avg_qty=("qty_sold","mean"),
         rev=("revenue","sum"),
         prof=("profit","sum"))
)
def quick_rec(row):
    r = recommend_price(row["avg_price"], row["avg_cost"], max(1,row["avg_qty"]))
    lift = r["predicted_profit"] - (row["avg_price"]-row["avg_cost"])*row["avg_qty"]
    return pd.Series({
        "suggested_price": r["suggested_price"],
        "profit_lift_index": round(lift,2)
    })
scan = today_items.join(today_items.apply(quick_rec, axis=1))
scan = scan.sort_values("profit_lift_index", ascending=False).head(10)
st.dataframe(scan[["item_name","avg_price","avg_cost","suggested_price","profit_lift_index"]], use_container_width=True)

st.success("Done. Use this as a weekly ‘batch’ recommendation. For live pilots, export this table and discuss changes with the owner.")
