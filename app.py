# app.py — WISER: Balanced Logistics Intelligence Dashboard
# Works with your headers like: Order_ID, Origin, Destination, Carrier,
# Promised_..., Actual_..., Delivery_Cost_INR, etc.

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="WISER — Balanced Logistics Intelligence", layout="wide")

# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    """Load CSV defensively (no parse_dates; we infer later)."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return None

def safe_rate(num, den):
    return float(num) / float(den) if den else 0.0

def normcols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def find_col(df: pd.DataFrame, exact=None, contains=None):
    """
    Find a column name in df:
    - exact: list of exact names (case-insensitive)
    - contains: list of substrings that must all be present (case-insensitive)
    Returns the *actual* column name or None.
    """
    cols = list(df.columns)
    if exact:
        for want in exact:
            for col in cols:
                if col.lower() == str(want).lower():
                    return col
    if contains:
        for col in cols:
            cl = col.lower()
            if all(sub.lower() in cl for sub in contains):
                return col
    return None

# ---------------------- Files & Sidebar ----------------------
import os

default_data_dir = "data_sample" if os.getenv("STREAMLIT_RUNTIME") else "data"
DATA_DIR = st.sidebar.text_input("Data folder", value="data")
files = {
    "orders": os.path.join(DATA_DIR, "orders.csv"),
    "delivery": os.path.join(DATA_DIR, "delivery_performance.csv"),
    "routes": os.path.join(DATA_DIR, "routes_distance.csv"),
    "fleet": os.path.join(DATA_DIR, "vehicle_fleet.csv"),
    "inventory": os.path.join(DATA_DIR, "warehouse_inventory.csv"),
    "feedback": os.path.join(DATA_DIR, "customer_feedback.csv"),
    "costs": os.path.join(DATA_DIR, "cost_breakdown.csv"),
}
with st.sidebar.expander("Expected CSVs", expanded=False):
    st.write(pd.DataFrame({"logical_name": list(files.keys()), "filename": list(files.values())}))

orders = load_csv(files["orders"])
delivery = load_csv(files["delivery"])
routes = load_csv(files["routes"])
fleet = load_csv(files["fleet"])
inventory = load_csv(files["inventory"])
feedback = load_csv(files["feedback"])
costs = load_csv(files["costs"])

def shape(df): return "missing" if df is None else f"{df.shape[0]} x {df.shape[1]}"
st.sidebar.markdown("### Data Health")
st.sidebar.write({
    "orders": shape(orders),
    "delivery_performance": shape(delivery),
    "routes_distance": shape(routes),
    "vehicle_fleet": shape(fleet),
    "warehouse_inventory": shape(inventory),
    "customer_feedback": shape(feedback),
    "cost_breakdown": shape(costs),
})

# ---------------------- Merge & Feature Engineering with Auto-Mapping ----------------------
def prep_dataset():
    if orders is None:
        return None, ["orders.csv not found."]

    issues = []
    base = normcols(orders)

    # Key
    oid = find_col(base, exact=["order_id", "Order_ID", "Order Id", "OrderID"])
    if not oid:
        issues.append("orders.csv is missing an order id column (e.g., Order_ID).")
        return None, issues
    base = base.rename(columns={oid: "order_id"})

    # Core fields in orders
    oc  = find_col(base, exact=["origin_city", "Origin", "From", "Source"])
    dc  = find_col(base, exact=["destination_city", "Destination", "To"])
    pr  = find_col(base, exact=["priority", "Priority"])
    odt = find_col(base, exact=["order_date", "Order_Date", "Order Date"])

    rename_map = {}
    if oc:  rename_map[oc]  = "origin_city"
    if dc:  rename_map[dc]  = "destination_city"
    if pr:  rename_map[pr]  = "priority"
    if odt: rename_map[odt] = "order_date"
    base = base.rename(columns=rename_map)
    if "order_date" in base.columns:
        base["order_date"] = pd.to_datetime(base["order_date"], errors="coerce")

    df = base

    # Delivery merge: carrier, promised/actual, delivery cost
    if delivery is not None:
        deliv = normcols(delivery)
        doid = find_col(deliv, exact=["order_id", "Order_ID", "Order Id", "OrderID"])
        if doid:
            deliv = deliv.rename(columns={doid: "order_id"})

            carr = find_col(deliv, exact=["carrier", "Carrier"])
            if carr: deliv = deliv.rename(columns={carr: "carrier"})

            pcol = find_col(deliv, exact=["promised_delivery"],
                            contains=["promis", "deliver"]) or find_col(deliv, contains=["promis", "deliver", "date"])
            acol = find_col(deliv, exact=["actual_delivery"],
                            contains=["actual", "deliver"]) or find_col(deliv, contains=["actual", "deliver", "date"])

            if pcol: deliv[pcol] = pd.to_datetime(deliv[pcol], errors="coerce")
            if acol: deliv[acol] = pd.to_datetime(deliv[acol], errors="coerce")

            # delivery cost (e.g., Delivery_Cost_INR)
            dcost = find_col(deliv, exact=["delivery_cost"], contains=["deliver", "cost"])
            if dcost: deliv = deliv.rename(columns={dcost: "delivery_cost"})

            df = df.merge(deliv, on="order_id", how="left", suffixes=("", "_del"))

            if pcol and acol:
                df["delay_hours"] = (df[acol] - df[pcol]).dt.total_seconds() / 3600.0
                df["delay_hours"] = df["delay_hours"].fillna(0)
                df["sla_breach"] = (df["delay_hours"] > 0).astype(int)
            else:
                df["delay_hours"] = np.nan
                df["sla_breach"] = np.nan
        else:
            issues.append("delivery_performance.csv does not have an order id column (e.g., Order_ID).")

    # Routes: distance_km
    if routes is not None:
        rtes = normcols(routes)
        roid = find_col(rtes, exact=["order_id", "Order_ID", "Order Id", "OrderID"])
        if roid:
            rtes = rtes.rename(columns={roid: "order_id"})
            dist = find_col(rtes, exact=["distance_km", "Distance_km", "distance", "Distance"], contains=["distance"])
            if dist: rtes = rtes.rename(columns={dist: "distance_km"})
            df = df.merge(rtes, on="order_id", how="left", suffixes=("", "_rt"))
        else:
            issues.append("routes_distance.csv missing order id (e.g., Order_ID).")

    # Costs (optional if not already in delivery)
    if costs is not None:
        cst = normcols(costs)
        coid = find_col(cst, exact=["order_id", "Order_ID", "Order Id", "OrderID"])
        if coid:
            cst = cst.rename(columns={coid: "order_id"})
            dcost = find_col(cst, exact=["delivery_cost"], contains=["deliver", "cost"])
            if dcost: cst = cst.rename(columns={dcost: "delivery_cost"})
            df = df.merge(cst, on="order_id", how="left", suffixes=("", "_cost"))

    # CO2 proxy
    if "distance_km" in df.columns and "co2_kg" not in df.columns:
        try:
            df["co2_kg"] = pd.to_numeric(df["distance_km"], errors="coerce") * 0.18
        except Exception:
            df["co2_kg"] = np.nan

    # Priority encoding
    if "priority" in df.columns:
        pr_map = {"Express": 3, "Standard": 2, "Economy": 1}
        df["priority_code"] = df["priority"].map(pr_map).fillna(0)

    return df, issues

df, prep_issues = prep_dataset()

st.title("WISER — Balanced Logistics Intelligence Dashboard")
st.caption("Predict → Rebalance → Save Cost & CO₂")

if prep_issues:
    for msg in prep_issues:
        st.warning(msg)

if df is None:
    st.warning("Please provide at least orders.csv and delivery_performance.csv (with Order_ID) to proceed.")
    st.stop()

# ---------------------- Filters ----------------------
cols = st.columns(4)
with cols[0]:
    city_from = st.multiselect("Origin City", sorted(df["origin_city"].dropna().unique()) if "origin_city" in df.columns else [])
with cols[1]:
    city_to = st.multiselect("Destination City", sorted(df["destination_city"].dropna().unique()) if "destination_city" in df.columns else [])
with cols[2]:
    priorities = st.multiselect("Priority", sorted(df["priority"].dropna().unique()) if "priority" in df.columns else [])
with cols[3]:
    carriers = st.multiselect("Carrier", sorted(df["carrier"].dropna().unique()) if "carrier" in df.columns else [])

f = df.copy()
if city_from:
    f = f[f.get("origin_city").isin(city_from)]
if city_to:
    f = f[f.get("destination_city").isin(city_to)]
if priorities:
    f = f[f.get("priority").isin(priorities)]
if carriers:
    f = f[f.get("carrier").isin(carriers)]

# ---------------------- KPI Snapshot ----------------------
k1, k2, k3, k4, k5 = st.columns(5)
total_orders = len(f)
on_time = int((f["sla_breach"] == 0).sum()) if "sla_breach" in f.columns else 0
avg_delay = float(f["delay_hours"].clip(lower=0).mean()) if "delay_hours" in f.columns else 0.0

# Prefer delivery_cost if present anywhere
cost_col_candidates = [c for c in f.columns if "delivery_cost" in c.lower() or c.lower() == "cost" or "total_cost" in c.lower()]
cost_col = cost_col_candidates[0] if cost_col_candidates else None
avg_cost = float(f[cost_col].mean()) if cost_col else float("nan")

co2 = float(f["co2_kg"].sum()) if "co2_kg" in f.columns else float("nan")

k1.metric("Orders", total_orders)
k2.metric("On-Time Rate", f"{safe_rate(on_time, total_orders):.0%}")
k3.metric("Avg Delay (hrs)", f"{avg_delay:.2f}")
k4.metric("Avg Cost / Order", f"{avg_cost:.2f}" if not np.isnan(avg_cost) else "—")
k5.metric("Total CO₂ (kg)", f"{co2:,.0f}" if not np.isnan(co2) else "—")

# ---------------------- Visualizations ----------------------
st.markdown("### Performance & Cost Visuals")
v1, v2 = st.columns(2)

if "carrier" in f.columns and "sla_breach" in f.columns:
    perf_by_carrier = f.groupby("carrier")["sla_breach"].apply(lambda s: (s == 0).mean()).reset_index(name="on_time_rate")
    fig = px.bar(perf_by_carrier, x="carrier", y="on_time_rate", title="On-Time Rate by Carrier", text_auto=".0%")
    fig.update_layout(yaxis_tickformat=".0%")
    v1.plotly_chart(fig, use_container_width=True)

if "distance_km" in f.columns and cost_col:
    fig2 = px.scatter(
        f, x="distance_km", y=cost_col, color=f["priority"] if "priority" in f.columns else None,
        hover_data=["carrier", "order_id"] if "carrier" in f.columns else ["order_id"],
        trendline="ols", title=f"Cost vs Distance (using '{cost_col}')"
    )
    v2.plotly_chart(fig2, use_container_width=True)

v3, v4 = st.columns(2)
if all(c in f.columns for c in ["origin_city", "destination_city", "delay_hours"]):
    lane_delay = f.groupby(["origin_city", "destination_city"])["delay_hours"].mean().reset_index()
    fig3 = px.density_heatmap(lane_delay, x="origin_city", y="destination_city", z="delay_hours", title="Average Delay (hrs) by Lane")
    v3.plotly_chart(fig3, use_container_width=True)

if "priority" in f.columns and cost_col and "sla_breach" in f.columns:
    tmp = f.assign(on_time=(f["sla_breach"] == 0).astype(int)).groupby("priority").agg(
        avg_cost=(cost_col, "mean"), on_time=("on_time", "mean")).reset_index()
    fig4 = px.line(tmp, x="priority", y=["avg_cost", "on_time"], title="Priority vs Avg Cost & On-Time Rate")
    v4.plotly_chart(fig4, use_container_width=True)

# ---------------------- Predictive Models ----------------------
st.markdown("## Predictive Delivery Risk & Delay Size")

# Build numeric feature set excluding obvious label/meta columns
exclude_cols = {
    "order_id", "order_date", "promised_delivery", "actual_delivery",
    "carrier", "origin_city", "destination_city", "priority",
    "sla_breach", "delay_hours"
}
feature_candidates = [c for c in f.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(f[c])]

# Classification
if "sla_breach" in f.columns and f["sla_breach"].notna().sum() > 20 and len(feature_candidates) >= 3:
    X = f[feature_candidates].fillna(0)
    y = f["sla_breach"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    st.text("Classification report (SLA Breach):")
    st.text(classification_report(y_test, clf.predict(X_test)))
    f["breach_risk"] = clf.predict_proba(f[feature_candidates].fillna(0))[:, 1]
else:
    st.info("Need more labeled data (sla_breach) and numeric features for classification.")

# Regression
if "delay_hours" in f.columns and f["delay_hours"].notna().sum() > 20 and len(feature_candidates) >= 3:
    Xr = f[feature_candidates].fillna(0)
    yr = f["delay_hours"].clip(lower=0)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)
    rgr = RandomForestRegressor(n_estimators=200, random_state=42)
    rgr.fit(Xr_train, yr_train)
    st.write(f"MAE (Delay Hours): {mean_absolute_error(yr_test, rgr.predict(Xr_test)):.2f}")
    f["pred_delay_hours"] = rgr.predict(Xr)
else:
    st.info("Need more labeled data for delay regression.")

# ---------------------- Rebalancing Suggestions (Balanced) ----------------------
st.markdown("## Rebalancing Suggestions (Balanced: On-time + Cost + CO₂)")

recs = []
if "carrier" in df.columns and "sla_breach" in df.columns:
    bench = df.copy()
    bench["on_time"] = (bench["sla_breach"] == 0).astype(int)
    lane_cols = [c for c in ["origin_city", "destination_city", "priority"] if c in bench.columns and bench[c].notna().any()]
    if lane_cols:
        grp = bench.groupby(lane_cols + ["carrier"], dropna=False)["on_time"].mean().reset_index()
        best = grp.sort_values(["on_time"], ascending=False).groupby(lane_cols, dropna=False).head(1)

        for _, row in f.iterrows():
            key = tuple(row.get(c) for c in lane_cols)
            if not lane_cols:
                continue
            sub = best.copy()
            for i, c in enumerate(lane_cols):
                sub = sub[sub[c] == key[i]]
            if not sub.empty:
                best_carrier = sub["carrier"].iloc[0]
                best_rate = sub["on_time"].iloc[0]
                if pd.notna(row.get("carrier")) and best_carrier != row.get("carrier"):
                    recs.append({
                        "order_id": row.get("order_id"),
                        "action": "Switch Carrier",
                        "suggestion": f"Switch to {best_carrier} for this lane/priority (historical on-time: {best_rate:.0%})."
                    })

recs_df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["order_id", "action", "suggestion"])
st.dataframe(recs_df, use_container_width=True)

st.caption("Tip: Put your CSVs into the /data folder. Adjust 'Data folder' in the sidebar if needed.")


