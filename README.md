
# WISER — Balanced Logistics Intelligence Dashboard

Interactive Streamlit app that predicts delivery delays, explains drivers, and recommends rebalancing actions that improve on-time rate, lower cost, and reduce CO₂.

## Quickstart (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Place your CSVs into `./data/` with exact names:
orders.csv, delivery_performance.csv, routes_distance.csv, vehicle_fleet.csv, warehouse_inventory.csv, customer_feedback.csv, cost_breakdown.csv
