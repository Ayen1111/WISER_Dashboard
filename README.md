#  WISER — Balanced Logistics Intelligence Dashboard  
*AI-powered delivery performance tracking, delay prediction & smart carrier optimization*

🔗 Live App: https://wiserdashboard-kxdzrrgapvwnkj7nro72sb.streamlit.app/  
 Repository: WISER_Dashboard

---

##  Project Overview

WISER is an end-to-end AI solution designed to help logistics companies:

 1) Predict delayed deliveries before they happen  
 2) Identify best carrier for each lane & priority  
 3) Reduce logistics cost, CO₂ emissions & SLA penalties  
 4) Provide real-time insights into supply chain performance  

This dashboard supports Data-Driven **Predict → Rebalance → Save Cost & CO₂** decisions.

---

##  Key Features

| Feature | Description |
|--------|-------------|
|  Data Visualization | Performance by carrier, cost vs distance trends |
|  SLA Risk Prediction | ML model predicts delivery delays using Random Forest |
|  Smart Carrier Recommendation | Suggests carrier switch to improve on-time %
|  Sustainability Tracking | Calculates CO₂ footprint per order using distance |
|  Multi-dataset Support | Works with sample or real enterprise data |
|  Auto Data Mapping | Column-name independent processing |

---

##  Machine Learning

 Classification model for **SLA Breach (Delay vs No Delay)**  
 Regression model for **expected delay hours**

**Metrics (sample dataset):**  
- Accuracy: 100% *(limited sample — expected lower on real data)*  
- Zero False Negatives → No delayed order missed

 Shows business-first ML capability

---

##  Dataset Structure

| File | Use |
|------|-----|
| orders.csv | Order metadata (origin, destination, priority, value) |
| delivery_performance.csv | Promised & actual delivery, carrier, cost |
| routes_distance.csv | Route KM — used for CO₂ calculation |
| vehicle_fleet.csv | Carrier capability data |
| warehouse_inventory.csv | Warehouse mapping |
| customer_feedback.csv | Ratings + issues |
| cost_breakdown.csv | Cost splits |

 Order_ID joins all datasets  
 Sample & full datasets included

---

##  Tech Stack

| Layer | Technology |
|------|------------|
| Frontend UI | Streamlit |
| ML Models | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visuals | Plotly, Altair |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

##  How to Run Locally

```bash
git clone https://github.com/Ayen1111/WISER_Dashboard.git
cd WISER_Dashboard
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


