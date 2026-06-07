# ⚡ California Electric Load Forecasting & Grid Demand Planning Tool

**Time series demand forecasting and grid scenario planning for electric utility operations**  
Built by Sherriff Abdul-Hamid | poverty360.org

---

## Business Problem

Load forecasting error costs U.S. utilities an estimated $2–5B annually through inefficient
capacity procurement, unnecessary reserve margins, and preventable grid stress events.
For major electric utilities — serving millions of customers across fast-growing
utility territories — accurate demand forecasting is not optional.
It is the foundation of every capital investment, rate design, and reliability decision.

This tool addresses three operational questions utility planning teams face daily:

1. What will electricity demand look like over the next 12–24 months by zone?
2. How does accelerating EV adoption and rooftop solar penetration change peak demand?
3. At what load level does the grid operator need to trigger additional capacity investments?

---

## Features

### Tab 1 — Historical Load Analysis
- Daily and weekly load profiles across four SoCal zones: Inland Empire, LA Basin,
  Orange County, and Coachella Valley
- Temperature-load correlation analysis
- Seasonal decomposition showing trend, seasonality, and residuals
- Peak demand statistics and load factor analysis by zone

### Tab 2 — Demand Forecasting
- SARIMA time series model with automatic parameter selection
- 12–24 month forecast horizon (user-adjustable via slider)
- 80% and 95% confidence intervals
- Model performance metrics: MAE, RMSE, MAPE on holdout period
- Forecast vs. actuals comparison chart

### Tab 3 — EV Adoption & Solar Penetration Scenarios
- Interactive sliders: EV adoption rate (0–40%), rooftop solar penetration (0–30%),
  commercial demand growth (±10%), industrial load shift (0–15%)
- Three scenario comparison: Conservative, Base Case, Aggressive
- Load shape impact visualisation showing how each scenario changes peak timing and magnitude
- Grid stress indicator showing days above 95% capacity threshold per scenario

### Tab 4 — Executive Planning Dashboard
- Monthly peak demand forecast with capacity reserve margins
- Investment trigger analysis: at what load level does additional capacity become necessary
- Grid stress heatmap by month and zone
- One-page executive KPI summary suitable for regulatory filings

---

## Methodology
Forecasting Model: SARIMA(p,d,q)(P,D,Q,s)
Seasonal Period:   s = 12 (monthly), s = 52 (weekly)
EV Load Impact:    Additional kWh = EV_count × avg_annual_mileage × kWh_per_mile / 8760
Solar Offset:      Load_net = Load_gross - (solar_capacity × capacity_factor × penetration_rate)
Reserve Margin:    (Installed_capacity - Peak_demand) / Peak_demand × 100

---

## Applications in Utility Grid Planning

| Use Case | How This Tool Addresses It |
|----------|---------------------------|
| Demand forecasting | SARIMA model produces 12-24 month load forecasts with confidence intervals |
| EV integration planning | Scenario builder quantifies charging demand impact on peak load by zone |
| Distributed energy resource planning | Solar penetration slider models net load reduction under different DER scenarios |
| Capacity investment triggering | Investment trigger analysis identifies load thresholds requiring new capacity |
| Reserve margin management | Monthly reserve margin forecasts support procurement planning |
| Regulatory integrated resource planning | Executive dashboard output suitable for IRP and CPUC filing support |

---

## Data

All load data is simulated using realistic parameters for Southern California:
- Temperature-correlated load curves (summer AC peaks, mild winters)
- Zone-level disaggregation calibrated to published utility service territory data
- EV adoption projections aligned to California ZEV mandate trajectory
- Solar penetration based on California NEM programme growth rates

In production deployment this model would ingest:
- Utility SCADA real-time load feeds
- CAISO day-ahead and real-time market data
- NOAA temperature and weather forecasts
- CPUC-approved EV and DER adoption projections

---

## Run Locally

```bash
git clone https://github.com/S-ABDUL-AI/ca-grid-demand-forecast
cd ca-grid-demand-forecast
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Sherriff Abdul-Hamid**
Development Economist · Data Scientist · Public Infrastructure Analytics
poverty360.org | linkedin.com/in/abdul-hamid-sherriff-08583354
