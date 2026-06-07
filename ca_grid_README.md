# California Electric Load Forecasting & Grid Demand Planning Tool
### SCE Service Territory · SARIMA Forecasting · EV & Solar Scenarios · Grid Stress Analysis

**Built by Sherriff Abdul-Hamid**  
Development economist and data scientist — USAID · UNDP · UKAID · Obama Foundation Leaders Award (Top 1.3%)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ca-grid-demand-forecast.streamlit.app)

---

## The Operational Problem

Every year, SCE makes **multi-billion dollar capacity investment decisions** based on load forecasts. A 5% error in peak demand forecasting translates to roughly **$500M–$800M in avoidable capital expenditure** or, in the opposite direction, reliability failures affecting millions of customers during summer peak periods.

The forecasting challenge has grown significantly harder:

- **EV adoption** is adding unpredictable evening load spikes across the Inland Empire and LA Basin
- **Rooftop solar** is reducing mid-day net load while sharpening the "duck curve" — creating steep late-afternoon ramp requirements
- **Climate change** is extending summer peak seasons and intensifying extreme heat events in Coachella Valley
- **Industrial demand response** is creating flexible but uncertain load patterns

This tool gives SCE planners a structured, scenario-based framework for answering the questions that drive capital decisions: *When will peak demand exceed current capacity? Which zones face the earliest grid stress? At what EV adoption rate does the grid need new investment?*

---

## What This Tool Does

| Tab | Content | SCE Planning Use Case |
|---|---|---|
| **Historical Load Analysis** | 3 years of daily load data across 4 zones and 3 sectors. Seasonal decomposition, temperature correlation, day-of-week effects. | Baseline characterisation for IRP and annual load forecasting reports |
| **Demand Forecasting** | SARIMA(1,1,1)(1,1,1,12) forecast 12–24 months ahead with 90% confidence intervals and holdout validation. MAE, RMSE, MAPE reported. | Short-range capacity planning, rate case testimony, CPUC filings |
| **EV & Solar Scenarios** | Interactive sliders for EV adoption (0–40%), rooftop solar (0–30%), commercial growth (±10%), and industrial demand response (0–15%). Three preset scenarios (Conservative / Base Case / Aggressive) plus custom configuration. | IRP scenario analysis, DER integration planning, distribution system planning |
| **Executive Planning Dashboard** | Peak demand vs capacity thresholds, grid stress heatmap, reserve margin by scenario, investment trigger analysis with specific month-by-month recommendations. | Executive briefings, Board presentations, CPUC IRP filings |

---

## Job Description Alignment

| SCE Requirement | How This App Addresses It |
|---|---|
| **Time-series forecasting** (JD 6279) | SARIMA model with seasonal decomposition, holdout validation, and confidence intervals — industry-standard utility forecasting methodology |
| **Scenario modelling** (JD 4818) | Four-parameter interactive scenario builder producing Conservative / Base Case / Aggressive projections side by side |
| **Data visualisation for executive audiences** (both JDs) | Executive Planning Dashboard tab built around KPI cards, a grid stress heatmap, and investment trigger recommendations |
| **EV and DER integration analysis** (JD 4818) | Explicit modelling of EV load addition, solar net load reduction, and industrial demand response as scenario parameters |
| **Zone-level load disaggregation** (JD 6279) | Four-zone analysis: LA Basin, Inland Empire, Orange County, Coachella Valley — matching SCE's published service territory structure |
| **Grid reliability metrics** (both JDs) | Reserve margin, days above 95% capacity, and investment trigger logic tied to the 15% reserve margin target |
| **Temperature-sensitive demand modelling** | AC load coefficient for Southern California climate with Coachella Valley extreme heat adjustment |
| **Python / ML production skills** (JD 6279) | End-to-end pipeline: statsmodels SARIMAX, scikit-learn metrics, Plotly visualisation, Streamlit deployment |

---

## Model Specification

### SARIMA(1,1,1)(1,1,1,12)

```
Non-seasonal:  AR(1), first difference, MA(1)
Seasonal:      SAR(1), seasonal difference, SMA(1), period = 12 months
Holdout:       Final 6 months (Jun–Dec 2023)
Training:      Jan 2021 – May 2023 (30 months)
MAPE:          ~3.6%  (within ±5% industry benchmark)
```

The 12-period seasonal structure captures Southern California's strong summer AC peak and the relatively flat winter demand pattern — a more pronounced seasonal ratio than most US utility territories.

### Scenario Engine

```
Net load = Base forecast
         + (EV rate × 0.60 × base)          # EV adds charging demand
         - (Solar rate × 0.45 × base)        # Solar reduces net load
         + (Commercial growth × 0.42 × base) # Commercial is 42% of territory load
         - (Industrial shift × 0.18 × 0.40 × base)  # DR saves 40% of shifted load
```

Coefficients are calibrated to CAISO DER integration studies and SCE IRP published assumptions.

---

## Zone Profiles

| Zone | Share of Peak Load | Key Driver | Summer Peak Characteristic |
|---|---|---|---|
| LA Basin | ~45% | Commercial density | Strong AC + commercial workday peak |
| Inland Empire | ~25% | Residential + logistics | Fastest EV adoption growth |
| Orange County | ~20% | Residential + commercial | High rooftop solar potential |
| Coachella Valley | ~10% | Extreme summer heat | Highest temperature coefficient; 108°F+ July peak |

---

## Repository Structure

```
├── app.py              # Main Streamlit application (4 tabs, 1,180 lines)
├── requirements.txt    # Runtime dependencies
└── README.md           # This file
```

---

## Run Locally

```bash
git clone https://github.com/S-ABDUL-AI/ca-grid-demand-forecast.git
cd ca-grid-demand-forecast
pip install -r requirements.txt
streamlit run app.py
```

**Dependencies:** `streamlit` · `pandas` · `numpy` · `plotly` · `statsmodels` · `scikit-learn` · `scipy`

---

## Data Note

All load profiles are **calibrated synthetic data** generated to approximate SCE service territory conditions:

- Seasonal patterns consistent with CAISO published annual load shapes
- Temperature coefficients calibrated to Southern California climate normals (NOAA)
- Zone shares consistent with SCE Integrated Resource Plan published data
- Growth rates (1.8% YoY) consistent with CPUC demand forecasts

**Not for operational use.** Connect to SCE's OATI or EMS data feeds for production deployment.

---

## Methodological Note: From Development Economics to Utility Forecasting

The core methodological challenge in utility load forecasting — separating trend, seasonality, and irregular components in a short time series with strong structural breaks — is identical to the challenge I faced modelling oil price transmission to Ghana's public debt in my MSc thesis.

Both problems require:

| Econometric challenge | Thesis application | Load forecasting application |
|---|---|---|
| Seasonal decomposition | Annual commodity price cycles | Monthly AC/heating load cycles |
| Structural breaks | HIPC debt relief, oil production start | COVID-19 demand shock, EV inflection |
| Forecast confidence intervals | IRF confidence bands (VECM) | SARIMA 90% prediction intervals |
| Scenario analysis | Oil price shock scenarios | EV adoption / solar penetration scenarios |
| Small-sample inference | n=37 annual observations | n=36 monthly observations for training |

The SARIMA framework used here is the direct descendant of the VAR/VECM methodology applied in the thesis — both are stationary time-series models that capture autoregressive dynamics and seasonal structure. The transition from macroeconomic forecasting to utility load forecasting requires domain knowledge, not a change in analytical framework.

---

## GitHub Topics

```
streamlit python load-forecasting time-series sarima energy-demand ev-adoption
solar-integration california-energy utility-planning plotly scikit-learn
```

---

## About the Author

**Sherriff Abdul-Hamid** is a development economist and data scientist with 10+ years of experience in quantitative research, forecasting, and data-driven decision support for infrastructure, energy, and public policy programs.

- MSc Economics (Econometrics), KNUST — thesis: *Oil Price Shocks and Public Debt: Empirical Evidence from Ghana*
- Harvard Business School · Senior Executive Program
- Former Founder & CEO, Poverty 360 — 25,000+ beneficiaries
- Directed $200M+ in resource allocation decisions for USAID, UNDP, UKAID
- **Obama Foundation Leaders Award** — Top 1.3% globally, 2023
- **Mandela Washington Fellow** — Top 0.3%, U.S. Department of State, 2018

**Connect:**  
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) ·
[GitHub](https://github.com/S-ABDUL-AI) ·
[Portfolio](https://share.streamlit.io/user/s-abdul-ai)

---

## Related SCE Portfolio Apps

| App | Description |
|---|---|
| [Grid Asset Risk & Wildfire Dashboard](https://share.streamlit.io/user/s-abdul-ai) | Random Forest wildfire risk and asset failure prediction across SCE territory |
| [Grid Investment Prioritization Engine](https://share.streamlit.io/user/s-abdul-ai) | Benefit-cost optimization and portfolio scenario builder for grid capital allocation |
| [Oil Shock Transmission Dashboard](https://share.streamlit.io/user/s-abdul-ai) | Interactive VAR/IRF/FEVD econometrics from MSc thesis — oil price shocks and public debt |
