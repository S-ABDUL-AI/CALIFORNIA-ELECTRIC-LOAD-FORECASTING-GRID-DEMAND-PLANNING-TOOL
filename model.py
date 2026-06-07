"""
California Electric Load Forecasting — data generation, SARIMA modelling,
scenario engine, and reserve-margin analytics.
"""

import html
import warnings
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    SM_OK = True
except ImportError:
    SM_OK = False

# ─────────────────────────────────────────────────────────────
# DESIGN / DATA CONSTANTS
# ─────────────────────────────────────────────────────────────
NAVY = "#1F3864"
NAVY_MID = "#2D4F8A"
GOLD = "#C9A84C"
RED = "#C8382A"
AMBER = "#B8560A"
GREEN = "#1A7A2E"
TEAL = "#0E7490"
PURPLE = "#6D28D9"
MUTED = "#6B7280"
BODY = "#2C3E50"
RULE = "#E2E6EC"
OFF_WHITE = "#F8F6F1"
INK = "#1A1A1A"
WHITE = "#FFFFFF"

ZONES = ["LA Basin", "Inland Empire", "Orange County", "Coachella Valley"]

ZONE_COLORS = {
    "LA Basin": NAVY,
    "Inland Empire": GOLD,
    "Orange County": TEAL,
    "Coachella Valley": RED,
}
SECTOR_COLORS = {
    "Residential": NAVY,
    "Commercial": TEAL,
    "Industrial": GOLD,
}

CAPACITY_MW = 12_500
RESERVE_TARGET = 0.15
TRIGGER_MW = CAPACITY_MW * (1 - RESERVE_TARGET)
LOAD_FACTOR = 0.72

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
HOLDOUT_STEPS = 6
TRAIN_MONTHS = 30

SCENARIOS = {
    "Conservative": dict(ev_pct=5, solar_pct=5, comm_growth=2, ind_shift=3),
    "Base Case": dict(ev_pct=15, solar_pct=12, comm_growth=5, ind_shift=8),
    "Aggressive": dict(ev_pct=30, solar_pct=25, comm_growth=8, ind_shift=15),
}
SCENARIO_COLORS = {"Conservative": TEAL, "Base Case": NAVY, "Aggressive": RED}

DEFAULT_SEED = 7


# ─────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────
def generate_load_data(seed=DEFAULT_SEED):
    """
    Simulated SCE service-territory daily load data, 2021–2023.
    Zones: LA Basin, Inland Empire, Orange County, Coachella Valley.
    Sectors: Residential, Commercial, Industrial.
    """
    np.random.seed(seed)
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    n = len(dates)

    temps_base = {
        1: 57, 2: 59, 3: 62, 4: 67, 5: 71, 6: 77,
        7: 84, 8: 85, 9: 81, 10: 73, 11: 62, 12: 57,
    }
    cv_offset = {
        1: 7, 2: 10, 3: 13, 4: 18, 5: 23, 6: 27,
        7: 26, 8: 27, 9: 24, 10: 15, 11: 9, 12: 7,
    }
    ie_offset = {k: 4 for k in range(1, 13)}

    def daily_temp(month, zone):
        base = temps_base[month]
        if zone == "Coachella Valley":
            base += cv_offset[month]
        elif zone == "Inland Empire":
            base += ie_offset[month]
        return base + np.random.normal(0, 3)

    zone_base = {
        "LA Basin": 4_500,
        "Inland Empire": 2_250,
        "Orange County": 1_850,
        "Coachella Valley": 720,
    }
    sector_share = {
        "LA Basin": {"Residential": 0.40, "Commercial": 0.42, "Industrial": 0.18},
        "Inland Empire": {"Residential": 0.45, "Commercial": 0.32, "Industrial": 0.23},
        "Orange County": {"Residential": 0.42, "Commercial": 0.45, "Industrial": 0.13},
        "Coachella Valley": {"Residential": 0.55, "Commercial": 0.35, "Industrial": 0.10},
    }

    records = []
    for i, date in enumerate(dates):
        m = date.month
        dow = date.dayofweek
        yr = date.year
        yoy_growth = 1 + 0.018 * (yr - 2021)

        for zone, base_mw in zone_base.items():
            temp = daily_temp(m, zone)
            ac_factor = max(0, (temp - 70) * 0.028)
            htg_factor = max(0, (60 - temp) * 0.008)
            t_factor = 1 + ac_factor + htg_factor
            dow_factor = 1.06 if dow < 5 else (0.94 if dow == 6 else 0.98)

            total_mw = base_mw * t_factor * dow_factor * yoy_growth
            total_mw += np.random.normal(0, base_mw * 0.018)

            for sector, share in sector_share[zone].items():
                if sector == "Residential":
                    s_dow = 1.0 if dow < 5 else 1.04
                elif sector == "Commercial":
                    s_dow = 1.10 if dow < 5 else 0.88
                else:
                    s_dow = 1.12 if dow < 5 else 0.82
                mw = total_mw * share * s_dow / (
                    sector_share[zone]["Residential"] * (1.0 if dow < 5 else 1.04)
                    + sector_share[zone]["Commercial"] * (1.10 if dow < 5 else 0.88)
                    + sector_share[zone]["Industrial"] * (1.12 if dow < 5 else 0.82)
                )
                records.append({
                    "date": date,
                    "year": yr,
                    "month": m,
                    "month_name": date.strftime("%b"),
                    "dow": dow,
                    "dow_name": date.strftime("%a"),
                    "zone": zone,
                    "sector": sector,
                    "temperature_f": round(temp, 1),
                    "load_mw": round(max(mw, 0), 1),
                })

    return pd.DataFrame(records)


def aggregate_data(df_in: pd.DataFrame):
    df = df_in.copy()
    daily = (
        df.groupby(["date", "year", "month", "month_name", "dow", "dow_name"])
        .agg(total_mw=("load_mw", "sum"), temperature_f=("temperature_f", "mean"))
        .reset_index()
    )
    df["gwh"] = df["load_mw"] * 24 / 1000
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            total_gwh=("gwh", "sum"),
            peak_mw=("load_mw", "max"),
            avg_mw=("load_mw", "mean"),
            avg_temp=("temperature_f", "mean"),
        )
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("date").reset_index(drop=True)
    return daily, monthly


# ─────────────────────────────────────────────────────────────
# SARIMA FORECASTING
# ─────────────────────────────────────────────────────────────
def fit_sarima(series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER):
    if not SM_OK:
        raise ImportError("statsmodels is required for SARIMA forecasting")
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def generate_forecast(fit_result, steps, alpha=0.10):
    fc = fit_result.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int(alpha=alpha)


def fit_sarima_forecast(monthly_in: pd.DataFrame, horizon):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if not SM_OK:
        raise ImportError("statsmodels is required for SARIMA forecasting")

    monthly = monthly_in.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    series = pd.Series(
        monthly["total_gwh"].values,
        index=pd.DatetimeIndex(monthly["date"].values, freq="MS"),
    )
    train = series.iloc[:TRAIN_MONTHS]
    test = series.iloc[TRAIN_MONTHS:]

    fit_result = fit_sarima(train)
    hold_pred, hold_ci = generate_forecast(fit_result, HOLDOUT_STEPS)

    fit_full = fit_sarima(series)
    fore_mean, fore_ci = generate_forecast(fit_full, horizon)

    mae = float(mean_absolute_error(test.values, hold_pred.values))
    rmse = float(mean_squared_error(test.values, hold_pred.values) ** 0.5)
    mape = float(np.mean(np.abs((test.values - hold_pred.values) / test.values)) * 100)

    return {
        "train": train,
        "test": test,
        "hold_pred": hold_pred,
        "hold_ci": hold_ci,
        "fore_mean": fore_mean,
        "fore_ci": fore_ci,
        "mae": round(mae, 1),
        "rmse": round(rmse, 1),
        "mape": round(mape, 2),
        "aic": round(fit_result.aic, 1),
        "bic": round(fit_result.bic, 1),
    }


# ─────────────────────────────────────────────────────────────
# SCENARIO ENGINE
# ─────────────────────────────────────────────────────────────
def apply_scenario(fore_mean, fore_ci, ev_pct, solar_pct, comm_growth, ind_shift):
    """
    Adjust base forecast for EV adoption, solar penetration, commercial and
    industrial load changes.
    """
    base = fore_mean.copy()
    ev_delta = base * (ev_pct / 100) * 0.60
    solar_delta = base * (solar_pct / 100) * (-0.45)
    comm_delta = base * 0.42 * (comm_growth / 100)
    ind_delta = base * 0.18 * (-(ind_shift / 100) * 0.4)
    return base + ev_delta + solar_delta + comm_delta + ind_delta


apply_ev_scenario = apply_scenario


def apply_all_scenarios(base_fore, fore_ci):
    return {
        name: apply_scenario(base_fore, fore_ci, **params)
        for name, params in SCENARIOS.items()
    }


# ─────────────────────────────────────────────────────────────
# RESERVE MARGIN / CAPACITY HELPERS
# ─────────────────────────────────────────────────────────────
def gwh_to_peak_mw(gwh_monthly, n_days=30):
    return (gwh_monthly * 1000) / (n_days * 24 * LOAD_FACTOR)


def stress_months(peak_mw_arr):
    return int(np.sum(peak_mw_arr > CAPACITY_MW * 0.95))


def reserve_margin(peak_mw_arr):
    return round((CAPACITY_MW / peak_mw_arr.max() - 1) * 100, 1)


def compute_reserve_margins(scenario_gwh_series):
    """
    Convert monthly GWh series to implied peak MW and reserve-margin metrics.
    """
    peak_mw = gwh_to_peak_mw(scenario_gwh_series.values)
    return {
        "peak_mw": peak_mw,
        "max_peak_mw": float(peak_mw.max()),
        "reserve_margin_pct": reserve_margin(peak_mw),
        "stress_months": stress_months(peak_mw),
        "monthly_margins": [round((CAPACITY_MW / p - 1) * 100, 1) for p in peak_mw],
    }


# ─────────────────────────────────────────────────────────────
# NARRATIVE & REPORT BUILDERS
# ─────────────────────────────────────────────────────────────
def build_executive_narrative(sar, monthly_full, scenario_results, forecast_horizon):
    peak_hist = int(monthly_full["peak_mw"].max())
    yoy = round(
        (
            monthly_full[monthly_full["year"] == 2023]["total_gwh"].sum()
            / monthly_full[monthly_full["year"] == 2022]["total_gwh"].sum()
            - 1
        )
        * 100,
        1,
    )
    base_peak_gwh = float(sar["fore_mean"].max())
    base_peak_mw = float(gwh_to_peak_mw(sar["fore_mean"].values).max())
    agg_peak_mw = float(gwh_to_peak_mw(scenario_results["Aggressive"].values).max())
    base_stress = stress_months(gwh_to_peak_mw(sar["fore_mean"].values))
    util_pct = round(base_peak_mw / CAPACITY_MW * 100, 1)
    return (
        f"Historical territory peak reached <strong>{peak_hist:,} MW</strong> with "
        f"<strong>+{yoy}%</strong> load growth from 2022 to 2023. "
        f"The SARIMA baseline projects a <strong>{base_peak_gwh:,.0f} GWh</strong> peak month "
        f"(~<strong>{base_peak_mw:,.0f} MW</strong> implied peak, "
        f"<strong>{util_pct:.0f}%</strong> of {CAPACITY_MW:,} MW capacity) over the next "
        f"<strong>{forecast_horizon}</strong> months (MAPE <strong>{sar['mape']:.1f}%</strong>). "
        f"Under the aggressive EV/solar scenario, implied peak rises to "
        f"<strong>{agg_peak_mw:,.0f} MW</strong> with "
        f"<strong>{base_stress}</strong> stress months above 95% utilisation — "
        f"signalling capacity planning review if electrification outpaces forecasts."
    )


def build_report_html(sar, monthly_full, scenario_results, meta):
    rows = ""
    for name in ["Conservative", "Base Case", "Aggressive"]:
        peak = float(gwh_to_peak_mw(scenario_results[name].values).max())
        net = (scenario_results[name].sum() / sar["fore_mean"].sum() - 1) * 100
        rows += (
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{peak:,.0f} MW</td>"
            f"<td>{net:+.1f}%</td>"
            f"<td>{reserve_margin(gwh_to_peak_mw(scenario_results[name].values)):.1f}%</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CA Grid Load Forecast Brief</title>
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif; color:{BODY}; margin:40px; }}
  h1 {{ color:{NAVY}; font-size:24px; }}
  .meta {{ color:{MUTED}; font-size:13px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:12px 0 20px; }}
  th {{ background:{NAVY}; color:{GOLD}; text-align:left; padding:8px 10px; }}
  td {{ border-bottom:1px solid {RULE}; padding:7px 10px; }}
  .note {{ background:#FFFBF0; border-left:4px solid {GOLD}; padding:12px 14px;
            font-size:12.5px; line-height:1.6; }}
</style></head><body>
<h1>California Grid Load Forecast — Executive Brief</h1>
<p class="meta">Generated {date.today().isoformat()} · Horizon {meta['horizon']} months ·
Zones: {html.escape(meta['zones'])} · Simulated SCE territory data</p>
<p>Historical peak: <strong>{int(monthly_full['peak_mw'].max()):,} MW</strong> ·
SARIMA MAPE: <strong>{sar['mape']:.1f}%</strong> · AIC: <strong>{sar['aic']:.1f}</strong></p>
<h2>Scenario Peak Comparison</h2>
<table><thead><tr><th>Scenario</th><th>Implied Peak</th><th>Net Load vs Base</th>
<th>Reserve Margin</th></tr></thead><tbody>{rows}</tbody></table>
<div class="note"><strong>Disclaimer:</strong> Synthetic calibrated data for planning demonstration only.
Not for operational dispatch, CAISO market participation, or regulatory filing.</div>
<p style="font-size:11px;color:{MUTED};">Sherriff Abdul-Hamid · poverty360.org</p>
</body></html>"""


def build_report_pdf(sar, monthly_full, scenario_results, meta):
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    w, h = letter
    margin = 0.75 * inch
    c = canvas.Canvas(buf, pagesize=letter)
    y = h - margin

    c.setFillColor(rl_colors.HexColor(NAVY))
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, "CA Grid Load Forecast Brief")
    y -= 22
    c.setFillColor(rl_colors.HexColor(MUTED))
    c.setFont("Helvetica", 10)
    c.drawString(
        margin,
        y,
        f"Generated {date.today().isoformat()} · Horizon {meta['horizon']} mo · "
        f"MAPE {sar['mape']:.1f}%",
    )
    y -= 26
    c.setFont("Helvetica", 9.5)
    c.drawString(
        margin,
        y,
        f"Historical peak: {int(monthly_full['peak_mw'].max()):,} MW · "
        f"Capacity assumption: {CAPACITY_MW:,} MW",
    )
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(rl_colors.HexColor(NAVY))
    c.drawString(margin, y, "Scenario Peaks")
    y -= 14
    c.setFont("Helvetica", 9)
    for name in ["Conservative", "Base Case", "Aggressive"]:
        peak = float(gwh_to_peak_mw(scenario_results[name].values).max())
        net = (scenario_results[name].sum() / sar["fore_mean"].sum() - 1) * 100
        c.drawString(
            margin,
            y,
            f"{name}: {peak:,.0f} MW peak · {net:+.1f}% vs base · "
            f"reserve {reserve_margin(gwh_to_peak_mw(scenario_results[name].values)):.1f}%",
        )
        y -= 12
    c.setFillColor(rl_colors.HexColor(MUTED))
    c.setFont("Helvetica", 8.5)
    c.drawString(
        margin,
        margin - 2,
        "Decision-support demo · simulated SCE load · not for operational use.",
    )
    c.save()
    return buf.getvalue()
