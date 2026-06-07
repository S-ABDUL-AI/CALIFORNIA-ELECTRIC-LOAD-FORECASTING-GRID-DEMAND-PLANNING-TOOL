"""
California Electric Load Forecasting & Grid Demand Planning Tool
SCE Service Territory — Residential · Commercial · Industrial
Sherriff Abdul-Hamid | Development Economist & Data Scientist
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from model import (
    AMBER,
    BODY,
    CAPACITY_MW,
    DEFAULT_SEED,
    GOLD,
    GREEN,
    MUTED,
    NAVY,
    OFF_WHITE,
    PURPLE,
    RED,
    RULE,
    SCENARIO_COLORS,
    SCENARIOS,
    SECTOR_COLORS,
    TEAL,
    TRIGGER_MW,
    WHITE,
    ZONE_COLORS,
    ZONES,
    aggregate_data,
    apply_scenario,
    build_executive_narrative,
    build_report_html,
    build_report_pdf,
    fit_sarima_forecast,
    generate_load_data,
    gwh_to_peak_mw,
    reserve_margin,
    stress_months,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA Grid Demand Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif; background:{OFF_WHITE};
  }}
  .hero-wrap {{
    background: linear-gradient(135deg,{NAVY} 0%,#2D4F8A 60%,#1E3A6E 100%);
    border-left: 6px solid {GOLD}; border-radius: 6px;
    padding: 32px 40px 26px; margin-bottom: 22px;
  }}
  .hero-eye   {{ font-size:11px; font-weight:700; letter-spacing:2.5px;
                 color:{GOLD}; text-transform:uppercase; margin-bottom:9px; }}
  .hero-title {{ font-size:26px; font-weight:700; color:#FFFFFF; line-height:1.3; margin-bottom:9px; }}
  .hero-sub   {{ font-size:13.5px; color:#B0BFD8; line-height:1.65; max-width:860px; }}
  .sec-lbl {{ font-size:10px; font-weight:700; letter-spacing:2px;
              color:{GOLD}; text-transform:uppercase; margin-bottom:3px; }}
  .sec-ttl {{ font-size:18px; font-weight:700; color:{NAVY}; margin-bottom:3px; }}
  .sec-sub {{ font-size:12.5px; color:{MUTED}; margin-bottom:14px; }}
  .kpi-card  {{ background:{WHITE}; border:1px solid {RULE};
                border-top:3px solid {NAVY}; border-radius:4px;
                padding:13px 17px; box-shadow:0 1px 4px rgba(0,0,0,.05); }}
  .kpi-label {{ font-size:10px; font-weight:700; letter-spacing:1px;
                color:{MUTED}; text-transform:uppercase; margin-bottom:3px; }}
  .kpi-val   {{ font-size:23px; font-weight:700; color:{NAVY}; line-height:1.1; }}
  .kpi-sub   {{ font-size:11px; color:{MUTED}; margin-top:2px; }}
  .kpi-good  {{ border-top-color:{GREEN} !important; }}
  .kpi-warn  {{ border-top-color:{AMBER} !important; }}
  .kpi-alert {{ border-top-color:{RED}   !important; }}
  .scope-box {{ background:#FFFBF0; border:1px solid {AMBER};
                border-left:4px solid {AMBER}; border-radius:4px;
                padding:8px 14px; font-size:11.5px; color:{AMBER};
                margin-bottom:16px; }}
  .disclaimer-box {{
    background:#FFFBF0; border:1px solid #E8C97A; border-left:5px solid {AMBER};
    border-radius:4px; padding:14px 18px; font-size:12px; color:{BODY};
    line-height:1.65; margin-bottom:18px;
  }}
  .disclaimer-title {{
    font-size:10px; font-weight:700; letter-spacing:1.5px;
    color:{AMBER}; text-transform:uppercase; margin-bottom:6px;
  }}
  .insight-box {{
    background:#F0F4FF; border-left:4px solid {NAVY}; border-radius:4px;
    padding:14px 18px; font-size:13px; color:{BODY}; line-height:1.65;
    margin:12px 0 18px;
  }}
  .insight-label {{
    font-size:10px; font-weight:700; letter-spacing:1.5px;
    color:{GOLD}; text-transform:uppercase; margin-bottom:6px;
  }}
  .report-card {{
    background:{WHITE}; border:1px solid {RULE}; border-left:5px solid {GOLD};
    border-radius:6px; padding:20px 24px; margin-bottom:22px;
    box-shadow:0 2px 12px rgba(31,56,100,0.06);
  }}
  .report-title {{ font-size:16px; font-weight:700; color:{NAVY}; margin-bottom:4px; }}
  .report-sub   {{ font-size:13px; color:{MUTED}; line-height:1.55; margin-bottom:12px; }}
  .report-loc   {{ font-size:11.5px; color:{AMBER}; font-weight:600; margin-bottom:10px; }}
  .brief-navy {{ background:#F0F4FF; border-left:4px solid {NAVY};
                 border:1px solid #C4D0F5; border-radius:4px; padding:13px 16px; }}
  .brief-gold {{ background:#FFFBF0; border-left:4px solid {GOLD};
                 border:1px solid #E8C97A; border-radius:4px; padding:13px 16px; }}
  .brief-red  {{ background:#FFF5F5; border-left:4px solid {RED};
                 border:1px solid #FFC9C9; border-radius:4px; padding:13px 16px; }}
  .brief-green{{ background:#F0FFF4; border-left:4px solid {GREEN};
                 border:1px solid #A8D5B5; border-radius:4px; padding:13px 16px; }}
  .brief-head {{ font-size:10px; font-weight:700; letter-spacing:1.5px;
                 text-transform:uppercase; margin-bottom:5px; }}
  .brief-body {{ font-size:12.5px; color:{BODY}; line-height:1.6; }}
  .byline     {{ background:{NAVY}; border-radius:4px; padding:15px 22px;
                 font-size:11.5px; color:#B0BFD8; line-height:1.8; margin-top:26px; }}
  .byline a   {{ color:{GOLD}; text-decoration:none; }}
  div[data-testid="stButton"] > button {{
    background:{NAVY}; color:#FFFFFF; border:none; border-radius:3px; font-weight:600;
  }}
  div[data-testid="stDownloadButton"] > button {{
    background:{NAVY}; color:#FFFFFF; border:none; border-radius:4px; font-weight:600;
  }}
  .stSlider   {{ padding-top:4px !important; }}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# CACHED DATA / MODEL CALLS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_data(seed=DEFAULT_SEED):
    return generate_load_data(seed)


@st.cache_data
def cached_aggregate(df_in: pd.DataFrame):
    return aggregate_data(df_in)


@st.cache_data
def cached_sarima(monthly_in: pd.DataFrame, horizon: int):
    return fit_sarima_forecast(monthly_in, horizon)


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def base_layout(height=320, margin=None, **kw):
    m = margin or dict(t=32, b=40, l=55, r=20)
    xaxis = dict(showgrid=False, zeroline=False, linecolor=RULE, showline=True)
    yaxis = dict(showgrid=True, gridcolor=RULE, zeroline=False, linecolor=RULE)
    if "xaxis" in kw:
        xaxis.update(kw.pop("xaxis"))
    if "yaxis" in kw:
        yaxis.update(kw.pop("yaxis"))
    return dict(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        height=height,
        margin=m,
        font=dict(family="Source Sans 3,sans-serif", color=BODY, size=11),
        xaxis=xaxis,
        yaxis=yaxis,
        **kw,
    )


GRADIENT_LOAD = [[0, NAVY], [0.5, GOLD], [1, RED]]


def _gradient_marker(values, colorscale):
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return dict(color=NAVY)
    return dict(
        color=arr,
        colorscale=colorscale,
        cmin=float(arr.min()),
        cmax=float(arr.max()),
        line=dict(width=0),
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
    <div style="background:{NAVY};border-radius:4px;padding:13px 15px;margin-bottom:14px;">
      <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:{GOLD};
                  text-transform:uppercase;margin-bottom:4px;">⚡ Planning Tool</div>
      <div style="font-size:13px;font-weight:700;color:#FFFFFF;">CA Grid Load Forecast</div>
      <div style="font-size:11px;color:#B0BFD8;margin-top:2px;">SCE Service Territory</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️  About this tool"):
        st.markdown(
            """
        Simulates and forecasts electricity load across SCE's four service territory
        zones using SARIMA time-series modelling, temperature-correlated demand curves,
        and scenario analysis for EV adoption and rooftop solar.

        **Download:** use the **Executive Forecast Brief** card below the hero banner.

        **Data:** Calibrated synthetic load profiles (2021–2023), realistic for
        Southern California summer peaks, day-of-week patterns, and zone-level
        disaggregation.
        """
        )

    st.markdown(f'<div class="sec-lbl">Display Settings</div>', unsafe_allow_html=True)
    selected_zones = st.multiselect(
        "Zones to display",
        ZONES,
        default=ZONES,
    )
    show_sectors = st.checkbox("Show sector breakdown", value=True)
    forecast_horizon = st.slider("Forecast horizon (months)", 12, 24, 18, step=6)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df_raw = cached_load_data()
df_filt = df_raw[df_raw["zone"].isin(selected_zones)] if selected_zones else df_raw
daily_df, monthly_df = cached_aggregate(df_filt)

_, monthly_full = cached_aggregate(df_raw)

with st.spinner("Fitting SARIMA model…"):
    sar = cached_sarima(monthly_full, forecast_horizon)

base_fore = sar["fore_mean"].copy()
scenario_results = {
    name: apply_scenario(base_fore, sar["fore_ci"], **params)
    for name, params in SCENARIOS.items()
}
all_scenario_results = scenario_results.copy()
report_meta = {
    "horizon": forecast_horizon,
    "zones": ", ".join(selected_zones) if selected_zones else "All zones",
}

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero-wrap">
  <div class="hero-eye">
    California Electric Load Forecast · SCE Service Territory ·
    SARIMA · EV &amp; Solar Scenarios · Grid Stress Analysis
  </div>
  <div class="hero-title">
    What will California's electricity demand look like<br>in 2024–2026 — and is the grid ready?
  </div>
  <div class="hero-sub">
    This tool combines temperature-driven load simulation, SARIMA time-series forecasting,
    and scenario analysis across EV adoption, rooftop solar penetration, and industrial
    demand response — giving SCE planners a data-driven basis for capacity investment
    decisions across four service-territory zones.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="disclaimer-box">
  <div class="disclaimer-title">Important disclaimer</div>
  All load profiles in this tool are <strong>calibrated synthetic data</strong> approximating
  SCE service territory conditions. Forecasts, scenario outputs, and capacity stress indicators
  are <strong>decision-support analytics only</strong> — not official SCE operational data,
  CAISO market forecasts, or CPUC regulatory filings. Do not use for grid dispatch,
  resource adequacy compliance, or investment authorization without independent validation
  against verified metering and planning models.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="report-card">
  <div class="report-title">📄 Executive Forecast Brief</div>
  <div class="report-sub">One-page brief with SARIMA accuracy, scenario peak comparison,
  and reserve margin analysis — ready for planning review.</div>
  <div class="report-loc">↓ Download location: use the buttons below</div>
</div>
""",
    unsafe_allow_html=True,
)
dl1, dl2, dl3 = st.columns(3)
report_slug = f"ca_grid_forecast_{forecast_horizon}mo"
with dl1:
    try:
        st.download_button(
            "Download report (.pdf)",
            data=build_report_pdf(sar, monthly_full, scenario_results, report_meta),
            file_name=f"{report_slug}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except ModuleNotFoundError:
        st.caption("PDF requires reportlab on deploy.")
with dl2:
    st.download_button(
        "Download report (.html)",
        data=build_report_html(sar, monthly_full, scenario_results, report_meta).encode("utf-8"),
        file_name=f"{report_slug}.html",
        mime="text/html",
        use_container_width=True,
    )
with dl3:
    st.download_button(
        "Download monthly data (.csv)",
        data=monthly_full.to_csv(index=False).encode("utf-8"),
        file_name="sce_monthly_load.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    f"""
<div class="insight-box">
  <div class="insight-label">Executive insight</div>
  {build_executive_narrative(sar, monthly_full, scenario_results, forecast_horizon)}
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Historical Load Analysis",
    "📈  Demand Forecasting",
    "🔋  EV & Solar Scenarios",
    "🎯  Executive Planning Dashboard",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — HISTORICAL LOAD ANALYSIS
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-lbl">Historical Load Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-ttl">SCE Service Territory — 2021–2023</div>', unsafe_allow_html=True)

    total_gwh = int(monthly_full["total_gwh"].sum())
    peak_mw = int(monthly_full["peak_mw"].max())
    avg_daily = int(daily_df["total_mw"].mean())
    yoy_growth = round(
        (
            monthly_full[monthly_full["year"] == 2023]["total_gwh"].sum()
            / monthly_full[monthly_full["year"] == 2022]["total_gwh"].sum()
            - 1
        )
        * 100,
        1,
    )

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub, cls in [
        (k1, "Total Load 3yr", f"{total_gwh:,} GWh", "Jan 2021–Dec 2023", ""),
        (k2, "Recorded Peak", f"{peak_mw:,} MW", "SCE territory max", "kpi-alert"),
        (k3, "Avg Daily Demand", f"{avg_daily:,} MW", "3-year average", ""),
        (
            k4,
            "YoY Load Growth",
            f"+{yoy_growth}%",
            "2022→2023",
            "kpi-good" if yoy_growth < 3 else "kpi-warn",
        ),
    ]:
        with col:
            st.markdown(
                f"""
            <div class="kpi-card {cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sec-lbl">Exhibit 1 — Daily Peak Load by Zone</div>',
        unsafe_allow_html=True,
    )
    daily_zone = df_filt.groupby(["date", "zone"])["load_mw"].sum().reset_index()
    fig1 = go.Figure()
    for zone in selected_zones:
        z_data = daily_zone[daily_zone["zone"] == zone]
        fig1.add_trace(
            go.Scatter(
                x=z_data["date"],
                y=z_data["load_mw"],
                mode="lines",
                name=zone,
                line=dict(color=ZONE_COLORS.get(zone, MUTED), width=1.5),
            )
        )
    fig1.update_layout(
        **base_layout(height=290),
        title=dict(text="Daily Load by Zone (MW)", font=dict(size=12, color=NAVY), x=0),
        xaxis_title="Date",
        yaxis_title="MW",
        legend=dict(orientation="h", y=-0.25, x=0),
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div class="sec-lbl">Exhibit 2 — Monthly Seasonal Profile</div>',
            unsafe_allow_html=True,
        )
        mon_avg = monthly_full.groupby("month")["total_gwh"].mean().reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        mon_avg["month_name"] = mon_avg["month"].apply(lambda x: month_names[x - 1])
        fig2 = go.Figure(
            go.Bar(
                x=mon_avg["month_name"],
                y=mon_avg["total_gwh"],
                marker=_gradient_marker(mon_avg["total_gwh"], GRADIENT_LOAD),
                text=[f"{v:,.0f}" for v in mon_avg["total_gwh"]],
                textposition="outside",
                textfont=dict(size=9),
            )
        )
        fig2.update_layout(
            **base_layout(height=270),
            title=dict(text="Avg Monthly Load (GWh)", font=dict(size=11, color=NAVY), x=0),
            xaxis_title="Month",
            yaxis_title="GWh",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown(
            '<div class="sec-lbl">Exhibit 3 — Temperature–Load Correlation</div>',
            unsafe_allow_html=True,
        )
        daily_sample = daily_df.sample(min(800, len(daily_df)), random_state=7)
        fig3 = px.scatter(
            daily_sample,
            x="temperature_f",
            y="total_mw",
            trendline="ols",
            color_discrete_sequence=[NAVY],
            labels={"temperature_f": "Temperature (°F)", "total_mw": "Daily Load (MW)"},
        )
        fig3.update_traces(marker=dict(size=4, opacity=0.5))
        fig3.update_layout(
            **base_layout(height=270),
            title=dict(text="Temperature vs Daily Load", font=dict(size=11, color=NAVY), x=0),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(
            '<div class="sec-lbl">Exhibit 4 — Day-of-Week Load Effect</div>',
            unsafe_allow_html=True,
        )
        dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_avg = daily_df.groupby("dow_name")["total_mw"].mean().reindex(dow_order).reset_index()
        fig4 = go.Figure(
            go.Bar(
                x=dow_avg["dow_name"],
                y=dow_avg["total_mw"],
                marker_color=[NAVY if d not in ["Sat", "Sun"] else TEAL for d in dow_avg["dow_name"]],
            )
        )
        fig4.update_layout(
            **base_layout(height=250),
            title=dict(text="Avg Load by Day of Week (MW)", font=dict(size=11, color=NAVY), x=0),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.markdown(
            '<div class="sec-lbl">Exhibit 5 — Trend Decomposition (Monthly)</div>',
            unsafe_allow_html=True,
        )
        series_full = pd.Series(
            monthly_full["total_gwh"].values,
            index=pd.DatetimeIndex(monthly_full["date"].values, freq="MS"),
        )
        decomp = seasonal_decompose(series_full, model="additive", period=12, extrapolate_trend="freq")
        fig5 = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Trend", "Seasonal", "Residual"],
            vertical_spacing=0.08,
        )
        for i, (component, color) in enumerate(
            [(decomp.trend, NAVY), (decomp.seasonal, GOLD), (decomp.resid, RED)], 1
        ):
            fig5.add_trace(
                go.Scatter(
                    x=series_full.index,
                    y=component.values,
                    line=dict(color=color, width=1.8),
                    showlegend=False,
                ),
                row=i,
                col=1,
            )
        fig5.update_layout(
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE,
            height=270,
            margin=dict(t=35, b=20, l=50, r=15),
            font=dict(size=9),
        )
        for ax in [f"xaxis{i}" for i in ["", "2", "3"]] + [f"yaxis{i}" for i in ["", "2", "3"]]:
            fig5.update_layout(**{ax: dict(showgrid=False, zeroline=False)})
        st.plotly_chart(fig5, use_container_width=True)

    if show_sectors:
        st.markdown(
            '<div class="sec-lbl">Exhibit 6 — Monthly Load by Sector (GWh)</div>',
            unsafe_allow_html=True,
        )
        df_raw["gwh"] = df_raw["load_mw"] * 24 / 1000
        sec_monthly = df_raw.groupby(["year", "month", "sector"])["gwh"].sum().reset_index()
        sec_monthly["date"] = pd.to_datetime(
            sec_monthly["year"].astype(str) + "-" + sec_monthly["month"].astype(str).str.zfill(2)
        )
        fig6 = go.Figure()
        for sector, color in SECTOR_COLORS.items():
            s = sec_monthly[sec_monthly["sector"] == sector]
            fig6.add_trace(
                go.Bar(x=s["date"], y=s["gwh"], name=sector, marker_color=color)
            )
        fig6.update_layout(
            **base_layout(height=270),
            barmode="stack",
            title=dict(text="Monthly Load by Sector — Stacked", font=dict(size=11, color=NAVY), x=0),
            xaxis_title="Month",
            yaxis_title="GWh",
            legend=dict(orientation="h", y=-0.28, x=0),
        )
        st.plotly_chart(fig6, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 2 — DEMAND FORECASTING
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-lbl">Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-ttl">SARIMA(1,1,1)(1,1,1,12) — Monthly GWh Forecast</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val, sub, cls in [
        (k1, "Forecast Horizon", f"{forecast_horizon} mo", "User-selected", ""),
        (k2, "MAE", f"{sar['mae']:,.0f} GWh", "Holdout 6 months", ""),
        (k3, "RMSE", f"{sar['rmse']:,.0f} GWh", "Holdout 6 months", ""),
        (
            k4,
            "MAPE",
            f"{sar['mape']:.2f}%",
            "Mean abs % error",
            "kpi-good" if sar["mape"] < 5 else "kpi-warn",
        ),
        (k5, "SARIMA AIC", f"{sar['aic']:,.1f}", "Model fit criterion", ""),
    ]:
        with col:
            st.markdown(
                f"""
            <div class="kpi-card {cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sec-lbl">Exhibit 1 — Historical + Holdout + Forecast (GWh)</div>',
        unsafe_allow_html=True,
    )
    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=sar["train"].index,
            y=sar["train"].values,
            mode="lines",
            name="Actual (training)",
            line=dict(color=NAVY, width=2),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=sar["test"].index,
            y=sar["test"].values,
            mode="lines+markers",
            name="Actual (holdout)",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=7, color=GREEN),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=sar["hold_pred"].index,
            y=sar["hold_pred"].values,
            mode="lines+markers",
            name="Forecast (holdout)",
            line=dict(color=GOLD, width=2.2, dash="dot"),
            marker=dict(size=7, color=GOLD),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=sar["fore_mean"].index,
            y=sar["fore_mean"].values,
            mode="lines",
            name=f"Forecast ({forecast_horizon} mo ahead)",
            line=dict(color=RED, width=2.5),
        )
    )
    ci = sar["fore_ci"]
    fig_fc.add_trace(
        go.Scatter(
            x=list(sar["fore_mean"].index) + list(sar["fore_mean"].index[::-1]),
            y=list(ci.iloc[:, 1]) + list(ci.iloc[::-1, 0]),
            fill="toself",
            fillcolor="rgba(200,56,42,0.10)",
            line=dict(width=0),
            name="90% CI",
            showlegend=True,
        )
    )
    fig_fc.add_vline(
        x="2024-01-01",
        line=dict(color=MUTED, width=1, dash="dash"),
        annotation_text=" Forecast starts",
        annotation_font=dict(size=10, color=MUTED),
    )
    fig_fc.update_layout(
        **base_layout(height=340),
        title=dict(
            text="Monthly Electricity Demand Forecast — SCE Territory (GWh)",
            font=dict(size=13, color=NAVY),
            x=0,
        ),
        xaxis_title="Month",
        yaxis_title="GWh",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.markdown(
            '<div class="sec-lbl">Exhibit 2 — Actual vs Forecast (Holdout Period)</div>',
            unsafe_allow_html=True,
        )
        fig_hold = go.Figure()
        fig_hold.add_trace(
            go.Scatter(
                x=sar["test"].index,
                y=sar["test"].values,
                mode="lines+markers",
                name="Actual",
                line=dict(color=NAVY, width=2.5),
                marker=dict(size=9, color=NAVY),
            )
        )
        fig_hold.add_trace(
            go.Scatter(
                x=sar["hold_pred"].index,
                y=sar["hold_pred"].values,
                mode="lines+markers",
                name="SARIMA Forecast",
                line=dict(color=GOLD, width=2.2, dash="dot"),
                marker=dict(size=9, color=GOLD, symbol="diamond"),
            )
        )
        ci_h = sar["hold_ci"]
        fig_hold.add_trace(
            go.Scatter(
                x=list(sar["hold_pred"].index) + list(sar["hold_pred"].index[::-1]),
                y=list(ci_h.iloc[:, 1]) + list(ci_h.iloc[::-1, 0]),
                fill="toself",
                fillcolor="rgba(201,168,76,0.15)",
                line=dict(width=0),
                name="90% CI",
            )
        )
        fig_hold.update_layout(
            **base_layout(height=280),
            title=dict(text="6-Month Holdout: Actual vs Forecast", font=dict(size=11, color=NAVY), x=0),
            legend=dict(orientation="h", y=-0.30, x=0),
        )
        st.plotly_chart(fig_hold, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-lbl">Model Performance Summary</div>', unsafe_allow_html=True)
        holdout_table = pd.DataFrame({
            "Month": [d.strftime("%b %Y") for d in sar["test"].index],
            "Actual (GWh)": [f"{v:,.0f}" for v in sar["test"].values],
            "Forecast (GWh)": [f"{v:,.0f}" for v in sar["hold_pred"].values],
            "Error %": [
                f"{abs((a - f) / a) * 100:.1f}%"
                for a, f in zip(sar["test"].values, sar["hold_pred"].values)
            ],
        })
        st.dataframe(holdout_table, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
        <div class="brief-navy" style="margin-top:10px;">
          <div class="brief-head" style="color:{NAVY};">Model Specification</div>
          <div class="brief-body">
            <strong>SARIMA(1,1,1)(1,1,1,12)</strong> — non-seasonal AR(1),
            one regular and one seasonal difference, seasonal AR(1) and MA(1)
            with 12-period cycle. Fit on 30 months training data.
            MAPE of <strong>{sar['mape']:.2f}%</strong> is within industry benchmark
            of ±5% for short-range utility load forecasting.
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-lbl">Exhibit 3 — Monthly Forecast Table</div>',
        unsafe_allow_html=True,
    )
    ci_full = sar["fore_ci"]
    fore_tbl = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in sar["fore_mean"].index],
        "Forecast (GWh)": sar["fore_mean"].values.round(0).astype(int),
        "Lower CI (GWh)": ci_full.iloc[:, 0].values.round(0).astype(int),
        "Upper CI (GWh)": ci_full.iloc[:, 1].values.round(0).astype(int),
        "YoY Change": [
            "+{:.1f}%".format(
                (f / monthly_full[monthly_full["month"] == d.month]["total_gwh"].mean() - 1) * 100
            )
            for d, f in zip(sar["fore_mean"].index, sar["fore_mean"].values)
        ],
    })
    st.dataframe(fore_tbl, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# TAB 3 — EV ADOPTION & SOLAR SCENARIOS
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="sec-lbl">EV Adoption & Solar Penetration Scenarios</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-ttl">How do electrification and distributed energy resources reshape peak demand?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sec-lbl" style="margin-top:4px;">Custom Scenario — Adjust Parameters</div>',
        unsafe_allow_html=True,
    )
    sl1, sl2, sl3, sl4 = st.columns(4)
    with sl1:
        ev_rate = st.slider(
            "EV Adoption Rate (%)",
            0,
            40,
            15,
            1,
            help="% of personal vehicles electrified in territory",
        )
    with sl2:
        sol_rate = st.slider(
            "Rooftop Solar Penetration (%)",
            0,
            30,
            12,
            1,
            help="% of residential/commercial rooftops with solar",
        )
    with sl3:
        comm_g = st.slider(
            "Commercial Demand Growth (%)",
            -10,
            10,
            5,
            1,
            help="Net change in commercial sector load",
        )
    with sl4:
        ind_sh = st.slider(
            "Industrial Load Shift (%)",
            0,
            15,
            8,
            1,
            help="% of industrial load shifted via demand response",
        )

    custom_fore = apply_scenario(base_fore, sar["fore_ci"], ev_rate, sol_rate, comm_g, ind_sh)
    all_scenario_results = {**scenario_results, "Custom": custom_fore}

    st.markdown("<br>", unsafe_allow_html=True)
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    for col_obj, sname, cls_name in [
        (s_col1, "Conservative", "kpi-good"),
        (s_col2, "Base Case", ""),
        (s_col3, "Aggressive", "kpi-warn"),
        (s_col4, "Custom", ""),
    ]:
        sc_total = all_scenario_results[sname].sum()
        sc_peak = all_scenario_results[sname].max()
        base_tot = base_fore.sum()
        delta = (sc_total / base_tot - 1) * 100
        with col_obj:
            st.markdown(
                f"""
            <div class="kpi-card {cls_name}">
              <div class="kpi-label">{sname}</div>
              <div class="kpi-val">{sc_peak:,.0f} GWh</div>
              <div class="kpi-sub">Peak month · Net load {delta:+.1f}% vs base</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sec-lbl">Exhibit 1 — Net Load Under Three Scenarios vs Base Forecast</div>',
        unsafe_allow_html=True,
    )
    fig_sc = go.Figure()
    fig_sc.add_trace(
        go.Scatter(
            x=base_fore.index,
            y=base_fore.values,
            mode="lines",
            name="Base Forecast (no new load)",
            line=dict(color=MUTED, width=1.5, dash="dot"),
        )
    )
    for sname, color in SCENARIO_COLORS.items():
        sc = all_scenario_results[sname]
        fig_sc.add_trace(
            go.Scatter(
                x=sc.index,
                y=sc.values,
                mode="lines",
                name=sname,
                line=dict(color=color, width=2.2),
            )
        )
    fig_sc.add_trace(
        go.Scatter(
            x=custom_fore.index,
            y=custom_fore.values,
            mode="lines",
            name="Custom (slider)",
            line=dict(color=PURPLE, width=2.2, dash="dash"),
        )
    )
    fig_sc.update_layout(
        **base_layout(height=320),
        title=dict(text="Monthly Net Load by Scenario (GWh)", font=dict(size=13, color=NAVY), x=0),
        xaxis_title="Month",
        yaxis_title="GWh",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown(
            '<div class="sec-lbl">Exhibit 2 — Peak Demand Delta by Scenario Component</div>',
            unsafe_allow_html=True,
        )
        components = ["EV Adoption", "Rooftop Solar", "Commercial Growth", "Industrial DR"]
        base_peak = float(base_fore.max())
        deltas = {
            "Conservative": [
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=5, solar_pct=0, comm_growth=0, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=5, comm_growth=0, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=0, comm_growth=2, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=0, comm_growth=0, ind_shift=3).max() - base_peak),
            ],
            "Base Case": [
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=15, solar_pct=0, comm_growth=0, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=12, comm_growth=0, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=0, comm_growth=5, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=0, comm_growth=0, ind_shift=8).max() - base_peak),
            ],
            "Aggressive": [
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=30, solar_pct=0, comm_growth=0, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=25, comm_growth=0, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=0, comm_growth=8, ind_shift=0).max() - base_peak),
                float(apply_scenario(base_fore, sar["fore_ci"], ev_pct=0, solar_pct=0, comm_growth=0, ind_shift=15).max() - base_peak),
            ],
        }
        fig_delta = go.Figure()
        comp_colors = [AMBER, GREEN, TEAL, NAVY]
        for j, (comp, ccolor) in enumerate(zip(components, comp_colors)):
            fig_delta.add_trace(
                go.Bar(
                    name=comp,
                    x=list(SCENARIO_COLORS.keys()),
                    y=[deltas[s][j] for s in SCENARIO_COLORS.keys()],
                    marker_color=ccolor,
                )
            )
        fig_delta.update_layout(
            **base_layout(height=280),
            barmode="group",
            title=dict(
                text="Peak Demand Δ vs Base (GWh, by component)",
                font=dict(size=11, color=NAVY),
                x=0,
            ),
            legend=dict(orientation="h", y=-0.32, x=0, font=dict(size=9)),
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    with col_f:
        st.markdown(
            '<div class="sec-lbl">Exhibit 3 — Load Shape: Weekday Peak Profile</div>',
            unsafe_allow_html=True,
        )
        hours = list(range(24))
        base_shape = np.array([
            0.72, 0.70, 0.68, 0.67, 0.68, 0.72, 0.78, 0.84, 0.89, 0.92,
            0.94, 0.96, 0.97, 0.98, 0.99, 1.00, 1.00, 0.99, 0.98, 0.97,
            0.95, 0.90, 0.84, 0.77,
        ])
        ev_add = np.array(
            [0] * 8 + [-0.01] * 3 + [0.01] * 3 + [0] * 3 + [0.06, 0.10, 0.12, 0.12, 0.10, 0.07, 0.04]
        )
        sol_sub = np.array(
            [0] * 6 + [-0.01, -0.04, -0.08, -0.11, -0.13, -0.14, -0.14, -0.13, -0.11, -0.08, -0.04, -0.01] + [0] * 6
        )
        fig_shape = go.Figure()
        fig_shape.add_trace(
            go.Scatter(
                x=hours,
                y=base_shape,
                mode="lines",
                name="Base Load Shape",
                line=dict(color=MUTED, width=1.8, dash="dot"),
            )
        )
        for sname, ev_p, sol_p in [
            ("Conservative", 5, 5),
            ("Base Case", 15, 12),
            ("Aggressive", 30, 25),
        ]:
            adj = base_shape + (ev_p / 100) * ev_add + (sol_p / 100) * sol_sub
            fig_shape.add_trace(
                go.Scatter(
                    x=hours,
                    y=adj,
                    mode="lines",
                    name=sname,
                    line=dict(color=SCENARIO_COLORS[sname], width=2),
                )
            )
        fig_shape.update_layout(
            **base_layout(
                height=280,
                xaxis=dict(
                    title="Hour of Day",
                    tickvals=list(range(0, 24, 3)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                    showgrid=False,
                    zeroline=False,
                    linecolor=RULE,
                    showline=True,
                ),
            ),
            title=dict(
                text="Normalised Hourly Load Shape (Summer Weekday)",
                font=dict(size=11, color=NAVY),
                x=0,
            ),
            yaxis_title="Load (normalised)",
            legend=dict(orientation="h", y=-0.30, x=0, font=dict(size=9)),
        )
        st.plotly_chart(fig_shape, use_container_width=True)

    st.markdown(
        '<div class="sec-lbl" style="margin-top:4px;">Scenario Assumptions Summary</div>',
        unsafe_allow_html=True,
    )
    sc1, sc2, sc3 = st.columns(3)
    for col_obj, sname, css, color in [
        (sc1, "Conservative", "brief-navy", NAVY),
        (sc2, "Base Case", "brief-gold", AMBER),
        (sc3, "Aggressive", "brief-red", RED),
    ]:
        p = SCENARIOS[sname]
        net = (all_scenario_results[sname].sum() / base_fore.sum() - 1) * 100
        with col_obj:
            st.markdown(
                f"""
            <div class="{css}">
              <div class="brief-head" style="color:{color};">{sname}</div>
              <div class="brief-body">
                🚗 <strong>EV:</strong> {p['ev_pct']}% adoption<br>
                ☀️ <strong>Solar:</strong> {p['solar_pct']}% penetration<br>
                🏢 <strong>Commercial growth:</strong> +{p['comm_growth']}%<br>
                🏭 <strong>Industrial DR:</strong> {p['ind_shift']}% shifted<br>
                📊 <strong>Net load change:</strong> {net:+.1f}% vs base
              </div>
            </div>""",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════
# TAB 4 — EXECUTIVE PLANNING DASHBOARD
# ═════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-lbl">Executive Planning Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-ttl">Grid Capacity, Stress Indicators & Investment Triggers</div>',
        unsafe_allow_html=True,
    )

    fore_peak_mw_base = gwh_to_peak_mw(base_fore.values)
    fore_peak_mw = {
        sname: gwh_to_peak_mw(all_scenario_results[sname].values)
        for sname in all_scenario_results
    }
    STRESS_THRESH = CAPACITY_MW * 0.95

    base_peak_mw = float(fore_peak_mw_base.max())
    base_stress = stress_months(fore_peak_mw_base)
    base_reserve = reserve_margin(fore_peak_mw_base)
    agg_peak = float(fore_peak_mw["Aggressive"].max())
    agg_stress = stress_months(fore_peak_mw["Aggressive"])

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val, sub, cls in [
        (k1, "SCE Installed Capacity", f"{CAPACITY_MW:,} MW", "Planning assumption", ""),
        (k2, "Base Forecast Peak", f"{base_peak_mw:,.0f} MW", "Max monthly implied", ""),
        (
            k3,
            "Base Reserve Margin",
            f"{base_reserve:.1f}%",
            "vs 15% target",
            "kpi-good" if base_reserve > 15 else "kpi-warn",
        ),
        (
            k4,
            "Stress Months (Base)",
            str(base_stress),
            "Months >95% capacity",
            "kpi-alert" if base_stress > 3 else ("kpi-warn" if base_stress > 0 else "kpi-good"),
        ),
        (
            k5,
            "Investment Trigger",
            f"{TRIGGER_MW:,.0f} MW",
            "Load level requiring new capacity",
            "kpi-warn",
        ),
    ]:
        with col:
            st.markdown(
                f"""
            <div class="kpi-card {cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sec-lbl">Exhibit 1 — Projected Peak Load vs Capacity by Scenario</div>',
        unsafe_allow_html=True,
    )
    fig_cap = go.Figure()
    fig_cap.add_hline(
        y=CAPACITY_MW,
        line=dict(color=RED, width=1.5, dash="dash"),
        annotation_text="  Installed Capacity (12,500 MW)",
        annotation_font=dict(size=10, color=RED),
    )
    fig_cap.add_hline(
        y=STRESS_THRESH,
        line=dict(color=AMBER, width=1.5, dash="dot"),
        annotation_text="  95% Stress Threshold (11,875 MW)",
        annotation_font=dict(size=10, color=AMBER),
    )
    fig_cap.add_hline(
        y=TRIGGER_MW,
        line=dict(color=GOLD, width=1.5, dash="dot"),
        annotation_text="  Investment Trigger (10,625 MW)",
        annotation_font=dict(size=10, color=AMBER),
    )
    for sname, color in SCENARIO_COLORS.items():
        fig_cap.add_trace(
            go.Scatter(
                x=sar["fore_mean"].index,
                y=fore_peak_mw[sname],
                mode="lines+markers",
                name=sname,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            )
        )
    fig_cap.add_trace(
        go.Scatter(
            x=sar["fore_mean"].index,
            y=fore_peak_mw_base,
            mode="lines",
            name="Base Forecast",
            line=dict(color=MUTED, width=1.5, dash="dot"),
        )
    )
    fig_cap.update_layout(
        **base_layout(
            height=340,
            yaxis=dict(
                range=[0, CAPACITY_MW * 1.08],
                showgrid=True,
                gridcolor=RULE,
                zeroline=False,
                linecolor=RULE,
            ),
        ),
        title=dict(
            text="Monthly Implied Peak Demand vs Grid Capacity (MW)",
            font=dict(size=13, color=NAVY),
            x=0,
        ),
        xaxis_title="Month",
        yaxis_title="MW",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    st.plotly_chart(fig_cap, use_container_width=True)

    col_g, col_h = st.columns([1.4, 1])
    with col_g:
        st.markdown(
            '<div class="sec-lbl">Exhibit 2 — Monthly Peak Demand & Reserve Margin by Scenario</div>',
            unsafe_allow_html=True,
        )
        months_list = [d.strftime("%b %Y") for d in sar["fore_mean"].index]
        peak_tbl = pd.DataFrame({
            "Month": months_list,
            "Base (MW)": fore_peak_mw_base.round(0).astype(int),
            "Conservative": fore_peak_mw["Conservative"].round(0).astype(int),
            "Base Case": fore_peak_mw["Base Case"].round(0).astype(int),
            "Aggressive": fore_peak_mw["Aggressive"].round(0).astype(int),
            "Reserve (Base)": [f"{((CAPACITY_MW / p) - 1) * 100:.1f}%" for p in fore_peak_mw_base],
        })
        st.dataframe(peak_tbl, use_container_width=True, hide_index=True)

    with col_h:
        st.markdown('<div class="sec-lbl">Exhibit 3 — Grid Stress Heatmap</div>', unsafe_allow_html=True)
        stress_data = []
        for sname in ["Conservative", "Base Case", "Aggressive"]:
            for p, m in zip(fore_peak_mw[sname], months_list):
                stress_pct = round(p / CAPACITY_MW * 100, 1)
                stress_data.append({"Scenario": sname, "Month": m, "Utilisation %": stress_pct})
        stress_df = pd.DataFrame(stress_data)
        stress_pivot = stress_df.pivot(index="Scenario", columns="Month", values="Utilisation %")
        stress_pivot = stress_pivot[months_list]
        fig_heat = go.Figure(
            go.Heatmap(
                z=stress_pivot.values,
                x=[m[:3] for m in months_list],
                y=stress_pivot.index.tolist(),
                colorscale=[
                    [0.0, "#F0F4FF"],
                    [0.70, "#C9A84C"],
                    [0.85, "#B8560A"],
                    [1.0, "#C8382A"],
                ],
                zmin=60,
                zmax=100,
                text=stress_pivot.values.round(0).astype(int),
                texttemplate="%{text}%",
                colorbar=dict(title="% Capacity", thickness=12, len=0.7),
            )
        )
        fig_heat.update_layout(
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE,
            height=220,
            margin=dict(t=20, b=30, l=100, r=20),
            font=dict(size=9.5),
            title=dict(
                text="Grid Utilisation % by Scenario & Month",
                font=dict(size=11, color=NAVY),
                x=0,
            ),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        '<div class="sec-lbl" style="margin-top:6px;">Exhibit 4 — Capacity Reserve Margin by Scenario</div>',
        unsafe_allow_html=True,
    )
    fig_res = go.Figure()
    fig_res.add_hline(
        y=15,
        line=dict(color=GREEN, width=1.5, dash="dash"),
        annotation_text="  15% Reserve Margin Target",
        annotation_font=dict(size=10, color=GREEN),
    )
    fig_res.add_hline(y=0, line=dict(color=RED, width=1, dash="dot"))
    for sname, color in SCENARIO_COLORS.items():
        margins = [round((CAPACITY_MW / p - 1) * 100, 1) for p in fore_peak_mw[sname]]
        fig_res.add_trace(
            go.Scatter(
                x=sar["fore_mean"].index,
                y=margins,
                mode="lines+markers",
                name=sname,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            )
        )
    fig_res.update_layout(
        **base_layout(height=270),
        title=dict(
            text="Forecast Reserve Margin % by Scenario (target: 15%)",
            font=dict(size=12, color=NAVY),
            x=0,
        ),
        xaxis_title="Month",
        yaxis_title="Reserve Margin (%)",
        legend=dict(orientation="h", y=-0.28, x=0),
    )
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown(
        '<div class="sec-lbl" style="margin-top:4px;">Investment Trigger Recommendations</div>',
        unsafe_allow_html=True,
    )
    ir1, ir2, ir3 = st.columns(3)
    with ir1:
        months_to_trigger = next(
            (i + 1 for i, p in enumerate(fore_peak_mw["Base Case"]) if p > TRIGGER_MW),
            None,
        )
        trigger_txt = (
            f"Month {months_to_trigger} of forecast ({sar['fore_mean'].index[months_to_trigger - 1].strftime('%b %Y')})"
            if months_to_trigger
            else "Not triggered in horizon"
        )
        st.markdown(
            f"""
        <div class="brief-navy">
          <div class="brief-head" style="color:{NAVY};">Base Case — Capacity Trigger</div>
          <div class="brief-body">
            Under base-case EV/Solar assumptions, peak demand crosses the
            <strong>{TRIGGER_MW:,} MW investment trigger</strong> at:
            <strong>{trigger_txt}</strong>.
            Reserve margin falls to {reserve_margin(fore_peak_mw['Base Case']):.1f}% at peak.
            Recommended action: begin procurement planning 18–24 months ahead.
          </div>
        </div>""",
            unsafe_allow_html=True,
        )
    with ir2:
        agg_months = stress_months(fore_peak_mw["Aggressive"])
        st.markdown(
            f"""
        <div class="brief-red">
          <div class="brief-head" style="color:{RED};">Aggressive Scenario — Grid Stress Risk</div>
          <div class="brief-body">
            Under aggressive EV adoption (30%) and commercial growth (8%),
            the grid exceeds <strong>95% utilisation in {agg_months} months</strong>
            within the forecast window. Reserve margin falls to
            {reserve_margin(fore_peak_mw['Aggressive']):.1f}%.
            Recommended action: accelerate demand response programs
            and fast-track storage procurement immediately.
          </div>
        </div>""",
            unsafe_allow_html=True,
        )
    with ir3:
        cons_reserve = reserve_margin(fore_peak_mw["Conservative"])
        st.markdown(
            f"""
        <div class="brief-green">
          <div class="brief-head" style="color:{GREEN};">Conservative — Solar Offset Opportunity</div>
          <div class="brief-body">
            Under conservative EV (5%) with strong solar incentives, reserve margin
            stays at <strong>{cons_reserve:.1f}%</strong> throughout the forecast horizon.
            High solar penetration creates mid-day over-supply risk requiring
            flexible load management and battery storage coordination.
            Recommended: invest in grid-scale storage and time-of-use pricing
            to capture the solar dividend.
          </div>
        </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85rem; padding:10px 0;'>"
    "<strong>Sherriff Abdul-Hamid</strong><br>"
    "Data Scientist · Energy Economist · Public Infrastructure Analytics<br>"
    "<a href='https://poverty360.org' target='_blank'>poverty360.org</a> · "
    "<a href='https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/' target='_blank'>LinkedIn</a> · "
    "<a href='https://github.com/S-ABDUL-AI' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
