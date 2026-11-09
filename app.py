"""
Hispanic Heritage Historical Data Explorer
Single-file Streamlit application for historical World Bank data and function analysis.
Run: python -m streamlit run app.py
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp

try:
    import xlsxwriter  # noqa: F401
except ImportError:
    xlsxwriter = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    )
except ImportError:
    SimpleDocTemplate = None

st.set_page_config(page_title="Hispanic Heritage Data Lab", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC",
    "Uruguay": "UY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns continued."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people. This is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, reported in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue", "time", "dollars", "revenue", "Suggested model: exponential growth or a polynomial trend."),
    "Engineering": ("Displacement", "time", "meters", "displacement", "Suggested model: polynomial motion model."),
    "Medical": ("Concentration", "time", "mg/L", "concentration", "Suggested model: exponential decay."),
    "Sports science": ("Position", "time", "meters", "position", "Suggested model: polynomial motion model."),
    "Biology": ("Population", "time", "organisms", "population", "Suggested model: logistic-like or exponential growth."),
    "Bioengineering": ("Signal", "time", "units", "signal", "Suggested model: sinusoidal signal."),
    "Economic forecasting": ("Output", "time", "index points", "output", "Suggested model: polynomial or exponential trend."),
    "Weather": ("Temperature", "time", "°C", "temperature", "Suggested model: sinusoidal seasonal cycle."),
    "Custom": ("Output", "input", "units", "output", "Describe your own relationship."),
}
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"

def world_bank_data(country: str, indicator: str, start: int, end: int) -> pd.DataFrame:
    url = WB_URL.format(country=country, indicator=indicator, start=start, end=end)
    response = requests.get(url, timeout=25, headers={"User-Agent": "Hispanic-Heritage-Data-Explorer/1.0"})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("The World Bank response did not contain an observation list.")
    rows = []
    for item in payload[1]:
        if item and item.get("date") and item.get("value") is not None:
            try:
                rows.append({"Year": int(item["date"]), "Value": float(item["value"])})
            except (TypeError, ValueError):
                continue
    frame = pd.DataFrame(rows, columns=["Year", "Value"])
    if not frame.empty:
        frame = frame.sort_values("Year").drop_duplicates("Year", keep="last").reset_index(drop=True)
    return frame

@st.cache_data(ttl=3600, show_spinner=False)
def cached_world_bank_data(country: str, indicator: str, start: int, end: int) -> pd.DataFrame:
    return world_bank_data(country, indicator, start, end)

def validate_data(frame: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"Year", "Value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["Year", "Value"]), ["The table must contain Year and Value columns."]
    out = frame[["Year", "Value"]].copy()
    years = pd.to_numeric(out["Year"], errors="coerce")
    values = pd.to_numeric(out["Value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Years must be whole numbers.")
    if years.duplicated().any():
        errors.append("Each year must appear only once.")
    if values.isna().any():
        errors.append("Values must be numeric and cannot be blank.")
    if np.isinf(values.dropna()).any():
        errors.append("Values must be finite.")
    out["Year"], out["Value"] = years, values
    out = out.dropna().sort_values("Year").reset_index(drop=True)
    if indicator_name in ("Life expectancy at birth",) and ((out.Value < 0) | (out.Value > 125)).any():
        errors.append("Life expectancy values should be between 0 and 125 years.")
    if indicator_name in ("Unemployment, national estimate", "Central government debt") and (out.Value < 0).any():
        errors.append("This indicator should not contain negative percentages.")
    return out, errors

def fit_polynomial(frame: pd.DataFrame, degree: int):
    if len(frame) < degree + 1 or frame.Year.nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years.")
    base_year = int(frame.Year.min())
    t = frame.Year.to_numpy(dtype=float) - base_year
    y = frame.Value.to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    model = np.poly1d(coeff)
    fitted = model(t)
    residuals = y - fitted
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(residuals ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    return base_year, coeff, model, fitted, residuals, r2, mae

def make_sympy_expression(text: str, x_symbol):
    cleaned = text.replace("^", "**")
    allowed = {"x": x_symbol, "t": x_symbol, "pi": sp.pi, "E": sp.E,
               "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs}
    if len(cleaned) > 180:
        raise ValueError("Function expression is too long.")
    try:
        expr = sp.sympify(cleaned, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not read that function. Try notation such as 2*x^2 + 3*x - 1.") from exc
    if expr.free_symbols - {x_symbol}:
        raise ValueError("Use only x (or t) as the independent variable.")
    return expr

def safe_numeric(expr, x_symbol, values):
    fn = sp.lambdify(x_symbol, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        result = np.asarray(fn(values), dtype=float)
    if result.ndim == 0:
        result = np.full(np.asarray(values).shape, float(result))
    return result

def fmt(value, digits=5):
    try:
        return f"{float(value):,.{digits}g}"
    except Exception:
        return str(value)

def build_excel(summary: dict, data: pd.DataFrame, results: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        results.to_excel(writer, sheet_name="Calculated Years", index=False)
        for sheet in writer.sheets.values():
            sheet.set_column(0, 10, 20)
    return output.getvalue()

def build_pdf(summary: dict, data: pd.DataFrame, results: pd.DataFrame, chart_bytes: bytes) -> bytes:
    if SimpleDocTemplate is None:
        raise RuntimeError("ReportLab is not installed.")
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
             Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]), Spacer(1, 10)]
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(RLImage(io.BytesIO(chart_bytes), width=7.1*inch, height=3.6*inch))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = [["Year", "Value"]] + [[str(int(r.Year)), fmt(r.Value)] for r in data.head(18).itertuples()]
    table = Table(preview, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                               ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
                               ("FONTSIZE", (0,0), (-1,-1), 8)]))
    story.append(table)
    if not results.empty:
        story.append(Paragraph("Calculated year results", styles["Heading2"]))
        rows = [["Year", "Estimate", "Classification"]] + [[str(r.Year), fmt(r.Estimate), r.Classification] for r in results.itertuples()]
        table2 = Table(rows, repeatRows=1)
        table2.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                    ("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 8)]))
        story.append(table2)
    doc.build(story)
    return output.getvalue()

def world_bank_link(country_code, indicator_code):
    return f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"

st.title("📊 Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab • Historical data, mathematical models, and interactive function analysis")
st.info("Brazil is included as a Latin American comparison country. The selected World Bank indicator is retrieved as published; missing values are not invented.")

with st.sidebar:
    st.header("Historical data")
    country_name = st.selectbox("Country", list(COUNTRIES))
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_name = st.selectbox("Compare with another country", ["None"] + [c for c in COUNTRIES if c != country_name])
    printer_mode = st.checkbox("Printer-friendly view", value=False)

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
country_code = COUNTRIES[country_name]
indicator_code, units, definition = INDICATORS[indicator_name]
retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"**Research window:** {start_year}–{end_year} inclusive. The API may return fewer than 70 observations.")

try:
    with st.spinner("Retrieving official World Bank observations..."):
        source_data = cached_world_bank_data(country_code, indicator_code, start_year, end_year)
    if source_data.empty:
        st.warning("The World Bank API returned no non-null observations for this country, indicator, and period.")
except requests.Timeout:
    st.error("The World Bank request timed out. Check your connection and try again.")
    source_data = pd.DataFrame(columns=["Year", "Value"])
except requests.RequestException as exc:
    st.error(f"World Bank request failed: {exc}")
    source_data = pd.DataFrame(columns=["Year", "Value"])
except (ValueError, TypeError) as exc:
    st.error(f"Could not interpret the World Bank response: {exc}")
    source_data = pd.DataFrame(columns=["Year", "Value"])

if "working_data" not in st.session_state:
    st.session_state.working_data = source_data.copy()
if "data_signature" not in st.session_state:
    st.session_state.data_signature = (country_code, indicator_code)
if st.session_state.data_signature != (country_code, indicator_code):
    st.session_state.working_data = source_data.copy()
    st.session_state.data_signature = (country_code, indicator_code)
    st.session_state.pop("calc_results", None)

st.subheader(f"{country_name}: {indicator_name}")
st.write(definition)
st.caption(f"Units: {units} • World Bank indicator: {indicator_code} • Retrieved: {retrieved_at}")
st.markdown(f"Source: [{world_bank_link(country_code, indicator_code)}]({world_bank_link(country_code, indicator_code)})")
st.metric("Available observations", len(source_data))
if not source_data.empty:
    st.caption(f"Actual coverage: {int(source_data.Year.min())}–{int(source_data.Year.max())} ({source_data.Year.nunique()} distinct years)")
else:
    st.caption("Actual coverage: no observations available.")

st.subheader("Validate and edit working data")
st.caption("The source data remains unchanged. Edits below are a separate scenario and are clearly marked.")
left, right = st.columns([1, 1])
with left:
    if st.button("Reset to source data"):
        st.session_state.working_data = source_data.copy()
        st.session_state.pop("calc_results", None)
with right:
    edited_flag = not st.session_state.working_data.equals(source_data)
    st.write("**Status:** " + ("User-modified scenario" if edited_flag else "Original source data"))
edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True,
                        column_config={"Year": st.column_config.NumberColumn("Year", step=1),
                                       "Value": st.column_config.NumberColumn("Value", format="%.6f")}, key="data_editor")
valid_data, validation_errors = validate_data(edited, indicator_name)
st.session_state.working_data = edited.copy()
for err in validation_errors:
    st.error(err)
if len(valid_data) < degree + 1:
    st.warning(f"At least {degree + 1} valid distinct observations are needed for a degree-{degree} fit.")

model_ready = not validation_errors and len(valid_data) >= degree + 1
model_info = None
if model_ready:
    try:
        model_info = fit_polynomial(valid_data, degree)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(f"Model could not be fit: {exc}")
        model_ready = False

if model_ready:
    base_year, coeff, model, fitted, residuals, r2, mae = model_info
    y = valid_data.Value.to_numpy(dtype=float)
    years = valid_data.Year.to_numpy(dtype=int)
    st.subheader("Polynomial model")
    st.latex("t = \\text{year} - " + str(base_year))
    terms = []
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        terms.append(f"{c:.10g}" + (f"t^{{{power}}}" if power > 1 else ("t" if power == 1 else "")))
    equation = "f(t) = " + (" + ".join(terms) if terms else "0")
    st.latex(sp.latex(sp.sympify(equation.replace("f(t) = ", "").replace("^", "**"))) if False else equation)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (in-sample)", fmt(r2, 5))
    m2.metric("MAE (in-sample)", f"{fmt(mae)} {units}")
    m3.metric("Observations", len(valid_data))
    m4.metric("Mean", f"{fmt(valid_data.Value.mean())} {units}")
    st.caption(f"Base year: {base_year} • Degree: {degree} • Historical range: {years.min()}–{years.max()}. Metrics describe fit to these same observations, not future accuracy.")
    residual_table = pd.DataFrame({"Year": years, "Actual": y, "Fitted": fitted, "Residual (actual - fitted)": residuals})
    with st.expander("Residuals and in-sample fitted values"):
        st.dataframe(residual_table, use_container_width=True)
    if len(valid_data) >= max(10, degree + 4):
        split = max(degree + 1, int(len(valid_data) * 0.8))
        train = valid_data.iloc[:split]
        test = valid_data.iloc[split:]
        if len(test):
            try:
                b2, c2, m2hold, *_ = fit_polynomial(train, degree)
                hold_preds = m2hold(test.Year.to_numpy(dtype=float) - b2)
                hold_mae = float(np.mean(np.abs(test.Value.to_numpy() - hold_preds)))
                st.caption(f"Chronological holdout MAE (later {len(test)} years; separate from in-sample MAE): {fmt(hold_mae)} {units}.")
            except Exception:
                pass

    st.subheader("Calculate estimates")
    c1, c2 = st.columns(2)
    with c1:
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(max(1950, min(2100, years.max()))), step=1)
        if st.button("Calculate value"):
            st.session_state.single_result = int(single_year)
    with c2:
        batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            range_col1, range_col2, range_col3 = st.columns(3)
            with range_col1:
                batch_start = st.number_input("Start", 1950, 2100, int(max(1950, min(2100, years.min()))), key="batch_start")
            with range_col2:
                batch_end = st.number_input("End", 1950, 2100, int(max(1950, min(2100, years.max()))), key="batch_end")
            with range_col3:
                batch_step = st.number_input("Step", 1, 10, 1, key="batch_step")
            batch_text = ""
        else:
            batch_text = st.text_input("Comma-separated years", f"{int(years.min())}, {int(years.max())}")
            batch_start, batch_end, batch_step = 0, 0, 1
        if st.button("Calculate batch"):
            requested, invalid = [], []
            if batch_mode == "Year range":
                if batch_start > batch_end:
                    invalid.append("Start year must be less than or equal to end year.")
                else:
                    requested = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
            else:
                for part in batch_text.split(","):
                    try:
                        val = int(part.strip())
                        if 1950 <= val <= 2100:
                            requested.append(val)
                        else:
                            invalid.append(f"{part.strip()} is outside 1950–2100.")
                    except ValueError:
                        if part.strip():
                            invalid.append(f"{part.strip()} is not a whole-number year.")
            requested = list(dict.fromkeys(requested))
            if invalid:
                st.warning(" ".join(invalid))
            if requested:
                vals = model(np.array(requested, dtype=float) - base_year)
                results = pd.DataFrame({"Year": requested, "Estimate": vals,
                                        "Classification": ["Within fitted historical range" if years.min() <= yr <= years.max() else "Outside fitted range (extrapolation)" for yr in requested]})
                st.session_state.calc_results = results
                st.session_state.calc_signature = (country_code, indicator_code, degree, tuple(valid_data.Year), tuple(valid_data.Value))
    if "single_result" in st.session_state:
        yr = st.session_state.single_result
        estimate = float(model(yr - base_year))
        actual_match = valid_data.loc[valid_data.Year == yr, "Value"]
        st.success(f"{yr}: estimated {fmt(estimate)} {units}. " +
                   (f"An actual observation also exists: {fmt(actual_match.iloc[0])} {units}." if len(actual_match) else "No actual observation is recorded for this year.") +
                   (" Within the fitted historical range." if years.min() <= yr <= years.max() else " Outside the fitted historical range: extrapolation."))
    results = st.session_state.get("calc_results", pd.DataFrame(columns=["Year", "Estimate", "Classification"]))
    current_sig = (country_code, indicator_code, degree, tuple(valid_data.Year), tuple(valid_data.Value))
    if st.session_state.get("calc_signature") != current_sig:
        results = pd.DataFrame(columns=["Year", "Estimate", "Classification"])
        st.session_state.calc_results = results
    if not results.empty:
        st.dataframe(results.assign(Units=units), use_container_width=True)

    st.subheader("Historical data and fitted curve")
    x_smooth = np.linspace(years.min(), years.max() + horizon, 500)
    y_smooth = model(x_smooth - base_year)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(years, y, label="Observed data")
    ax.plot(x_smooth[x_smooth <= years.max()], y_smooth[x_smooth <= years.max()], label="Fitted curve")
    if horizon > 0:
        ax.plot(x_smooth[x_smooth > years.max()], y_smooth[x_smooth > years.max()], linestyle="--", label="Extrapolated segment")
    ax.axvline(years.max(), linestyle=":", label="Historical boundary")
    marker_years = years[::sample_step]
    ax.scatter(marker_years, model(marker_years - base_year), marker="x", label=f"Model markers every {sample_step} observations")
    if not results.empty:
        ax.scatter(results.Year, results.Estimate, marker="D", label="Batch estimates")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({units})")
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
    chart_bytes = chart_buffer.getvalue()
    plt.close(fig)

    st.subheader("Interpretation and limitations")
    st.write(f"The polynomial summarizes the pattern in {len(valid_data)} selected observations from {int(years.min())} to {int(years.max())}. Its in-sample mean absolute error is {fmt(mae)} {units}.")
    st.write("A polynomial can bend sharply outside the observed years. Treat future or distant-year values as mathematical extrapolations, not official forecasts or guaranteed outcomes. Residuals and MAE describe typical in-sample error; they are not a validated 95% prediction interval.")
    if horizon and np.any(~np.isfinite(y_smooth)) or (horizon and np.nanmax(np.abs(y_smooth)) > max(1e6, np.nanmax(np.abs(y))*1000)):
        st.warning("The extrapolated curve reaches unusually large values. Review it cautiously; values are not clipped.")
    st.write("Historical associations do not establish causes. The World Bank may revise observations, and country coverage can differ.")

    st.subheader("Compare another country")
    if compare_name != "None":
        try:
            other = cached_world_bank_data(COUNTRIES[compare_name], indicator_code, start_year, end_year)
            if other.empty:
                st.warning(f"No observations returned for {compare_name}.")
            else:
                fig2, ax2 = plt.subplots(figsize=(9, 3.5))
                ax2.plot(valid_data.Year, valid_data.Value, marker="o", label=country_name)
                ax2.plot(other.Year, other.Value, marker="s", label=compare_name)
                ax2.set_xlabel("Year"); ax2.set_ylabel(units); ax2.legend(); ax2.grid(True, alpha=0.25)
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption(f"{country_name}: {valid_data.Year.min()}–{valid_data.Year.max()} ({len(valid_data)} values); {compare_name}: {other.Year.min()}–{other.Year.max()} ({len(other)} values). Missing years create coverage gaps.")
        except Exception as exc:
            st.warning(f"Comparison data could not be retrieved: {exc}")

    st.subheader("Function contexts and calculus lab")
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    context = st.selectbox("Context", list(CONTEXTS))
    default_y, default_x, default_yunit, _, suggestion = CONTEXTS[context]
    context_description = st.text_input("Context description", f"{context}: explore how {default_y.lower()} changes with {default_x}.")
    x_name = st.text_input("Independent variable name", default_x)
    y_name = st.text_input("Dependent variable name", default_y)
    x_unit = st.text_input("Independent variable units", "years" if context != "Custom" else "input units")
    y_unit = st.text_input("Dependent variable units", default_yunit)
    function_choice = st.radio("Function source", ["World Bank curve of best fit", "Suggested function", "Build with parameters", "Enter my own function"], horizontal=False)
    suggested = {
        "Business": "1000*exp(0.08*x)", "Engineering": "2*x^2 + 3*x",
        "Medical": "100*exp(-0.25*x)", "Sports science": "-4.9*x^2 + 12*x",
        "Biology": "20*exp(0.12*x)", "Bioengineering": "3*sin(2*pi*x/5)",
        "Economic forecasting": "100 + 4*x - 0.2*x^2", "Weather": "15 + 10*cos(2*pi*x/12)",
        "Custom": "2*x^2 + 3*x - 1",
    }
    param_a = st.number_input("Parameter a", value=2.0, key="param_a")
    param_b = st.number_input("Parameter b", value=3.0, key="param_b")
    param_c = st.number_input("Parameter c", value=1.0, key="param_c")
    if function_choice == "World Bank curve of best fit":
        function_text = equation.replace("t", "x")
        function_text = function_text.replace("f(x) = ", "")
        # Model uses x as calendar year offset from base_year.
        function_expr_text = function_text.replace("x", "(x-" + str(base_year) + ")")
    elif function_choice == "Suggested function":
        function_expr_text = suggested[context]
    elif function_choice == "Build with parameters":
        function_expr_text = f"{param_a}*x^2 + {param_b}*x + {param_c}"
    else:
        function_expr_text = st.text_input("Your function (use x; examples: 2*x^2+3*x-1, exp(-x), sin(x))", "2*x^2 + 3*x - 1")
    st.caption(suggestion)
    x_symbol = sp.Symbol("x", real=True)
    try:
        expr = make_sympy_expression(function_expr_text, x_symbol)
        deriv1, deriv2 = sp.diff(expr, x_symbol), sp.diff(expr, x_symbol, 2)
        st.markdown(f"**Function:** `{sp.sstr(expr)}`")
        st.latex("f'(x) = " + sp.latex(deriv1))
        st.latex("f''(x) = " + sp.latex(deriv2))
        lo_default, hi_default = 0.0, 10.0
        interval_col1, interval_col2 = st.columns(2)
        with interval_col1:
            interval_lo = st.number_input("Analysis interval start", value=lo_default)
        with interval_col2:
            interval_hi = st.number_input("Analysis interval end", value=hi_default)
        if interval_lo >= interval_hi:
            st.warning("The analysis interval start must be less than its end.")
        else:
            point = st.slider("Moving input point", float(interval_lo), float(interval_hi), float((interval_lo+interval_hi)/2))
            sec_col1, sec_col2 = st.columns(2)
            with sec_col1:
                sec_a = st.number_input("Average-rate interval start", value=float(interval_lo))
            with sec_col2:
                sec_b = st.number_input("Average-rate interval end", value=float(interval_hi))
            show_f = st.checkbox("Show f(x)", True)
            show_d1 = st.checkbox("Show f'(x)", True)
            show_d2 = st.checkbox("Show f''(x)", False)
            try:
                fpoint = float(safe_numeric(expr, x_symbol, np.array([point]))[0])
                slope = float(safe_numeric(deriv1, x_symbol, np.array([point]))[0])
                curvature = float(safe_numeric(deriv2, x_symbol, np.array([point]))[0])
                if not np.isfinite([fpoint, slope, curvature]).all():
                    raise ValueError("Function or derivative is undefined at the selected point.")
                st.write(f"At x = {fmt(point)} {x_unit}: f(x) = {fmt(fpoint)} {y_unit}; f'(x) = {fmt(slope)} {y_unit}/{x_unit}; f''(x) = {fmt(curvature)} {y_unit}/{x_unit}².")
                st.write(f"Context: {context_description} At this point, the rate of change is {fmt(slope)} {y_unit}/{x_unit}, and that rate is changing at {fmt(curvature)} {y_unit}/{x_unit}².")
                if sec_a != sec_b:
                    fa, fb = safe_numeric(expr, x_symbol, np.array([sec_a, sec_b]))
                    avg_rate = (fb-fa)/(sec_b-sec_a)
                    st.write(f"Average rate of change from {sec_a:g} to {sec_b:g}: {fmt(avg_rate)} {y_unit}/{x_unit}.")
                else:
                    st.warning("Choose different endpoints for average rate of change.")
                grid = np.linspace(interval_lo, interval_hi, 600)
                vals = safe_numeric(expr, x_symbol, grid)
                dvals = safe_numeric(deriv1, x_symbol, grid)
                ddvals = safe_numeric(deriv2, x_symbol, grid)
                fig3, ax3 = plt.subplots(figsize=(9, 4))
                if show_f: ax3.plot(grid, vals, label="f(x)")
                if show_d1: ax3.plot(grid, dvals, label="f'(x)")
                if show_d2: ax3.plot(grid, ddvals, label="f''(x)")
                ax3.scatter([point], [fpoint], label="Current point")
                tangent_x = np.array([point - (interval_hi-interval_lo)*0.15, point + (interval_hi-interval_lo)*0.15])
                if show_f: ax3.plot(tangent_x, fpoint + slope*(tangent_x-point), linestyle="--", label="Tangent line")
                if sec_a != sec_b:
                    sec_x = np.array([sec_a, sec_b])
                    ax3.plot(sec_x, [fa, fb], linestyle=":", label="Secant line")
                ax3.axhline(0, linewidth=0.6); ax3.grid(True, alpha=0.2); ax3.legend()
                ax3.set_xlabel(f"{x_name} ({x_unit})")
                ax3.set_ylabel(f"{y_name} / derivative values")
                st.pyplot(fig3)
                plt.close(fig3)
                if course == "Calculus":
                    critical = []
                    try:
                        roots = sp.solve(sp.Eq(deriv1, 0), x_symbol)
                        for root in roots:
                            if root.is_real is True:
                                rv = float(root)
                                if interval_lo <= rv <= interval_hi:
                                    critical.append(rv)
                    except Exception:
                        pass
                    st.markdown("**Critical-point and interval analysis**")
                    if critical:
                        for rv in critical:
                            eps = max(1e-4, (interval_hi-interval_lo)*1e-4)
                            left_s = float(safe_numeric(deriv1, x_symbol, np.array([rv-eps]))[0])
                            right_s = float(safe_numeric(deriv1, x_symbol, np.array([rv+eps]))[0])
                            label = "local minimum" if left_s < 0 < right_s else ("local maximum" if left_s > 0 > right_s else "stationary point; no sign-change extremum confirmed")
                            st.write(f"x ≈ {fmt(rv)}: {label}.")
                    else:
                        st.write("No real critical points were found symbolically in this interval.")
                    st.write("Increasing/decreasing intervals can be read from the sign of f'(x); concavity is determined by the sign of f''(x).")
                    try:
                        inf_roots = sp.solve(sp.Eq(deriv2, 0), x_symbol)
                        confirmed = []
                        for root in inf_roots:
                            if root.is_real is True:
                                rv = float(root)
                                if interval_lo < rv < interval_hi:
                                    eps = max(1e-4, (interval_hi-interval_lo)*1e-4)
                                    dl = float(safe_numeric(deriv2, x_symbol, np.array([rv-eps]))[0])
                                    dr = float(safe_numeric(deriv2, x_symbol, np.array([rv+eps]))[0])
                                    if dl*dr < 0: confirmed.append(rv)
                        st.write("Verified inflection point x-values: " + (", ".join(fmt(v) for v in confirmed) if confirmed else "none confirmed by a concavity sign change."))
                    except Exception:
                        st.write("Inflection analysis could not be solved exactly for this function.")
                    st.caption("Numerical plotting and root checks are approximations. Absolute extrema on a closed interval require comparing endpoint values with all interior critical points; inspect endpoints as well.")
                elif course == "Precalculus":
                    st.write("Turning points and concavity can be explored through how the graph changes direction and how average rates change.")
                else:
                    st.write("Domain and range depend on the selected interval. Compare average rates over different intervals and observe end behavior.")
                if context == "Engineering" or context == "Sports science":
                    st.caption("If the dependent variable represents position, its first derivative can represent velocity and its second derivative acceleration, provided the units are consistent.")
            except (ValueError, TypeError, OverflowError) as exc:
                st.error(f"Could not evaluate this function at the selected point: {exc}")
    except ValueError as exc:
        st.error(str(exc))

    st.subheader("Interactive visual simulation")
    sim_type = st.selectbox("Simulation visual", ["Moving object", "Tank / concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator", "Walking / prosthetic-leg application"])
    st.caption("The simulation maps the selected function's value to a visual bar/marker. The scale is normalized to the visible analysis interval; it is illustrative, not a physical or medical simulation.")
    if interval_lo < interval_hi:
        vals = safe_numeric(expr, x_symbol, np.linspace(interval_lo, interval_hi, 200))
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals):
            vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
            norm = 0.5 if vmax == vmin else float(np.clip((fpoint-vmin)/(vmax-vmin), 0, 1))
            st.progress(norm, text=f"{sim_type}: normalized indicator {norm:.0%} of displayed scale")
            st.caption(f"Current function value: {fmt(fpoint)} {y_unit}; displayed range: {fmt(vmin)} to {fmt(vmax)} {y_unit}.")
    play_col1, play_col2, play_col3 = st.columns(3)
    with play_col1:
        if st.button("▶ Play one step"):
            st.session_state.sim_point = min(interval_hi, point + max((interval_hi-interval_lo)/20, 0.01))
    with play_col2:
        if st.button("Ⅱ Pause"):
            st.session_state.sim_point = point
    with play_col3:
        if st.button("↺ Reset"):
            st.session_state.sim_point = (interval_lo+interval_hi)/2
    if "sim_point" in st.session_state:
        st.caption(f"Next simulation point: {fmt(st.session_state.sim_point)}. Press Play one step again to advance.")
    speed = st.select_slider("Playback step size", options=[0.5, 1, 1.5, 2, 3], value=1.0)
    st.caption(f"Step-size multiplier: {speed}×. Streamlit reruns on each button press; this is step-based playback, not continuous animation.")

    summary = {
        "Country": country_name, "Indicator": indicator_name, "Indicator code": indicator_code,
        "Units": units, "Coverage": f"{years.min()}–{years.max()}",
        "Observations": len(valid_data), "Degree": degree, "Base year": base_year,
        "Equation": equation, "In-sample R-squared": r2, "In-sample MAE": mae,
        "Data status": "User-modified scenario" if edited_flag else "Original source data",
        "Retrieved": retrieved_at,
    }
    st.subheader("Export current work")
    st.download_button("Download working data CSV", valid_data.to_csv(index=False).encode("utf-8"), "working_data.csv", "text/csv")
    try:
        xlsx_bytes = build_excel(summary, valid_data, results)
        st.download_button("Download Excel workbook", xlsx_bytes, "historical_data_explorer.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        st.error(f"Excel export failed: {exc}")
    try:
        pdf_bytes = build_pdf(summary, valid_data, results, chart_bytes)
        st.download_button("Download PDF report", pdf_bytes, "historical_data_explorer_report.pdf", "application/pdf")
    except Exception as exc:
        st.error(f"PDF export failed: {exc}")
else:
    st.info("Once the working table contains enough valid observations for the selected degree, the model, calculus lab, and exports will appear.")

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"{country_name} • {indicator_name} • {len(valid_data)} valid observations • {units}")
    st.write("Use the PDF report download for a print-ready report with chart and calculated results.")

with st.expander("Source and mathematical notes"):
    st.markdown("""
    - Historical observations are retrieved from the official World Bank Indicators API.
    - The requested date window is current year minus 70 through current year minus 1. Actual coverage depends on available non-null values.
    - Polynomial fitting uses centered time, `t = year − base year`, to improve numerical conditioning.
    - In-sample R² and MAE summarize fit to the same observations used to estimate coefficients.
    - Polynomial extrapolation can become implausible outside the observed range. It is not a forecast guarantee.
    - No missing values are replaced with random or fabricated values.
    - The app's interactive symbolic analysis uses SymPy. Numerical roots and graph sampling are approximations; symbolic solutions may not exist in closed form.
    """)
