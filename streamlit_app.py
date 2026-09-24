"""Streamlit deployment entrypoint for Phnom Penh climate forecasting."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import model_is_ready, predict_next_day


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = PROJECT_ROOT / "data" / "Dataset_with_RainStatus_V5.csv"

st.set_page_config(page_title="Phnom Penh Climate Forecast", page_icon="🌦️")
st.title("🌦️ Phnom Penh Climate Forecast")
st.caption("Next-day forecast from the baseline multi-task BiLSTM model")

if not model_is_ready():
    st.error("The deployable baseline model has not been trained yet.")
    st.code("uv run python experiments/train_baseline.py", language="bash")
    st.stop()

st.write(
    "Use the included NASA weather history or upload a CSV with the same raw columns. "
    "The model uses the most recent 30 usable daily observations."
)
uploaded_file = st.file_uploader("Optional: upload recent weather-history CSV", type="csv")

try:
    weather_data = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(SAMPLE_DATA)
    st.caption(f"Using {len(weather_data):,} raw observations.")
    with st.expander("Preview input data"):
        st.dataframe(weather_data.tail(), use_container_width=True)

    if st.button("Forecast next day", type="primary"):
        with st.spinner("Generating forecast..."):
            forecast = predict_next_day(weather_data)
        temperature, rain, rainfall = st.columns(3)
        temperature.metric("Temperature", f"{forecast['temperature_c']:.1f} °C")
        rain.metric("Rain probability", f"{forecast['rain_probability']:.0%}")
        rainfall.metric("Expected rainfall", f"{forecast['rainfall_mm']:.1f} mm")
        st.success(
            f"Forecast for {forecast['forecast_date']}: "
            f"{'rain is likely' if forecast['will_rain'] else 'no rain is likely'}."
        )
except (ValueError, KeyError) as error:
    st.error(f"Could not use this weather file: {error}")
