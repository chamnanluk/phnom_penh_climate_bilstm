from io import StringIO

import pandas as pd
import streamlit as st

from src.inference import ArtifactError, ValidationError
from src.inference.app_service import predict_from_input_dataframe


st.set_page_config(page_title="Phnom Penh Climate Prediction", page_icon="🌦️")


def parse_input_dataframe(uploaded_file, pasted_csv_text: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if pasted_csv_text and pasted_csv_text.strip():
        return pd.read_csv(StringIO(pasted_csv_text.strip()))
    raise ValidationError("Please upload a CSV file or paste CSV data before prediction.")


def run_prediction_from_dataframe(df: pd.DataFrame) -> dict:
    return predict_from_input_dataframe(df)


def main():
    st.title("Phnom Penh Climate Prediction App")
    st.write(
        "Upload recent daily weather data and run one-step-ahead climate prediction "
        "with the trained baseline multi-task BiLSTM model."
    )

    st.info(
        "This app only performs prediction. It does not retrain the model. "
        "If artifacts are missing, follow the README section 'Deploy the prediction app'."
    )

    with st.expander("Required CSV columns", expanded=False):
        st.code(
            "Date, PRECTOTCORR, WS2M, T2M_RANGE, T2M_MAX, T2M_MIN, "
            "PS, ALLSKY_SFC_SW_DWN, RH2M, T2MDEW"
        )

    input_mode = st.radio("Input mode", ["Upload CSV", "Paste CSV text"])

    uploaded_file = None
    pasted_csv_text = ""

    if input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload weather CSV", type=["csv"])
    else:
        pasted_csv_text = st.text_area(
            "Paste CSV data",
            height=220,
            placeholder="Date,PRECTOTCORR,WS2M,T2M_RANGE,T2M_MAX,T2M_MIN,PS,ALLSKY_SFC_SW_DWN,RH2M,T2MDEW\n...",
        )

    if st.button("Predict next day"):
        try:
            raw_df = parse_input_dataframe(uploaded_file, pasted_csv_text)
            result = run_prediction_from_dataframe(raw_df)

            st.success(f"Prediction generated for {result['prediction_for_date']}.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature (°C)", f"{result['temperature_c']:.2f}")
            c2.metric("Rain probability", f"{result['rain_probability'] * 100:.1f}%")
            c3.metric("Rain status", result["rain_status"])
            st.metric("Predicted rainfall amount (mm)", f"{result['rainfall_amount_mm']:.2f}")

        except ValidationError as exc:
            st.error(f"Input validation error: {exc}")
        except ArtifactError as exc:
            st.error(f"Model artifact error: {exc}")
        except Exception:
            st.error(
                "Unexpected error while running prediction. "
                "Please verify your input and artifact files, then try again."
            )


if __name__ == "__main__":
    main()
