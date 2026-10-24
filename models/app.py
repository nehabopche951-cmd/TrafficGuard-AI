
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from models.chatbot import render_chatbot

# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ETP_4_New_Data_Accidents.csv"
MODEL_PATH = BASE_DIR / "models" / "severity_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
HOTSPOT_PATH = BASE_DIR / "outputs" / "hotspots_deduplicated.csv"

st.set_page_config(
    page_title="TrafficGuard AI",
    page_icon="🚦",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(120deg, #102a43, #176b87);
    color: white;
}
.hero h1 {
    color: white;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    color: #172b4d;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #dce6f1;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #172b4d !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    date = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )
    df["Accident_Month"] = date.dt.month
    df["Accident_DayOfWeek"] = date.dt.day_of_week

    parts = df["Time_of_Accident"].astype(str).str.extract(
        r"^\s*(\d{1,2}):(\d{2})"
    )
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce")
    valid = hour.between(0, 23) & minute.between(0, 59)
    df["Accident_Hour"] = hour.where(valid)

    return df

st.markdown("""
<div class="hero">
    <h1>🚦 TrafficGuard AI</h1>
    <p>Intelligent Traffic Accident Risk Prediction & Hotspot Analysis</p>
    <p>Explore historical accident patterns and test a machine-learning severity prediction.</p>
</div>
""", unsafe_allow_html=True)

# Check required files
required_files = [DATA_PATH, MODEL_PATH, METADATA_PATH]
missing = [str(p.relative_to(BASE_DIR)) for p in required_files if not p.exists()]

if missing:
    st.error("Required project files are missing: " + ", ".join(missing))
    st.info("Check that the dataset is inside data/ and the trained model files are inside models/.")
    st.stop()

try:
    model, metadata = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load the model or dataset: {e}")
    st.stop()

features = metadata["features"]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Severity Prediction", "Accident Hotspots", "About", "AI Chatbot"]
)

# Dashboard
if page == "Dashboard":
    st.header("Project Dashboard")
    st.write("An overview of the historical accident dataset.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accident records", f"{len(df):,}")
    c2.metric(
        "Recorded severity classes",
        str(df["Accident_Severity_C"].nunique())
    )
    c3.metric(
        "Road locations",
        str(df["Accident_Location_A"].nunique())
    )

    st.subheader("Accidents by recorded severity")
    severity_counts = (
        df["Accident_Severity_C"]
        .value_counts()
        .sort_index()
    )
    st.bar_chart(severity_counts)

    st.subheader("Explore the dataset")
    st.dataframe(df.head(10), width="stretch")

# Severity prediction
elif page == "Severity Prediction":
    st.header("Accident Severity Prediction")
    st.write(
        "Enter road, location, time, and weather details to test "
        "the trained model. Use the category codes from your dataset."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            day = st.number_input(
                "Day of week code",
                min_value=1, max_value=7, value=2
            )
            month = st.number_input(
                "Accident month",
                min_value=1, max_value=12, value=9
            )
            day_of_week = st.number_input(
                "Date weekday code (Monday=0)",
                min_value=0, max_value=6, value=1
            )
            hour = st.number_input(
                "Accident hour (0–23)",
                min_value=0, max_value=23, value=13
            )
            location = st.number_input(
                "Location ID",
                min_value=0, value=2
            )

        with col2:
            chainage = st.number_input(
                "Chainage (km)",
                min_value=0.0, value=187.6, step=0.1
            )
            roadside = st.number_input(
                "Roadside code",
                min_value=0, value=1
            )
            road_feature = st.number_input(
                "Road feature code",
                min_value=0, value=2
            )
            road_condition = st.number_input(
                "Road condition code",
                min_value=0, value=3
            )
            weather = st.number_input(
                "Weather code",
                min_value=0, value=8
            )

        submitted = st.form_submit_button(
            "Predict Severity", type="primary"
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Day_of_Week": day,
            "Accident_Month": month,
            "Accident_DayOfWeek": day_of_week,
            "Accident_Hour": hour,
            "Accident_Location_A": location,
            "Accident_Location_A_Chainage_km": chainage,
            "Accident_Location_A_Chainage_km_RoadSide": roadside,
            "Road_Feature_E": road_feature,
            "Road_Condition_F": road_condition,
            "Weather_Conditions_H": weather
        }], columns=features)

        try:
            prediction = model.predict(input_data)[0]
            st.subheader("Prediction Result")
            st.metric("Predicted severity class", str(prediction))
            st.info(
                "This is a model prediction based on the entered values, "
                "not a guaranteed assessment of real-world danger. "
                "Confirm the category codes using your dataset."
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_
                probability_df = pd.DataFrame({
                    "Severity class": [str(c) for c in classes],
                    "Model probability (%)": probabilities * 100
                })
                st.subheader("Model probability by class")
                st.bar_chart(
                    probability_df.set_index("Severity class")
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Historical hotspots
elif page == "Accident Hotspots":
    st.header("Historical Accident Hotspots")
    st.write(
        "Locations with more recorded accidents in the available dataset. "
        "Counts alone do not establish the true risk rate."
    )

    if HOTSPOT_PATH.exists():
        hotspots = pd.read_csv(HOTSPOT_PATH)
        st.dataframe(hotspots.head(20), width="stretch")

        count_col = next(
            (
                c for c in hotspots.columns
                if c.lower() in [
                    "accident_count", "total_accidents",
                    "count", "number_of_accidents"
                ]
            ),
            None
        )

        if count_col:
            st.subheader("Top historical hotspots")
            chart_data = hotspots.head(10).copy()
            chart_data.index = chart_data.index.astype(str)
            st.bar_chart(chart_data.set_index(chart_data.columns[0])[count_col])
        else:
            st.caption(
                "The hotspot file is available above. Check its column "
                "names to configure a chart of accident counts."
            )
    else:
        st.warning(
            "Hotspot results were not found. Run hotspot_analysis.py "
            "first to generate outputs/hotspots_deduplicated.csv."
        )

# About
elif page == "About":
    st.header("About TrafficGuard AI")
    st.write("""
    **Objective:** Explore historical accident patterns, identify locations
    with high recorded accident counts, and demonstrate a machine-learning
    model for accident severity prediction.

    **Technologies:** Python, Pandas, Scikit-learn, Matplotlib, and Streamlit.

    **Important limitation:** This is a prototype for analysis and
    demonstration. It is not a validated live accident-warning system.
    Historical counts do not account for traffic volume or exposure, and
    predictions require independent evaluation before real-world use.
    """)
elif page == "AI Chatbot":
    render_chatbot()
