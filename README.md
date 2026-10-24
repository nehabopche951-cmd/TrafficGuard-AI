# TrafficGuard AI 🚦

### AI-Powered Traffic Accident Severity Prediction & Hotspot Analysis

## 📌 About the Project
TrafficGuard AI is a web application designed to analyze historical traffic accident data, predict accident severity, and identify accident hotspots.

It helps users explore accident patterns and understand which locations have recorded more accidents, supporting better traffic safety planning.

## 🎯 Problem Statement
Road accidents are a major safety concern. Analyzing accident records manually can make it difficult to identify accident-prone locations and understand accident severity patterns.

## 💡 Our Solution
TrafficGuard AI uses machine learning and historical accident data to provide:

- **Accident Severity Prediction:** Predicts a severity category based on selected road, weather, location, and time-related information.
- **Hotspot Analysis:** Identifies locations with higher recorded accident counts.
- **Interactive Dashboard:** Displays key statistics and visualizations.
- **Data-Based Insights:** Helps users explore historical accident patterns.

## ⚙️ How It Works
1. Historical accident data is loaded into the application.
2. A trained machine learning model predicts an accident severity category for the provided inputs.
3. The application summarizes recorded accident hotspots.
4. Users explore the results through the dashboard.

## 🖥️ Application Pages
- Dashboard
- Severity Prediction
- Accident Hotspots
- About

## 📊 Dataset
The project uses historical traffic accident data containing information about accident severity, location, time, road conditions, and weather.

Dataset source: [Zenodo – Traffic Accident Dataset](https://zenodo.org/records/16946653)

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Machine Learning

## ▶️ Run the Project Locally
1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a Python virtual environment.
4. Install dependencies:

   `pip install -r requirements.txt`

5. Start the application from the project root:

   `streamlit run models/app.py`

6. Open the local URL displayed in the terminal.

## ⚠️ Limitations
Predictions are based on historical data and the trained model. Hotspot counts show recorded accidents, not guaranteed future risk. Results should not be treated as emergency guidance or as a substitute for professional traffic safety assessments.

## 👥 Project
**Project Name:** TrafficGuard AI  
**Purpose:** Traffic accident severity prediction and hotspot analysis

