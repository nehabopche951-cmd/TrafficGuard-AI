
from pathlib import Path

import pandas as pd
import streamlit as st


def get_hotspot_data():
    """Load available historical hotspot results."""
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "outputs" / "hotspots_deduplicated.csv"

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def answer_question(question, hotspots):
    """Provide answers using simple keyword-based rules."""
    q = question.lower().strip()

    if any(word in q for word in [
        "hello", "hi", "hey", "namaste", "नमस्ते"
    ]):
        return (
            "Hello! 👋 Welcome to TrafficGuard AI Assistant. "
            "Ask me about accident hotspots, severity prediction, "
            "road safety, or this project."
        )

    if any(word in q for word in [
        "hotspot", "accident location", "high accident",
        "most accidents", "accident-prone", "accident count",
        "black spot", "हॉटस्पॉट", "दुर्घटना"
    ]):
        if hotspots is None or hotspots.empty:
            return (
                "The hotspot results file is not available right now. "
                "Please check that outputs/hotspots_deduplicated.csv "
                "exists in the project."
            )

        count_columns = [
            col for col in hotspots.columns
            if any(term in col.lower() for term in [
                "count", "total", "frequency", "number_of_accidents"
            ])
        ]

        result = hotspots.copy()

        if count_columns:
            result = result.sort_values(
                count_columns[0], ascending=False
            )

        st.session_state["show_hotspots"] = result.head(5)

        return (
            "Here are the top recorded hotspot entries in the "
            "available historical dataset. These counts describe "
            "recorded accidents, not guaranteed future risk."
        )

    if any(word in q for word in [
        "severity", "prediction", "predict", "class",
        "classification", "गंभीरता"
    ]):
        return (
            "TrafficGuard AI uses a trained machine-learning model "
            "to predict an accident severity category from the "
            "provided inputs. The prediction is an estimate based "
            "on historical data, not a guarantee. It does not "
            "predict the exact time or location of a future accident."
        )

    if any(word in q for word in [
        "weather", "rain", "fog", "road condition", "road safety"
    ]):
        return (
            "Road safety tips: reduce speed in rain or fog, maintain "
            "a safe following distance, wear a seat belt or helmet, "
            "avoid mobile-phone use while driving, and follow road "
            "signs. Historical data can help explore associations "
            "with road and weather conditions, but does not prove "
            "that a single factor caused an accident."
        )

    if any(word in q for word in [
        "project", "trafficguard", "what can you do", "features"
    ]):
        return (
            "TrafficGuard AI offers an interactive dashboard, "
            "machine-learning-based severity prediction, historical "
            "accident hotspot analysis, and this road-safety assistant."
        )

    if any(word in q for word in [
        "thank", "thanks", "dhanyavaad", "धन्यवाद"
    ]):
        return "You're welcome! Stay safe on the road. 🚦"

    return (
        "I can help with these topics:\n\n"
        "- Historical accident hotspots and recorded counts\n"
        "- Accident severity predictions\n"
        "- Road safety and weather-related precautions\n"
        "- TrafficGuard AI project features\n\n"
        "Try asking: 'Show the top accident hotspots' or "
        "'What does severity prediction mean?'"
    )


def render_chatbot():
    st.title("🤖 TrafficGuard AI Assistant")
    st.caption(
        "A free, rule-based assistant for historical accident "
        "data and general road safety."
    )

    if "traffic_chat" not in st.session_state:
        st.session_state["traffic_chat"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi! 👋 Ask me about accident hotspots, "
                    "severity prediction, or road safety."
                ),
            }
        ]

    for message in st.session_state["traffic_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("show_hotspots"):
                st.dataframe(
                    message["show_hotspots"],
                    width="stretch",
                    hide_index=True,
                )

    if prompt := st.chat_input("Ask about traffic safety..."):
        st.session_state["traffic_chat"].append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            hotspots = get_hotspot_data()
            response = answer_question(prompt, hotspots)

            st.markdown(response)

            if st.session_state.get("show_hotspots") is not None:
                if any(word in prompt.lower() for word in [
                    "hotspot", "accident location", "high accident",
                    "most accidents", "accident-prone", "accident count",
                    "black spot", "हॉटस्पॉट", "दुर्घटना"
                ]):
                    st.dataframe(
                        st.session_state["show_hotspots"],
                        width="stretch",
                        hide_index=True,
                    )

                    st.session_state["traffic_chat"].append({
                        "role": "assistant",
                        "content": response,
                        "show_hotspots": st.session_state[
                            "show_hotspots"
                        ],
                    })
                else:
                    st.session_state["traffic_chat"].append({
                        "role": "assistant",
                        "content": response,
                    })
            else:
                st.session_state["traffic_chat"].append({
                    "role": "assistant",
                    "content": response,
                })

    if st.button("Clear chat"):
        st.session_state["traffic_chat"] = []
        st.session_state["show_hotspots"] = None
        st.rerun()
