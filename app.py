"""
Streamlit control tower for the Denver Trip Generator models.

Capabilities wired to notebook artifacts:
- Address embeddings (SentenceTransformer) -> distance regression (Keras).
- Booking risk head (Keras) -> probability of Performed vs NoShow.
- Route clustering (KMeans) to place trip into an existing run.
- Address suggestions drawn from the CSV catalog for call-taker speed.

Run with: `streamlit run app.py`
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Fixed feature order used to train the booking status model
STATUS_FEATURE_ORDER = [
    "BookingId",
    "ClientId",
    "SchTime",
    "SchLate",
    "PassCount",
    "DayofMonth",
    "Month",
    "le_Purpose",
    "le_FareTypeAbbr",
    "le_DayOfWeek",
]


# -----------------------------
# Cached loaders
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_sentence_model() -> SentenceTransformer:
    import torch

    torch_load = torch.load

    def _torch_load_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return torch_load(*args, **kwargs)

    torch.load = _torch_load_cpu  # type: ignore
    try:
        return joblib.load(MODELS_DIR / "addresses_model_embeddings.joblib")
    finally:
        torch.load = torch_load  # restore original


@st.cache_resource(show_spinner=False)
def load_distance_model():
    return keras.models.load_model(MODELS_DIR / "distance_model.h5", compile=False)


@st.cache_resource(show_spinner=False)
def load_status_model():
    return keras.models.load_model(MODELS_DIR / "booking_status_model.h5", compile=False)


@st.cache_resource(show_spinner=False)
def load_route_model():
    return joblib.load(MODELS_DIR / "route_clustering_model.joblib")


@st.cache_data(show_spinner=False)
def load_encoders_and_mapping():
    encoders = joblib.load(MODELS_DIR / "booking_label_encoders.joblib")
    mapping_df = pd.read_csv(MODELS_DIR / "booking_le_mapping.csv")
    return encoders, mapping_df


@st.cache_data(show_spinner=False)
def load_address_catalog(max_rows: int = 300_000) -> List[str]:
    """
    Pull a slice of the address CSV for auto-suggest.
    Loading the full 1.2M rows stays possible by bumping max_rows.
    """
    df = pd.read_csv(MODELS_DIR / "addresses.csv", nrows=max_rows)
    catalog = pd.unique(pd.concat([df["PickupAddress"], df["DropoffAddress"]], ignore_index=True))
    return catalog.tolist()


# -----------------------------
# Helpers
# -----------------------------

def time_to_seconds(t: time | str) -> int:
    if isinstance(t, str):
        t = datetime.strptime(t.strip(), "%H:%M").time()
    return t.hour * 3600 + t.minute * 60 + t.second


def suggest_addresses(query: str, catalog: List[str], limit: int = 8) -> list[str]:
    q = query.strip().lower()
    if len(q) < 3:
        return []
    return [addr for addr in catalog if q in addr.lower()][:limit]


@st.cache_data(show_spinner=False)
def embed_address(addr: str) -> np.ndarray:
    model = load_sentence_model()
    return model.encode([addr], show_progress_bar=False, convert_to_numpy=True, device="cpu")[0]


def build_embedding_pair(pickup: str, dropoff: str) -> np.ndarray:
    v1 = embed_address(pickup)
    v2 = embed_address(dropoff)
    return np.hstack([v1, v2]).reshape(1, -1)


def predict_distance(pickup: str, dropoff: str) -> float:
    pair = build_embedding_pair(pickup, dropoff)
    model = load_distance_model()
    pred = model.predict(pair, verbose=0)
    return float(pred[0][0])


def predict_route_cluster(pickup: str, dropoff: str) -> int:
    pair = build_embedding_pair(pickup, dropoff)
    model = load_route_model()
    label = model.predict(pair)
    return int(label[0])


def encode_status_features(
    ride_date: date,
    sched_time: time,
    pickup_window_mins: int,
    purpose: str,
    fare: str,
    pass_count: int,
    booking_id: int,
    client_id: int,
) -> np.ndarray:
    encoders, _ = load_encoders_and_mapping()

    sch_time = time_to_seconds(sched_time)
    sch_late = sch_time + pickup_window_mins * 60

    feat_dict = {
        "BookingId": booking_id,
        "ClientId": client_id,
        "SchTime": sch_time,
        "SchLate": sch_late,
        "PassCount": pass_count,
        "DayofMonth": ride_date.day,
        "Month": ride_date.month,
        "le_Purpose": int(encoders["Purpose"].transform([purpose])[0]),
        "le_FareTypeAbbr": int(encoders["FareTypeAbbr"].transform([fare])[0]),
        "le_DayOfWeek": int(encoders["DayOfWeek"].transform([ride_date.strftime("%A")])[0]),
    }

    ordered = [feat_dict[col] for col in STATUS_FEATURE_ORDER]
    return np.array(ordered, dtype="float32").reshape(1, -1), feat_dict


def predict_status(features: np.ndarray) -> float:
    """
    Returns probability of Performed (label 1). No-show prob = 1 - p.
    """
    model = load_status_model()
    prob_show = float(model.predict(features, verbose=0)[0][0])
    return prob_show


def add_manifest_row(row: dict):
    if "manifest" not in st.session_state:
        st.session_state["manifest"] = []
    st.session_state["manifest"].append(row)


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Trip Generator Control Tower", layout="wide")
st.title("Trip Generator • Booking + Routing Workbench")
st.caption(
    "Embeddings-powered distance, status risk, and route clustering built from your notebook artifacts."
)

encoders, mapping_df = load_encoders_and_mapping()
address_catalog = load_address_catalog()

purpose_options = encoders["Purpose"].classes_.tolist()
fare_options = encoders["FareTypeAbbr"].classes_.tolist()

tab_predict, tab_manifest, tab_case = st.tabs(["📞 Intake & Prediction", "🧾 Manifest", "📚 Case Study"])

# ----- Intake tab -----
with tab_predict:
    col_form, col_results = st.columns([3, 2], gap="large")

    with col_form:
        st.subheader("Trip intake")
        with st.form("trip_form"):
            ride_date = st.date_input("Ride date", value=date.today())
            sched_time = st.time_input("Scheduled pickup time", value=time(9, 0))
            pickup_window = st.slider("Pickup window (mins)", 5, 60, value=15, step=5)

            c1, c2 = st.columns(2)
            with c1:
                booking_id = st.number_input("Booking ID", min_value=1, value=18000000, step=1_000)
                client_id = st.number_input("Client ID", min_value=1, value=220000)
                pass_count = st.slider("Passengers", 1, 4, value=1)
            with c2:
                purpose = st.selectbox("Purpose", purpose_options, index=purpose_options.index("Medical Appointment"))
                fare = st.selectbox("Fare type", fare_options, index=fare_options.index("ADA"))

            st.markdown("**Addresses**")
            pickup_text = st.text_input("Pickup address (type at least 3 chars)")
            pickup_suggestions = suggest_addresses(pickup_text, address_catalog)
            pickup_choice = st.selectbox(
                "Pickup suggestions", ["Use typed input"] + pickup_suggestions, disabled=len(pickup_suggestions) == 0
            )
            pickup_addr = pickup_text if pickup_choice == "Use typed input" else pickup_choice

            drop_text = st.text_input("Dropoff address (type at least 3 chars)")
            drop_suggestions = suggest_addresses(drop_text, address_catalog)
            drop_choice = st.selectbox(
                "Dropoff suggestions", ["Use typed input"] + drop_suggestions, disabled=len(drop_suggestions) == 0
            )
            drop_addr = drop_text if drop_choice == "Use typed input" else drop_choice

            submitted = st.form_submit_button("Compute predictions")

        if submitted:
            if not pickup_addr or not drop_addr:
                st.error("Please provide both pickup and dropoff addresses.")
            else:
                with st.spinner("Embedding addresses and running models..."):
                    distance_miles = predict_distance(pickup_addr, drop_addr)
                    route_cluster = predict_route_cluster(pickup_addr, drop_addr)
                    status_features, feature_dict = encode_status_features(
                        ride_date, sched_time, pickup_window, purpose, fare, pass_count, int(booking_id), int(client_id)
                    )
                    p_show = predict_status(status_features)
                    p_noshow = 1 - p_show

                st.session_state["latest"] = {
                    "RideDate": ride_date.isoformat(),
                    "SchedTime": sched_time.strftime("%H:%M"),
                    "PickupWindowMins": pickup_window,
                    "BookingId": int(booking_id),
                    "ClientId": int(client_id),
                    "Purpose": purpose,
                    "FareTypeAbbr": fare,
                    "PassCount": pass_count,
                    "PickupAddress": pickup_addr,
                    "DropoffAddress": drop_addr,
                    "PredictedDistanceMiles": round(distance_miles, 2),
                    "RouteCluster": route_cluster,
                    "ProbPerformed": round(p_show, 3),
                    "ProbNoShow": round(p_noshow, 3),
                    "SchLateSec": feature_dict["SchLate"],
                }

    with col_results:
        st.subheader("Outputs")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("Fill the form and hit **Compute predictions** to see results.")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Distance (miles)", f"{latest['PredictedDistanceMiles']:.2f}")
            metric_cols[1].metric("Show probability", f"{latest['ProbPerformed']*100:.1f}%")
            metric_cols[2].metric("Route cluster", str(latest["RouteCluster"]))

            st.progress(latest["ProbPerformed"])
            st.caption("Status model returns probability of **Performed** (vs NoShow).")

            st.json(latest, expanded=False)

            if st.button("Add to manifest"):
                add_manifest_row(latest)
                st.success("Added to manifest.")

        st.divider()
        st.markdown(
            """
            **Suggestion pool:** {} addresses loaded from `models/addresses.csv`.
            Increase `max_rows` inside `load_address_catalog()` if you want full coverage (1.2M rows).
            """.format(len(address_catalog))
        )

# ----- Manifest tab -----
with tab_manifest:
    st.subheader("Live manifest")
    manifest = st.session_state.get("manifest", [])
    if not manifest:
        st.info("No rows yet. Compute a trip and click *Add to manifest*.")
    else:
        df_manifest = pd.DataFrame(manifest)
        st.dataframe(df_manifest, use_container_width=True, hide_index=True)
        st.download_button(
            "Download manifest as CSV",
            data=df_manifest.to_csv(index=False).encode("utf-8"),
            file_name="manifest_preview.csv",
            mime="text/csv",
        )

        st.markdown("**Route load**")
        route_counts = df_manifest["RouteCluster"].value_counts().sort_index()
        st.bar_chart(route_counts)

# ----- Case study tab -----
with tab_case:
    st.subheader("Architecture")
    st.markdown(
        """
1) Address encoder (`models/addresses_model_embeddings.joblib`) turns pickup + dropoff into dense vectors (all-MiniLM-L6-v2).
2) Distance regressor (`models/distance_model.h5`) predicts straight-line miles from the embeddings.
3) Booking status head (`models/booking_status_model.h5`) consumes derived calendar/time features + label-encoded cols; output is P(Performed).
4) Route clustering (`models/route_clustering_model.joblib`) KMeans over concatenated address embeddings to assign a run/route.
5) Human-readable mappings live in `models/booking_le_mapping.csv`; address suggestions use `models/addresses.csv`.
        """
    )

    st.subheader("Operational playbook")
    st.markdown(
        """
- Intake a request → type addresses → accept suggested entries for faster validation.
- Review the distance and status probabilities to set customer expectations.
- Use the cluster id to seat the trip into a run; export the manifest for dispatch.
        """
    )

    st.subheader("Notes")
    st.markdown(
        """
- Models run on CPU; no internet needed once artifacts are present.
- To widen address suggestions, raise `max_rows` in `load_address_catalog` or swap in a vector search (FAISS) against the same embeddings.
- Replace clustering with your live routing solver by swapping `predict_route_cluster`.
        """
    )


if __name__ == "__main__":
    # Running `python app.py` gives a quick, non-UI sanity check
    sample_pickup = "777 Bannock St, Denver, CO 80204"
    sample_drop = "12605 E 16th Ave, Aurora, CO 80045"
    print("Distance test", predict_distance(sample_pickup, sample_drop))
    print("Route cluster", predict_route_cluster(sample_pickup, sample_drop))
