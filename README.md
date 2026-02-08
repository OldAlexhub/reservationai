# Trip Generator – Reservation-to-Route Solver (POV)

This app turns the notebook artifacts in this repo into a reservation-stage decision aide: as soon as a call taker enters addresses and basic trip metadata, the system predicts distance, likelihood to perform vs. no-show, and assigns the trip to an existing run/route cluster. The manifest view then becomes the handoff for dispatch.

## Architecture (built from your notebook outputs)
- **Address encoder**: `models/addresses_model_embeddings.joblib` (all-MiniLM-L6-v2) maps pickup/dropoff text to vectors.
- **Distance regressor**: `models/distance_model.h5` consumes concatenated pickup/dropoff embeddings → direct distance estimate (miles).
- **Booking status head**: `models/booking_status_model.h5` predicts P(Performed) using encoded categorical features + calendar/time fields derived in `booking_cleaning`.
- **Route clustering**: `models/route_clustering_model.joblib` (KMeans) labels each trip with a run/route cluster based on the same address embeddings.
- **Human-readable encodings**: `models/booking_le_mapping.csv` and `models/booking_label_encoders.joblib` keep category ↔ code lookups consistent.
- **Address suggestion pool**: `models/addresses.csv` (1.2M rows) supplies quick typeahead; the app currently loads the first 300k rows for speed, configurable in `load_address_catalog`.

## Operational flow (reservation-stage solver)
1) **Intake**: Call taker enters ride date/time/window, purpose, fare type, passengers, booking/client IDs, and typed pickup/dropoff. Suggestions accelerate correct address capture.
2) **Embeddings & distance**: Pickup + dropoff text → embeddings → distance model → miles estimate for fare checks and schedule feasibility.
3) **Status risk**: Calendar/time + encoded purpose/fare → status model → probability of *Performed* (show) vs. *NoShow*. This flags fragile bookings early.
4) **Route seat**: Concatenated embeddings → clustering model → route/run cluster ID to slot the trip into an existing pattern while still on the call.
5) **Manifest build**: Operator clicks **Add to manifest**; trips accumulate in-session. Export CSV to feed downstream dispatching, pricing, or driver assignment.

## Running the app
```bash
# from C:\Users\moham\Desktop\Trip Generator
python -m pip install -r app_requirements.txt
streamlit run app.py
```
Open the URL Streamlit prints (e.g., http://localhost:8501).

## Key files
- `app.py` – Streamlit UI + orchestration.
- `app_requirements.txt` – dependency pin set.
- `models/addresses_model_embeddings.joblib` – sentence-transformer encoder (saved with CPU-safe load).
- `models/distance_model.h5` – Keras distance regressor.
- `models/booking_status_model.h5` – Keras booking status classifier.
- `models/route_clustering_model.joblib` – KMeans route/run clusters.
- `models/booking_label_encoders.joblib` / `models/booking_le_mapping.csv` – categorical encoders + readable mapping.
- `models/addresses.csv` – address catalog powering suggestions (can be swapped for vector search if you want fuzzier matches).

## Notes and extensions
- The status model output is P(Performed); no-show probability is `1 - p`. The progress bar in the UI uses the show probability.
- To widen suggestions or enable fuzzy matching, replace `suggest_addresses` with a FAISS/ANN lookup over the same embeddings.
- To integrate a live routing solver, swap `predict_route_cluster` with your optimizer while keeping the manifest schema.
- Everything runs CPU-only; no internet is needed once artifacts are present.

## Lifecycle POV
Reservation → (this app) → distance check, risk flag, provisional route seat → exported manifest → dispatch/driver assignment. By frontloading predictions at intake, call takers can set expectations, pre-balance runs, and spot fragile trips before they hit operations.
