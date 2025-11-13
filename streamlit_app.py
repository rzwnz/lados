"""
Streamlit web app for LADOS classification.
Backup frontend for the FastAPI backend.
"""

import io
import requests
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"
MAX_UPLOAD_SIZE_MB = 10

st.set_page_config(
    page_title="LADOS Classifier",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 LADOS Oil Spill Classification")
st.markdown("Upload an image to classify using the LADOS aerial oil-spill detection model")

# Sidebar for API status and metrics
with st.sidebar:
    st.header("📊 System Status")

    # Health check
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("model_loaded"):
                st.success("✅ Model Loaded")
                st.info(f"Device: {health_data.get('device', 'unknown')}")
            else:
                st.warning("⚠️ Model Not Loaded")
        else:
            st.error("❌ API Unavailable")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API")
        st.info("Make sure the FastAPI server is running on port 8000")

    st.divider()

    # Metrics
    st.subheader("📈 Metrics")
    try:
        metrics_response = requests.get(f"{API_URL}/metrics", timeout=2)
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()

            if metrics_data.get("training_metrics"):
                training_metrics = metrics_data["training_metrics"]
                if "macro_f1" in training_metrics:
                    st.metric("Macro F1 Score", f"{training_metrics['macro_f1']:.3f}")
                if "accuracy" in training_metrics:
                    st.metric("Accuracy", f"{training_metrics['accuracy']:.3f}")

            if metrics_data.get("inference_stats"):
                inference_stats = metrics_data["inference_stats"]
                st.metric("Total Requests", inference_stats.get("total_requests", 0))
                if "avg_latency_ms" in inference_stats:
                    st.metric("Avg Latency", f"{inference_stats['avg_latency_ms']:.1f} ms")
    except requests.exceptions.RequestException:
        st.info("Metrics unavailable")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an aerial image for oil spill classification",
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")

        # File size check
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            st.error(
                f"File too large ({file_size_mb:.2f} MB). Maximum size: {MAX_UPLOAD_SIZE_MB} MB"
            )
        else:
            st.info(f"File size: {file_size_mb:.2f} MB")

            # Predict button
            if st.button("🔍 Predict", type="primary", width="stretch"):
                with st.spinner("Processing image..."):
                    try:
                        # Prepare file for upload
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }

                        # Send prediction request
                        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                        if response.status_code == 200:
                            prediction_data = response.json()
                            st.session_state["prediction"] = prediction_data
                            st.session_state["image"] = image
                            st.rerun()
                        else:
                            st.error(f"Prediction failed: {response.status_code}")
                            st.error(response.text)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to API: {e}")
                        st.info("Make sure the FastAPI server is running")

with col2:
    st.header("📊 Prediction Results")

    if "prediction" in st.session_state and st.session_state["prediction"]:
        prediction_data = st.session_state["prediction"]

        # Display top prediction
        if "predictions" in prediction_data and len(prediction_data["predictions"]) > 0:
            top_pred = prediction_data["predictions"][0]
            top_class = top_pred.get("class", "Unknown")
            top_score = top_pred.get("score", 0.0)

            st.success(f"**Top Prediction:** {top_class}")
            st.metric("Confidence", f"{top_score * 100:.2f}%")

            # Inference time
            if "inference_time_ms" in prediction_data:
                st.caption(f"Inference time: {prediction_data['inference_time_ms']:.1f} ms")

            st.divider()

            # Bar chart
            if len(prediction_data["predictions"]) > 0:
                st.subheader("Class Probabilities")

                # Prepare data for chart
                predictions = prediction_data["predictions"][:10]  # Top 10
                classes = [p.get("class", "Unknown") for p in predictions]
                scores = [p.get("score", 0.0) * 100 for p in predictions]

                # Create bar chart
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=classes,
                            y=scores,
                            marker_color="steelblue",
                            text=[f"{s:.1f}%" for s in scores],
                            textposition="outside",
                        )
                    ]
                )

                fig.update_layout(
                    title="Prediction Scores",
                    xaxis_title="Class",
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 100],
                    height=400,
                    showlegend=False,
                )

                st.plotly_chart(fig, width="stretch")

                # Detailed table
                st.subheader("Detailed Results")
                df = pd.DataFrame(predictions)
                df["score"] = df["score"].apply(lambda x: f"{x * 100:.2f}%")
                df.columns = ["Class", "Confidence"]
                st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.warning("No predictions available")
    else:
        st.info("👈 Upload an image and click 'Predict' to see results here")

# Batch prediction section
st.divider()
st.header("📦 Batch Prediction")

uploaded_files = st.file_uploader(
    "Choose multiple image files",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    help="Upload multiple images for batch processing",
)

if uploaded_files and len(uploaded_files) > 0:
    st.info(f"{len(uploaded_files)} file(s) selected")

    if st.button("🚀 Process Batch", type="primary"):
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            try:
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

                response = requests.post(f"{API_URL}/predict_batch", files=files, timeout=60)

                if response.status_code == 200:
                    batch_data = response.json()

                    if "job_id" in batch_data:
                        st.info(f"Batch job queued: {batch_data['job_id']}")
                        st.info(f"Status: {batch_data.get('status', 'unknown')}")
                    elif "results" in batch_data:
                        results = batch_data["results"]
                        st.success(f"Processed {len(results)} images")

                        # Display results in expandable sections
                        for i, result in enumerate(results):
                            with st.expander(f"Image {i+1}: {result.get('filename', 'unknown')}"):
                                if "error" in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    if "top_class" in result:
                                        st.success(f"**Prediction:** {result['top_class']}")
                                        st.metric(
                                            "Confidence", f"{result.get('top_score', 0) * 100:.2f}%"
                                        )

                                    if "predictions" in result:
                                        pred_df = pd.DataFrame(result["predictions"][:5])
                                        pred_df["score"] = pred_df["score"].apply(
                                            lambda x: f"{x * 100:.2f}%"
                                        )
                                        pred_df.columns = ["Class", "Confidence"]
                                        st.dataframe(pred_df, hide_index=True)
                else:
                    st.error(f"Batch prediction failed: {response.status_code}")
                    st.error(response.text)
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")

# Footer
st.divider()
st.caption("LADOS Classification System | FastAPI Backend | Streamlit Frontend")
