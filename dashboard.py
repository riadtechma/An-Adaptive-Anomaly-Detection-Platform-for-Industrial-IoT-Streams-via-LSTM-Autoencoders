import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
import psutil
import os
import time

# IMPORT THE NEW AI MODEL
from ai_model import SupplyChainBrain

# ==========================================
# 1. UI CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Supply Chain AI Sentinel",
    layout="wide",
    page_icon="🏭"
)

st.title("🏭 Intelligent Supply Chain Monitor")
st.markdown("### Real-Time Intrusion Detection & Pattern Decoding")

# --- PARAMETERS ---
SEQ_LEN = 30  # Must match the Brain's sequence length
TRAIN_SIZE = 50  # Number of initial frames to calibrate normal behavior

# --- SESSION STATE ---
# We store the most recent valid window here so the button can access it
if 'last_window' not in st.session_state:
    st.session_state['last_window'] = []

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Connection Settings")

# Dropdown to select which Digital Twin to monitor
target_sensor = st.sidebar.selectbox(
    "Select Digital Twin to Monitor:",
    [
        "motor_vibration_1",
        "extruder_temp_1",
        "spectrometer_sorter_1",
        "truck_gps_tracker_1"
    ]
)
ws_url = f"ws://localhost:8000/ws/{target_sensor}"


# Initialize the AI Brain (Cached to persist across re-runs)
@st.cache_resource
def load_brain():
    print("🧠 Initializing Neural Network...")
    return SupplyChainBrain(input_features=1, seq_length=SEQ_LEN)


brain = load_brain()

st.sidebar.markdown("---")
st.sidebar.header("👮 Human-in-the-Loop")
st.sidebar.info("If the AI flags a safe event as an attack, click below to teach it.")

# --- THE TEACHING BUTTON ---
if st.sidebar.button("🛡️ False Alarm? Mark as Normal"):
    # Check if we have enough data in the buffer
    current_window = st.session_state['last_window']

    if len(current_window) == SEQ_LEN:
        with st.sidebar.status("🧠 Re-training Neural Network..."):
            # Force the model to overfit this specific pattern
            loss = brain.force_learn(current_window)
            st.write(f"Pattern Learned! Error dropped to: {loss:.4f}")
        st.sidebar.success("✅ Exception Added to Knowledge Base")
    else:
        st.sidebar.error("Not enough data to learn yet. Wait for stream.")

# ==========================================
# 3. REAL-TIME PLOT LAYOUT
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Stream (Reality vs. AI Prediction)")
    # Placeholder for the main time-series chart
    chart_placeholder = st.empty()

with col2:
    st.subheader("AI Brain Activity")

    # Metrics Row
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        anomaly_meter = st.empty()
    with metric_col2:
        status_text = st.empty()

    st.markdown("---")
    st.markdown("**Model Resource Footprint:**")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        latency_meter = st.empty()
    with res_col2:
        ram_meter = st.empty()
    with res_col3:
        cpu_meter = st.empty()

    st.markdown("---")
    st.markdown("**Latent Space (The AI's 'Thought' Pattern):**")
    # Placeholder for the Latent Vector Bar Chart
    latent_placeholder = st.empty()


# ==========================================
# 4. ASYNC WEBSOCKET CLIENT
# ==========================================
async def consume_twin_data():
    status_log = st.empty()
    status_log.info(f"⏳ Connecting to {target_sensor}...")

    # Buffers for plotting
    raw_history = deque(maxlen=100)
    anomaly_history = deque(maxlen=100)

    # Buffer for AI Processing (Rolling Window)
    ai_window = deque(maxlen=SEQ_LEN)

    # System Monitoring
    process = psutil.Process(os.getpid())
    num_cores = psutil.cpu_count()

    try:
        async with websockets.connect(ws_url) as websocket:
            status_log.success(f"✅ Connected to {target_sensor}!")
            st.toast("Connection Successful", icon="⚡")

            iteration = 0
            while True:
                try:
                    # 1. Receive Data Packet
                    message = await websocket.recv()
                    data = json.loads(message)

                    if "data_vector" not in data:
                        continue

                    # Extract sensor value (First dimension only for this demo)
                    sensor_val = data["data_vector"][0]

                    # 2. Update Buffers
                    raw_history.append(sensor_val)
                    ai_window.append(sensor_val)  # Flat list for the window

                    # --- CRITICAL: Save to Session State for the Button ---
                    # We convert to list immediately to snapshot it
                    if len(ai_window) == SEQ_LEN:
                        st.session_state['last_window'] = list(ai_window)

                    # 3. AI Logic Cycle
                    anomaly_score = 0.0
                    latent_vector = np.zeros(4)
                    current_status = "Waiting for Data..."
                    status_color = "gray"
                    inference_ms = 0.0

                    if len(ai_window) == SEQ_LEN:
                        start_time = time.perf_counter()
                        window_list = list(ai_window)

                        # A. Calibration Phase (First N frames)
                        if iteration < TRAIN_SIZE:
                            loss = brain.learn(window_list)
                            current_status = f"🔵 CALIBRATING ({iteration}/{TRAIN_SIZE})"
                            status_color = "blue"
                            anomaly_score = loss  # Show training loss as score temporarily

                        # B. Detection Phase
                        else:
                            anomaly_score, latent_vector = brain.detect(window_list)

                            # Threshold Logic
                            if anomaly_score > 0.5:
                                current_status = "🔴 INTRUSION DETECTED"
                                status_color = "red"
                            elif anomaly_score > 0.2:
                                current_status = "🟠 WARNING"
                                status_color = "orange"
                            else:
                                current_status = "🟢 SYSTEM NORMAL"
                                status_color = "green"

                        end_time = time.perf_counter()
                        inference_ms = (end_time - start_time) * 1000

                    anomaly_history.append(anomaly_score)

                    # 4. Resource Monitoring
                    mem_usage = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_usage = process.cpu_percent() / num_cores

                    # 5. Update UI (throttled to every 5th frame for performance)
                    if iteration % 5 == 0:
                        # --- Main Chart ---
                        fig = go.Figure()
                        # Sensor Data Line
                        fig.add_trace(go.Scatter(
                            y=list(raw_history),
                            mode='lines',
                            name='Sensor Data',
                            line=dict(color='#00CC96')
                        ))
                        # Anomaly Score Line
                        fig.add_trace(go.Scatter(
                            y=list(anomaly_history),
                            mode='lines',
                            name='Anomaly Score',
                            fill='tozeroy',
                            line=dict(color='#EF553B', width=1)
                        ))
                        fig.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=0, b=0),
                            template="plotly_dark",
                            xaxis_title="Time Steps",
                            yaxis_title="Amplitude / Error"
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)

                        # --- Metrics ---
                        display_score = min(anomaly_score, 1.0)
                        anomaly_meter.metric("Anomaly Score", f"{display_score:.4f}")
                        status_text.markdown(f"### :{status_color}[{current_status}]")

                        latency_meter.metric("Latency", f"{inference_ms:.1f} ms")
                        ram_meter.metric("RAM", f"{mem_usage:.0f} MB")
                        cpu_meter.metric("CPU", f"{min(cpu_usage, 100.0):.1f}%")

                        # --- Latent Space Visualization ---
                        if np.any(latent_vector):
                            fig_latent = px.bar(
                                x=["Dim 1", "Dim 2", "Dim 3", "Dim 4"],
                                y=latent_vector,
                                title="Decoded Pattern Signature",
                                labels={'y': 'Activation', 'x': 'Latent Dimension'}
                            )
                            fig_latent.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=30, b=0),
                                template="plotly_dark"
                            )
                            latent_placeholder.plotly_chart(fig_latent, use_container_width=True)

                    iteration += 1
                    # Small sleep to yield control back to the loop
                    await asyncio.sleep(0.01)

                except json.JSONDecodeError:
                    st.error("❌ Failed to decode JSON packet")
                except Exception as inner_e:
                    # Log but don't crash the loop
                    print(f"Loop Warning: {inner_e}")
                    await asyncio.sleep(0.1)

    except Exception as e:
        status_log.error(f"🔌 Connection Failed: {e}")
        st.error(f"Could not connect to {ws_url}. Is the Digital Twin API running?")


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Create and run the event loop
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(consume_twin_data())
    except KeyboardInterrupt:
        pass
    finally:
        # loop.close() # Streamlit manages its own lifecycle usually
        pass