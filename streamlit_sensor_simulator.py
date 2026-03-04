import streamlit as st
import websocket
import json
import time
import random

# Define the WebSocket server URL
WEBSOCKET_URL = "ws://127.0.0.1:5000/ws"

st.title("Streamlit Sensor Data Sender")

# Function to send sensor data
def send_sensor_data(ws, temperature, crying_detected, object_detected_list):
    try:
        sensor_data = {
            "id": 1,
            "timestamp": time.time() * 1000,
            "temperature": temperature,
            "cryingDetected": crying_detected,
            "objectDetected": [{"object_name": obj, "timestamp": time.time() * 1000} for obj in object_detected_list]
        }

        message = {
            "type": "sensor_update",
            "data": sensor_data
        }

        ws.send(json.dumps(message))
        st.success(f"Sent sensor data: {json.dumps(sensor_data)}")
    except Exception as e:
        st.error(f"Error sending data: {e}")

# WebSocket connection management
if 'ws' not in st.session_state:
    st.session_state.ws = None

if st.button("Connect to WebSocket"):
    if st.session_state.ws and st.session_state.ws.connected:
        st.warning("Already connected to WebSocket.")
    else:
        try:
            st.session_state.ws = websocket.create_connection(WEBSOCKET_URL)
            st.success(f"Connected to WebSocket at {WEBSOCKET_URL}")
        except Exception as e:
            st.error(f"Failed to connect to WebSocket: {e}")

if st.button("Disconnect from WebSocket"):
    if st.session_state.ws:
        st.session_state.ws.close()
        st.session_state.ws = None
        st.success("Disconnected from WebSocket.")
    else:
        st.warning("Not connected to WebSocket.")

st.subheader("Sensor Data Inputs")
temperature_input = st.slider("Temperature", min_value=60.0, max_value=100.0, value=75.0, step=0.1)
crying_detected_input = st.checkbox("Crying Detected", value=False)
object_detected_input = st.multiselect(
    "Objects Detected",
    ["toy", "bottle", "blanket", "pacifier"],
    []
)


if st.button("Send Sensor Data"):
    if st.session_state.ws and st.session_state.ws.connected:
        send_sensor_data(st.session_state.ws, temperature_input, crying_detected_input, object_detected_input)
    else:
        st.warning("Please connect to WebSocket first.")

st.subheader("Manual Cry Notification")
if st.button("Send Cry Notification"):
    if st.session_state.ws and st.session_state.ws.connected:
        # Force crying_detected to True while keeping current temperature and objects
        send_sensor_data(st.session_state.ws, temperature_input, True, object_detected_input)
    else:
        st.warning("Please connect to WebSocket first.")

st.subheader("Auto-send Random Sensor Data")
auto_send_enabled = st.checkbox("Enable Auto-send (every 5 seconds)")

if auto_send_enabled:
    if st.session_state.ws and st.session_state.ws.connected:
        st.write("Auto-sending random sensor data...")
        # Use a non-blocking way to send data periodically
        # Streamlit's rerun mechanism makes direct while loops problematic for UI updates
        # For simplicity, we'll just send one random data point per rerun if enabled
        random_temperature = round(random.uniform(70.0, 85.0), 2)
        random_crying_detected = random.choice([True, False])
        random_object_detected = []
        if random.random() < 0.3: # 30% chance of object detection
            random_object_detected.append(random.choice(["toy", "bottle", "blanket"]))

        send_sensor_data(st.session_state.ws, random_temperature, random_crying_detected, random_object_detected)
        time.sleep(5) # This will pause the Streamlit app for 5 seconds
        st.rerun() # Rerun to send next data point
    else:
        st.warning("Connect to WebSocket to enable auto-send.")
