import streamlit as st
import os
import numpy as np
from PIL import Image
import websocket
import json
import time
import random

from utils.model_loader import (
    load_cry_detection_model,
    load_object_detection_model,
    load_posture_detection_model,
)
from utils.inference import (
    predict_cry,
    predict_object,
    process_video_for_detection,
    predict_posture,
)


st.set_page_config(page_title="Baby Posture, Object Detection & Sensor Simulator", layout="wide")

WEBSOCKET_URL = "ws://127.0.0.1:5000/ws"

st.title("👶 Baby Posture/Object Detection & Sensor Simulator")

_ = load_posture_detection_model()
_ = load_object_detection_model()
_ = load_cry_detection_model()

posture_model = load_posture_detection_model()
object_model = load_object_detection_model()


def send_sensor_data(ws, temperature, crying_detected, object_detected_list):
    try:
        sensor_data = {
            "id": 1,
            "timestamp": time.time() * 1000,
            "temperature": temperature,
            "cryingDetected": crying_detected,
            "objectDetected": [
                {"object_name": obj, "timestamp": time.time() * 1000}
                for obj in object_detected_list
            ],
        }

        message = {
            "type": "sensor_update",
            "data": sensor_data,
        }

        ws.send(json.dumps(message))
        st.success(f"Sent sensor data: {json.dumps(sensor_data)}")
    except Exception as e:
        st.error(f"Error sending data: {e}")


if "ws" not in st.session_state:
    st.session_state.ws = None
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0
SEND_INTERVAL_FRAMES = 10

st.subheader("WebSocket Connection")
col_ws1, col_ws2 = st.columns(2)

with col_ws1:
    if st.button("Connect to WebSocket", key="connect_ws"):
        if st.session_state.ws and st.session_state.ws.connected:
            st.warning("Already connected to WebSocket.")
        else:
            try:
                st.session_state.ws = websocket.create_connection(WEBSOCKET_URL)
                st.success(f"Connected to WebSocket at {WEBSOCKET_URL}")
            except Exception as e:
                st.error(f"Failed to connect to WebSocket: {e}")

with col_ws2:
    if st.button("Disconnect from WebSocket", key="disconnect_ws"):
        if st.session_state.ws:
            st.session_state.ws.close()
            st.session_state.ws = None
            st.success("Disconnected from WebSocket.")
        else:
            st.warning("Not connected to WebSocket.")


st.header("Image and Video Analysis (Posture & Object Detection)")
st.write(
    "Upload an image or video to detect the baby's posture and objects around the baby simultaneously."
)

file_type = st.radio("Select input type", ["Image", "Video"])

if file_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image...", type=["jpg", "jpeg", "png"], key="image_uploader"
    )
    if uploaded_file is not None:
        st.write("Performing posture and object detection...")

        posture_results = predict_posture(posture_model, uploaded_file)
        object_results = predict_object(object_model, uploaded_file)

        if "error" in posture_results:
            st.error(f"Posture Detection Error: {posture_results['error']}")
        if "error" in object_results:
            st.error(f"Object Detection Error: {object_results['error']}")

        if "error" not in posture_results and "error" not in object_results:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original")
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.subheader("Posture Detection")
                if "output_image" in posture_results and isinstance(
                    posture_results["output_image"], Image.Image
                ):
                    st.image(
                        posture_results["output_image"],
                        caption="Posture Detection Output",
                        use_container_width=True,
                    )
                else:
                    st.warning("Posture detection output image not available.")

            with col3:
                st.subheader("Object Detection")
                if "output_image" in object_results and isinstance(
                    object_results["output_image"], Image.Image
                ):
                    st.image(
                        object_results["output_image"],
                        caption="Object Detection Output",
                        use_container_width=True,
                    )
                else:
                    st.warning("Object detection output image not available.")

            st.markdown("---")
            st.subheader("Analysis Summary:")
            st.write(f"**Posture Status:** {posture_results['posture']}")
            st.write(
                f"**Objects Detected:** {', '.join(object_results['detected_objects']) if object_results['detected_objects'] else 'None'}"
            )

            if object_results["hazardous_objects"]:
                st.error(
                    f"🚨 **DANGER! Hazardous objects detected near baby: {', '.join(object_results['hazardous_objects'])}!**"
                )
            else:
                st.success(
                    "✅ Baby is in a normal status. No hazardous objects detected."
                )

            st.success("Image analysis complete!")

            if st.session_state.ws and st.session_state.ws.connected:
                temperature = 75.0
                crying_detected = False

                object_status_message = ""
                if object_results["hazardous_objects"]:
                    object_status_message = (
                        "DANGER: Hazardous objects detected: "
                        + ", ".join(object_results["hazardous_objects"])
                    )
                else:
                    object_status_message = "SAFE: No hazardous objects detected."

                send_sensor_data(
                    st.session_state.ws,
                    temperature,
                    crying_detected,
                    [object_status_message],
                )
            else:
                st.warning("Connect to WebSocket to send sensor data.")

elif file_type == "Video":
    uploaded_file = st.file_uploader(
        "Upload a video...", type=["mp4", "avi", "mov"], key="video_uploader"
    )
    if uploaded_file is not None:
        st.write("Performing posture and object detection in video...")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_file)

        with col2:
            st.subheader("Live Processed Video (Posture & Object Detection)")
            live_video_placeholder = st.empty()

        st.markdown("---")
        st.subheader("Analysis Summary (Video):")
        posture_summary_placeholder = st.empty()
        objects_detected_placeholder = st.empty()
        hazardous_objects_placeholder = st.empty()
        status_message_placeholder = st.empty()

        final_video_path = None
        all_detected_objects_overall = set()
        all_hazardous_objects_overall = set()
        posture_summary_overall = "Unknown"

        st.session_state.frame_counter = 0

        try:
            for yielded_item in process_video_for_detection(
                posture_model, object_model, uploaded_file
            ):
                if isinstance(yielded_item, dict):
                    if "error" in yielded_item:
                        st.error(f"Video Processing Error: {yielded_item['error']}")
                        break
                    else:
                        final_video_path = yielded_item["output_video_path"]
                        all_detected_objects_overall = set(
                            yielded_item["all_detected_objects"]
                        )
                        all_hazardous_objects_overall = set(
                            yielded_item["all_hazardous_objects"]
                        )
                        posture_summary_overall = yielded_item["posture_summary"]
                        status_message_placeholder.success("Video analysis complete!")
                        break
                else:
                    (
                        frame,
                        current_posture,
                        current_frame_objects,
                        current_frame_hazardous_objects,
                    ) = yielded_item
                    live_video_placeholder.image(
                        frame, channels="BGR", use_container_width=True
                    )

                    posture_summary_placeholder.write(
                        f"**Posture Status (Current Frame):** {current_posture}"
                    )
                    objects_detected_placeholder.write(
                        f"**Objects Detected (Current Frame):** {', '.join(current_frame_objects) if current_frame_objects else 'None'}"
                    )

                    if current_frame_hazardous_objects:
                        hazardous_objects_placeholder.error(
                            f"🚨 **DANGER! Hazardous objects detected (Current Frame): {', '.join(current_frame_hazardous_objects)}!**"
                        )
                    else:
                        hazardous_objects_placeholder.success(
                            "✅ No hazardous objects detected (Current Frame)."
                        )

                    all_detected_objects_overall.update(current_frame_objects)
                    all_hazardous_objects_overall.update(
                        current_frame_hazardous_objects
                    )
                    if posture_summary_overall == "Unknown":
                        posture_summary_overall = current_posture

                    st.session_state.frame_counter += 1
                    if st.session_state.ws and st.session_state.ws.connected:
                        should_send = False
                        if current_frame_hazardous_objects:
                            should_send = True
                        elif (
                            st.session_state.frame_counter % SEND_INTERVAL_FRAMES
                            == 0
                        ):
                            should_send = True

                        if should_send:
                            temperature = round(
                                np.random.uniform(70.0, 85.0), 2
                            )
                            crying_detected = bool(
                                np.random.choice([True, False])
                            )

                            object_status_message = ""
                            if current_frame_hazardous_objects:
                                object_status_message = (
                                    "DANGER: Hazardous objects detected: "
                                    + ", ".join(current_frame_hazardous_objects)
                                )
                            else:
                                object_status_message = (
                                    "SAFE: No hazardous objects detected."
                                )

                            send_sensor_data(
                                st.session_state.ws,
                                temperature,
                                crying_detected,
                                [object_status_message],
                            )
                            st.session_state.frame_counter = 0

            if final_video_path:
                st.subheader("Processed Video (Full)")
                st.video(final_video_path)

                st.markdown("---")
                st.subheader("Overall Analysis Summary (Video):")
                st.write(
                    f"**Overall Posture Status:** {posture_summary_overall}"
                )
                st.write(
                    f"**Overall Objects Detected:** {', '.join(all_detected_objects_overall) if all_detected_objects_overall else 'None'}"
                )

                if all_hazardous_objects_overall:
                    st.error(
                        f"🚨 **Overall Hazardous objects detected in video: {', '.join(all_hazardous_objects_overall)}!**"
                    )
                else:
                    st.success(
                        "✅ Baby is in a normal status. No hazardous objects detected in video."
                    )
        finally:
            if final_video_path and os.path.exists(final_video_path):
                os.remove(final_video_path)


st.header("Sensor Data Simulator")

st.subheader("Sensor Data Inputs")
temperature_input = st.slider(
    "Temperature", min_value=60.0, max_value=100.0, value=75.0, step=0.1
)
crying_detected_input = st.checkbox("Crying Detected", value=False)
object_detected_input = st.multiselect(
    "Objects Detected",
    ["toy", "bottle", "blanket", "pacifier"],
    [],
)

if st.button("Send Sensor Data"):
    if st.session_state.ws and st.session_state.ws.connected:
        send_sensor_data(
            st.session_state.ws,
            temperature_input,
            crying_detected_input,
            object_detected_input,
        )
    else:
        st.warning("Please connect to WebSocket first.")

st.subheader("Manual Cry Notification")
if st.button("Send Cry Notification"):
    if st.session_state.ws and st.session_state.ws.connected:
        send_sensor_data(
            st.session_state.ws,
            temperature_input,
            True,
            object_detected_input,
        )
    else:
        st.warning("Please connect to WebSocket first.")

st.subheader("Auto-send Random Sensor Data")
auto_send_enabled = st.checkbox("Enable Auto-send (every 5 seconds)")

if auto_send_enabled:
    if st.session_state.ws and st.session_state.ws.connected:
        st.write("Auto-sending random sensor data...")

        random_temperature = round(random.uniform(70.0, 85.0), 2)
        random_crying_detected = random.choice([True, False])
        random_object_detected = []
        if random.random() < 0.3:
            random_object_detected.append(
                random.choice(["toy", "bottle", "blanket"])
            )

        send_sensor_data(
            st.session_state.ws,
            random_temperature,
            random_crying_detected,
            random_object_detected,
        )
        time.sleep(5)
        st.experimental_rerun()
    else:
        st.warning("Connect to WebSocket to enable auto-send.")

