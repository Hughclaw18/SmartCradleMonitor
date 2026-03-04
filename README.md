# Baby Posture, Object, and Cry Detection

This project provides a comprehensive baby monitoring system using advanced deep learning models for:
- **Posture Detection**: Identify various baby postures (e.g., on back, on stomach).
- **Object Detection**: Detect objects around the baby that might pose a risk.
- **Cry Detection**: Identify baby cries from audio input.

The application is built with Streamlit, offering an interactive web interface for easy use.

## Project Structure

```
.
├── assets/
│   ├── images/
│   │   ├── baby_on_back.jpg
│   │   └── sample_with_object.jpg
│   └── videos/
│       └── demo_video.mp4
├── models/
│   ├── cry_detection_yoloe.pt
│   ├── object_detection_yoloe.pt
│   └── posture_detection_yolov11.pt
├── utils/
│   ├── __init__.py
│   ├── inference.py
│   └── model_loader.py
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/baby-posture-detection.git
    cd baby-posture-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models:**
    The `.pt` model files are not included in the repository due to their size. You will need to download them separately and place them in the `models/` directory:
    -   `posture_detection_yolov11.pt`
    -   `object_detection_yoloe.pt`
    -   `cry_detection_yoloe.pt`
    *(Provide links to download these models if they are hosted online)*

## Running the Application

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Usage

Navigate through the sidebar to access different detection modes:
-   **Home**: Welcome page with an overview.
-   **Posture Detection**: Upload an image or video to analyze baby posture.
-   **Object Detection**: Upload an image or video to detect objects near the baby.
-   **Cry Detection**: Upload an audio file to detect baby cries.

## Contributing

Feel free to fork the repository, open issues, and submit pull requests.

## License

This project is licensed under the MIT License.
