# Video Brand Blurring

## Introduction
Our video brand blurring tool is designed to provide a simple solution for automatically blurring brands and logos in videos. The detection model is trained on the LogoDet-3k dataset, which claims to be the “largest logo detection dataset with full annotation, which has 3,000 logo categories, about 200,000 manually annotated logo objects, and 158,652 images” (Wang et al., 2022). Ultimately we used a subset of this dataset from Yuan (2023) which has ~250 categories (classes) and ~10000 images. All images for each of the 250 classes from the original dataset are included. 


Access the application online here:
https://video-brand-detection.streamlit.app/

## How It Works
1. Upload a video file directly through the UI.
2. The tool will automatically begin processing the video to detect and blur detected brands or logos that are in our dataset.
3. After processing, the blurred video is displayed on the interface, and is available to be downloaded.

## Setup and Installation
### Requirements
- Python 3.9+
- Streamlit
- OpenCV
- Ultralytics YOLO

To run the streamlit app locally, paste the following command into your terminal after replacing the filepath with your location.

```
streamlit run [filepath]/main.py
```