import streamlit as st
import requests
import base64
from PIL import Image
import io

# Define the URL of your FastAPI backend with port 8000
FASTAPI_URL = "https://3e2a-104-155-211-142.ngrok-free.app"

# Function to make requests to FastAPI endpoints
def make_request(endpoint, data=None):
    response = requests.post(f"{FASTAPI_URL}/{endpoint}", data=data)
    return response.json()

# Streamlit app
st.title("Tennis Vision App")

# Upload video file
video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Make request to FastAPI endpoint to process the video
    files = {"file": video_file}
    response = make_request("savefile", files)

    # Get processed video and other data from the response
    processed_video = base64.b64decode(response['video'])
    heatmap_image = Image.open(io.BytesIO(base64.b64decode(response['heatmap'])))
    graph_image = Image.open(io.BytesIO(base64.b64decode(response['graph'])))
    result_json = response['result_json']

    # Display processed video, heatmap, graph, and result JSON
    st.video(processed_video)
    st.image(heatmap_image, caption="Heatmap")
    st.image(graph_image, caption="Graph")
    st.json(result_json)
