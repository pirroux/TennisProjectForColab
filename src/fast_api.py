from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from process import main
import cv2
from PIL import Image
import io
import tempfile
import os
import uuid
import subprocess
from flask import Flask, send_file, jsonify, request
import base64

app = FastAPI()

@app.get("/")
def root():
    return {
    'greeting': 'Welcome to tennis vision API!'
}

@app.get("/predict")
def predict(minimap=0, bounce=0, input_video_name=None, ouput_video_name=None):
    subprocess.run(["python3", "predict_video.py", f"--input_video_path=VideoInput/{input_video_name}.mp4", f"--output_video_path=VideoOutput/{ouput_video_name}.mp4", f"--minimap={minimap}", f"--bounce={bounce}"])
    return {'greeting': "Please find below your treated videos"}


@app.post("/savefile")
async def convert_video_to_bw_frame(file: UploadFile = File(...)):

    #save the file in video input directory
    video_name = f"{uuid.uuid4()}.mp4"
    #save_directory = "/apivideos/"
    #video_path = os.path.join(save_directory, video_name)
    with open(video_name, "wb") as buffer:
        contents = await file.read()
        buffer.write(contents)

    #launchin main python file from api
    result_json = main(video_name)

    video_output_path = 'output/output.avi'
    with open(video_output_path, 'rb') as file:
        video = base64.b64encode(file.read()).decode('utf-8')
    # return the main json response
    return {'video': video, 'result_json': result_json}
