# Tennis Shot Analysis

A computer vision project for analyzing tennis shots and player movements in tennis videos.

## Features

- Ball tracking and trajectory analysis
- Shot type recognition (forehand, backhand, service)
- Player movement tracking
- Video output with visualizations
- JSON output with detailed analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Tennis_stats.git
cd Tennis_stats
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model weights:
The project requires pre-trained model weights for:
- Ball tracking (tracknet_weights_2_classes.pth)
- Stroke classification (storke_classifier_weights.pth)

Place these files in the appropriate directory (e.g., `/content/drive/MyDrive/Tennis_Weights/`).

## Usage

1. Basic video processing:
```python
from src.process import video_process

result = video_process(
    video_path="path/to/your/video.mp4",
    save_video=True,
    save_json=True,
    show_ball=True
)
```

2. Output:
- Processed video will be saved to `output/output.mp4`
- Analysis results will be saved to `output/shot_analysis.json`

## Project Structure

```
Tennis_stats/
├── src/
│   ├── ball_detection.py      # Ball tracking implementation
│   ├── ball_tracker_net.py    # Neural network for ball detection
│   ├── court_detection.py     # Tennis court detection
│   ├── detection.py           # Player detection
│   ├── pose.py               # Pose estimation
│   ├── process.py            # Main processing pipeline
│   ├── stroke_recognition.py  # Shot type classification
│   └── utils.py              # Utility functions
├── requirements.txt
└── README.md
```

## Output Format

The JSON output includes:
- Ball positions (raw and interpolated)
- Detected strokes with frame numbers
- Video information (fps, resolution)
- Processing time

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
