# Tennis Project

This project analyzes tennis videos to detect players, track the ball, and recognize strokes.

## Project Structure

```
TennisProjectForColab/
├── src/                    # Source code
│   ├── ball_detection.py   # Ball detection logic
│   ├── ball_tracker_net.py # Ball tracking neural network
│   ├── court_detection.py  # Court detection logic
│   ├── detection.py        # General detection utilities
│   ├── pose.py            # Pose detection
│   ├── process.py         # Main processing pipeline
│   ├── stroke_recognition.py # Stroke recognition logic
│   └── saved states/      # Model weights (not in git)
├── output/                 # Output videos and results
└── README.md
```

## Required Model Weights

The following model weights are required and should be downloaded from Google Drive:
1. `tracknet_weights_2_classes.pth` - Ball detection model weights
2. `storke_classifier_weights.pth` - Stroke classification model weights

Place these files in your Google Drive at: `/content/drive/MyDrive/Tennis_Weights/`

## Setup Instructions

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/pirroux/TennisProjectForColab.git
cd TennisProjectForColab
```

2. Create necessary directories:
```bash
mkdir -p src/saved\ states output
```

3. Download model weights from Google Drive and place them in `src/saved states/`

### Google Colab Setup
1. Mount Google Drive
2. Clone the repository:
```python
!git clone https://github.com/pirroux/TennisProjectForColab.git
%cd TennisProjectForColab
```

3. The model weights will be automatically loaded from `/content/drive/MyDrive/Tennis_Weights/`

## Usage

1. Place your input video in the project directory
2. Run the processing script:
```python
python src/process.py
```

The output will be saved in the `output/` directory.
