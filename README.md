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
1. Create a new cell at the start of your notebook with this setup code:
```python
# Function for colored status messages
def print_status(message, status="info"):
    colors = {
        "info": "\033[94m",  # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m"    # Red
    }
    end_color = "\033[0m"
    print(f"{colors.get(status, colors['info'])}{message}{end_color}")

# Mount Google Drive
print_status("Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Set up repository
repo_name = "TennisProjectForColab"
repo_url = "https://github.com/pirroux/TennisProjectForColab.git"

print_status("Setting up repository...")
try:
    # Try to clone the repository
    !git clone $repo_url 2>/dev/null || (cd $repo_name && git pull)
    %cd $repo_name
    print_status("Repository ready!", "success")
except Exception as e:
    print_status(f"Error setting up repository: {str(e)}", "error")
    raise e

# Create necessary directories
print_status("Creating directories...")
!mkdir -p src/saved\ states output

# Verify model weights
print_status("Checking model weights...")
import os

weights_dir = "/content/drive/MyDrive/Tennis_Weights"
required_weights = [
    "tracknet_weights_2_classes.pth",
    "storke_classifier_weights.pth"
]

missing_weights = []
for weight in required_weights:
    path = os.path.join(weights_dir, weight)
    if os.path.exists(path):
        print_status(f"Found {weight}", "success")
    else:
        print_status(f"Missing {weight}", "error")
        missing_weights.append(weight)

if missing_weights:
    print_status("\nMissing required weights files:", "warning")
    print_status("Please ensure the following files are in your Google Drive " +
                 f"at {weights_dir}:", "warning")
    for weight in missing_weights:
        print_status(f"  - {weight}", "warning")
else:
    print_status("\nAll required model weights found!", "success")
```

2. Run the cell to:
   - Mount your Google Drive
   - Clone/update the repository
   - Create necessary directories
   - Verify model weights are present

The setup will provide colored status messages and clear error reporting if anything is missing.

## Usage

1. Place your input video in the project directory
2. Run the processing script:
```python
python src/process.py
```

The output will be saved in the `output/` directory.
