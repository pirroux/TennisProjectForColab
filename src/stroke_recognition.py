import sys
sys.path.append('/content/TennisProject/src')
import os
import imutils
import torch
import torchvision
from torchvision.models import Inception_V3_Weights
import tensorflow as tf
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import ToTensor
from datasets import ThetisDataset, create_train_valid_test_datasets, StrokesDataset
from detection import center_of_box
from utils import get_dtype
import pandas as pd
import numpy
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
from numpy import dtype
from numpy.dtypes import Float64DType

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.inception_v3(
            weights=Inception_V3_Weights.DEFAULT
        )
        self.feature_extractor.fc = Identity()

    def forward(self, x):
        output = self.feature_extractor(x)
        return output


class LSTM_model(nn.Module):
    """
    Time sequence model for stroke classifying
    """
    def __init__(self, num_classes=3, input_size=2048, num_layers=3, hidden_size=90, dtype=torch.cuda.FloatTensor):
        super().__init__()
        self.dtype = dtype
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # If x is a TensorFlow tensor, convert it to a PyTorch tensor
        if isinstance(x, tf.Tensor):
            x = torch.from_numpy(x.numpy())  # Convert TensorFlow tensor to PyTorch tensor

        # x shape is (batch_size, seq_len, input_size)
        h0, c0 = self.init_state(x.size(0))
        output, (hn, cn) = self.LSTM(x, (h0, c0))

        # The size might need to be adjusted depending on your model design
        size = x.size(1) // 4  # This selects a subset of the sequence

        output = output[:, -size:, :]  # Selects the last 'size' timesteps from the output
        scores = self.fc(output.squeeze(0))  # Squeeze the first dimension (time steps)

        # scores shape is (batch_size, num_classes)
        return scores

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype))


class ActionRecognition:
    _instance = None
    _feature_extractor = None
    _lstm = None

    def __new__(cls, weights_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if weights_path:
                print("Initializing models...")
                # Initialize feature extractor first (this is pre-trained and doesn't need loading)
                print("Loading feature extractor...")
                cls._feature_extractor = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
                cls._feature_extractor.fc = Identity()
                cls._feature_extractor.to('cuda' if torch.cuda.is_available() else 'cpu')
                cls._feature_extractor.eval()
                print("Feature extractor loaded successfully")

                # Now load the LSTM weights
                print("Loading LSTM model...")
                cls._lstm = LSTM_model(num_classes=3)
                cls._lstm.to('cuda' if torch.cuda.is_available() else 'cpu')

                print("Loading weights from:", weights_path)
                try:
                    # Add numpy scalar, dtype and Float64DType to safe globals
                    add_safe_globals([scalar, dtype, Float64DType])

                    # Try loading with weights_only=True first
                    saved_state = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True)
                    print("Weights loaded successfully")

                    if isinstance(saved_state, dict) and 'model_state' in saved_state:
                        print("Loading state dict from model_state key")
                        cls._lstm.load_state_dict(saved_state['model_state'])
                    else:
                        print("Loading complete state dict")
                        cls._lstm.load_state_dict(saved_state)

                    cls._lstm.eval()
                    print("LSTM model initialized successfully")
                except Exception as e:
                    print(f"Error loading weights: {str(e)}")
                    raise

        return cls._instance

    def __init__(self, weights_path):
        """Initialize the models and load weights"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = self._feature_extractor
        self.lstm = self._lstm

        # Initialize parameters
        self.box_margin = 150
        self.frames_features_seq = None
        self.max_seq_len = 90
        self.strokes_label = ['forehand', 'backhand', 'service']

        # Set up transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.softmax = nn.Softmax(dim=1)

    def add_frame(self, frame, player_box):
        """
        Extract frame features using feature extractor model and add the results to the frames until now.
        """
        # ROI is a small box around the player
        box_center = center_of_box(player_box)

        # Calculate the coordinates of the patch around the player (ROI)
        top = int(box_center[1] - self.box_margin)
        bottom = int(box_center[1] + self.box_margin)
        left = int(box_center[0] - self.box_margin)
        right = int(box_center[0] + self.box_margin)

        # Ensure the patch is within the frame boundaries
        top = max(top, 0)
        bottom = min(bottom, frame.shape[0])
        left = max(left, 0)
        right = min(right, frame.shape[1])

        # Extract the patch (ROI) around the player
        patch = frame[top:bottom, left:right].copy()

        # Check if the patch has valid dimensions
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            print("Invalid patch: Skipping this frame.")
            return

        # Resize the patch to match the input size of the model
        patch = imutils.resize(patch, 299)

        # Preprocess the patch for the model
        frame_t = patch.transpose((2, 0, 1)) / 255.0
        frame_tensor = torch.from_numpy(frame_t).type(self.dtype).to(self.device)
        frame_tensor = self.normalize(frame_tensor).unsqueeze(0)

        with torch.no_grad():
            # Forward pass: Extract features
            features = self.feature_extractor(frame_tensor)

        # Ensure features are on the correct device
        features = features.unsqueeze(1).to(self.device)

        # Concatenate the new features to the existing sequence
        if self.frames_features_seq is None:
            self.frames_features_seq = features
        else:
            self.frames_features_seq = torch.cat([self.frames_features_seq, features], dim=1)

        # Ensure the sequence does not exceed the maximum length
        if self.frames_features_seq.size(1) > self.max_seq_len:
            remove = self.frames_features_seq[:, 0, :]
            remove = remove.detach().cpu()  # Detach and move to CPU for cleanup
            self.frames_features_seq = self.frames_features_seq[:, 1:, :]

    def predict_saved_seq(self, clear=True):
        """
        Use saved sequence and predict the stroke
        """
        with torch.no_grad():
            scores = self.lstm(self.frames_features_seq)[-1].unsqueeze(0)
            probs = self.softmax(scores).squeeze().cpu().numpy()

        if clear:
            self.frames_features_seq = None
        return probs, self.strokes_label[np.argmax(probs)]

    def predict_stroke(self, frame, player_box):
        """
        Predict the stroke for each frame
        """
        # Calculate the center of the player's bounding box
        box_center = center_of_box(player_box)

        # Extract a patch around the box center (ROI)
        top = int(box_center[1] - self.box_margin)
        bottom = int(box_center[1] + self.box_margin)
        left = int(box_center[0] - self.box_margin)
        right = int(box_center[0] + self.box_margin)

        # Ensure the patch is within the frame boundaries
        top = max(top, 0)
        bottom = min(bottom, frame.shape[0])
        left = max(left, 0)
        right = min(right, frame.shape[1])

        # Extract the patch (ROI) around the player
        patch = frame[top:bottom, left:right].copy()

        # Check if the patch has valid dimensions
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            print("Invalid patch: Skipping this frame.")
            return None, None

        # Resize the patch to match the input size of the model
        patch = imutils.resize(patch, 299)

        # Preprocess the patch for the model
        frame_t = patch.transpose((2, 0, 1)) / 255.0
        frame_tensor = torch.from_numpy(frame_t).type(self.dtype).to(self.device)
        frame_tensor = self.normalize(frame_tensor).unsqueeze(0)

        with torch.no_grad():
            # Forward pass: Extract features
            features = self.feature_extractor(frame_tensor)

        # Ensure features are on the correct device
        features = features.unsqueeze(1).to(self.device)

        # Manage the sequence of features
        if self.frames_features_seq is None:
            self.frames_features_seq = features
        else:
            self.frames_features_seq = torch.cat([self.frames_features_seq, features], dim=1)

        # Ensure the sequence does not exceed the maximum length
        if self.frames_features_seq.size(1) > self.max_seq_len:
            remove = self.frames_features_seq[:, 0, :]
            remove = remove.detach().cpu()  # Detach and move to CPU for cleanup
            self.frames_features_seq = self.frames_features_seq[:, 1:, :]

        with torch.no_grad():
            # Pass the sequence through the LSTM and calculate scores
            scores = self.lstm(self.frames_features_seq)[-1].unsqueeze(0)
            probs = self.softmax(scores).squeeze().cpu().numpy()

        # Return the probabilities and the predicted stroke label
        return probs, self.strokes_label[np.argmax(probs)]


def create_features_from_vids():
    """
    Use feature extractor model to create features for each video in the stroke dataset
    """

    dtype = get_dtype()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Define device
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    feature_extractor.type(dtype)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = StrokesDataset('../dataset/my_dataset/patches/labels.csv', '../dataset/my_dataset/patches/',
                            transform=transforms.Compose([ToTensor(), normalize]), use_features=False)
    batch_size = 32
    count = 0
    for vid in dataset:
        count += 1
        frames = vid['frames']
        print(len(frames))

        features = []
        for batch in frames.split(batch_size):
            batch = batch.type(dtype).to(device)  # Move batch to the MPS device
            with torch.no_grad():
                # forward pass
                batch_features = feature_extractor(batch)
                features.append(batch_features.cpu().numpy())

        df = pd.DataFrame(np.concatenate(features, axis=0))

        outfile_path = os.path.join('../dataset/my_dataset/patches/',  os.path.splitext(vid['vid_name'])[0] + '.csv')
        df.to_csv(outfile_path, header=None, index=False)


        print(count)


if __name__ == "__main__":
    create_features_from_vids()
    '''batch = None
    video = cv2.VideoCapture('../videos/vid1.mp4')
    while True:
        ret, frame = video.read()
        if ret:
            frame_t = frame.transpose((2, 0, 1)) / 255
            frame_tensor = torch.from_numpy(frame_t).type(dtype)
            frame_tensor = normalize(frame_tensor).unsqueeze(0)
            with torch.no_grad():
                # forward pass
                features = feature_extractor(frame_tensor)
            features = features.unsqueeze(1)
            if batch is None:
                batch = features
            else:
                batch = torch.cat([batch, features], dim=1)
            if batch.size(1) > seq_len:
                # TODO this might be problem, need to get the vector out of gpu
                remove = batch[:,0,:]
                remove.detach().cpu()
                batch = batch[:, 1:, :]
                output = model(batch)
        else:
            break
    video.release()

    cv2.destroyAllWindows()'''
