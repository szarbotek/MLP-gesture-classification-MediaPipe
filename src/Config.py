from pathlib import Path
from enum import Enum
import os

from fontTools.varLib.avar.plan import SAMPLES

from src._T_typing import _T_cmap_colorHex, _T_root

PROJECT_ROOT: _T_root = (Path(__file__).resolve().parent / "Config.py").parents[1]
PROJECT_NAME = "({})".format( str(PROJECT_ROOT).split("/")[-1] )

# ------------------------------------------------------------------
# Specify
# ------------------------------------------------------------------
PROJECT_PATH_TRAINING_DATASET = Path("/mnt/BARACUDA/projects/artkuły_inż/codon/training/")
PROJECT_PATH_MEDIAPIPE_MODEL = PROJECT_ROOT / Path("data/models/hand_landmarker.task")
SAMPLES_LIMIT = 1000
# ------------------------------------------------------------------
# Specify
# ------------------------------------------------------------------

class DataPackageFileNames(str, Enum):
    training_images_classes_dataset= "training_images_classes_dataset"
    points = "points"
    labels = "labels"


PROJECT_MLP_ACCESSIBLE_CLASSES = [
    "grip",
    "one",
    "rock",
    "three3",
    "thumb_index",
    "fist",
    "dislike",
    "stop",
    "peace",
    "three",
    "call",
    "little_finger",
    "like",
]

class PROJECT_CLASS_LABELS(str, Enum):
    GRIP = "grip"
    ONE = "one"
    ROCK = "rock"
    THREE3 = "three3"
    THUMB_INDEX = "thumb_index"
    FIST = "fist"
    DISLIKE = "dislike"
    STOP = "stop"
    PEACE = "peace"
    THREE = "three"
    CALL = "call"
    LITTLE_FINGER = "little_finger"
    LIKE = "like"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

from types import MappingProxyType

COLOR_MAP: _T_cmap_colorHex = MappingProxyType({
    PROJECT_CLASS_LABELS.GRIP: "#E63946",
    PROJECT_CLASS_LABELS.ONE: "#F4A261",
    PROJECT_CLASS_LABELS.ROCK: "#E9C46A",
    PROJECT_CLASS_LABELS.THREE3: "#2A9D8F",
    PROJECT_CLASS_LABELS.THUMB_INDEX: "#264653",
    PROJECT_CLASS_LABELS.FIST: "#457B9D",
    PROJECT_CLASS_LABELS.DISLIKE: "#1D3557",
    PROJECT_CLASS_LABELS.STOP: "#D62828",
    PROJECT_CLASS_LABELS.PEACE: "#6A4C93",
    PROJECT_CLASS_LABELS.THREE: "#8338EC",
    PROJECT_CLASS_LABELS.CALL: "#3A86FF",
    PROJECT_CLASS_LABELS.LITTLE_FINGER: "#00B4D8",
    PROJECT_CLASS_LABELS.LIKE: "#2DC653",
})

class set_up:
    @staticmethod
    def environ():
        os.environ["EGL_PLATFORM"] = "surfaceless"
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
