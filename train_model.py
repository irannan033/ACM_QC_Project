"""
Optimized Two-Area Panel QC Counter
====================================

IMPORTANT STARTUP INSTRUCTION
-----------------------------
Keep both work areas clear and keep hands outside the areas while the
application performs its initial background calibration.

KEYBOARD
--------
Q = Quit
C = Reset all counts
R = Reset workflow; keep main count
M = Show/hide masks
B = Recalibrate panel backgrounds

PERFORMANCE
-----------
The application requests 60 FPS from a webcam. Actual FPS depends on:

1. Camera support for 60 FPS at the requested resolution.
2. USB bandwidth.
3. CPU/GPU performance.
4. Lighting and camera exposure time.

Recommended command:

python train_model.py --camera-fps 60 --process-width 640 \
    --hand-detect-every 2 --reselect

If the camera is already correctly oriented:

python train_model.py --camera-fps 60 --process-width 640 \
    --hand-detect-every 2 --keep-camera-orientation
"""

import argparse
import csv
import json
import math
import os
import ssl
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from qc_db import QCDatabase  # SQLite storage (qc_db.py, same folder)


# ===========================================================================
# Dependencies
# ===========================================================================

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ModuleNotFoundError as error:
    print("\nERROR: A required package is missing:")
    print(error)
    print("\nInstall dependencies with:")
    print(
        "python -m pip install --upgrade "
        "mediapipe opencv-contrib-python numpy"
    )
    sys.exit(1)


# Enable optimized OpenCV routines.
cv2.setUseOptimized(True)

try:
    number_of_threads = max(1, (os.cpu_count() or 4) - 1)
    cv2.setNumThreads(number_of_threads)
except Exception:
    pass

try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
except Exception:
    pass


# ===========================================================================
# Project configuration
# ===========================================================================

PROJECT_DIRECTORY = Path(__file__).resolve().parent

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

DEFAULT_MODEL_FILE = PROJECT_DIRECTORY / "hand_landmarker.task"
DEFAULT_ROI_FILE = PROJECT_DIRECTORY / "two_area_rois.json"
DEFAULT_COUNT_FILE = PROJECT_DIRECTORY / "inspection_count.json"
DEFAULT_CSV_FILE = PROJECT_DIRECTORY / "inspection_events.csv"
DEFAULT_DB_FILE = PROJECT_DIRECTORY / "qc_events.db"

MAIN_WINDOW_NAME = "Panel QC Inspection Counter"
DEBUG_WINDOW_NAME = "Panel Detection Debug"

PALM_SAMPLE_INDICES = [0, 5, 9, 13, 17]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ===========================================================================
# Model management
# ===========================================================================

def model_is_valid(path):
    path = Path(path)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 1_000_000
    )


def download_model_with_python(url, destination):
    context = ssl.create_default_context()

    if hasattr(ssl, "VERIFY_X509_STRICT"):
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Panel-QC-Counter"}
    )

    with urllib.request.urlopen(
        request,
        context=context,
        timeout=180
    ) as response:
        with open(destination, "wb") as output_file:
            while True:
                data = response.read(1024 * 1024)

                if not data:
                    break

                output_file.write(data)


def ensure_model(path):
    path = Path(path).resolve()
    temporary_path = Path(str(path) + ".part")

    if model_is_valid(path):
        print(f"Using MediaPipe model: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    for file_path in [path, temporary_path]:
        if file_path.exists():
            file_path.unlink()

    print("Downloading MediaPipe hand model...")

    try:
        download_model_with_python(
            MODEL_URL,
            temporary_path
        )

        if not model_is_valid(temporary_path):
            raise RuntimeError("Downloaded model is incomplete.")

        os.replace(temporary_path, path)

        print(f"Model downloaded: {path}")
        return path

    except Exception as python_error:
        print(f"Python download failed: {python_error}")

        if temporary_path.exists():
            temporary_path.unlink()

        if os.name == "nt":
            try:
                subprocess.run(
                    [
                        "curl.exe",
                        "-L",
                        "--fail",
                        "--retry",
                        "3",
                        "--connect-timeout",
                        "30",
                        "--output",
                        str(temporary_path),
                        MODEL_URL
                    ],
                    check=True
                )

                if not model_is_valid(temporary_path):
                    raise RuntimeError(
                        "curl downloaded an incomplete model."
                    )

                os.replace(temporary_path, path)
                return path

            except Exception as curl_error:
                print(f"curl download failed: {curl_error}")

    raise RuntimeError(
        "\nCould not obtain the MediaPipe model.\n"
        f"Download it manually from:\n{MODEL_URL}\n\n"
        f"Save it as:\n{path}\n"
    )


# ===========================================================================
# Video source
# ===========================================================================

def parse_video_source(value):
    value = str(value).strip()

    if value.isdigit():
        return int(value)

    return value


def open_video_source(
    source,
    width,
    height,
    requested_fps
):
    if isinstance(source, int) and os.name == "nt":
        capture = cv2.VideoCapture(
            source,
            cv2.CAP_DSHOW
        )

        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        return capture

    if isinstance(source, int):
        # MJPG usually provides better access to 60 FPS webcam modes.
        try:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG")
            )
        except Exception:
            pass

        if width > 0:
            capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                width
            )

        if height > 0:
            capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                height
            )

        if requested_fps > 0:
            capture.set(
                cv2.CAP_PROP_FPS,
                requested_fps
            )

        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    return capture


def correct_camera_orientation(
    frame,
    is_camera,
    unmirror_camera
):
    if is_camera and unmirror_camera:
        return cv2.flip(frame, 1)

    return frame


# ===========================================================================
# ROI management
# ===========================================================================

def normalize_rectangle(
    rectangle,
    frame_width,
    frame_height
):
    x, y, width, height = rectangle

    return [
        x / frame_width,
        y / frame_height,
        width / frame_width,
        height / frame_height
    ]


def restore_rectangle(
    rectangle,
    frame_width,
    frame_height
):
    x, y, width, height = rectangle

    return (
        int(round(x * frame_width)),
        int(round(y * frame_height)),
        int(round(width * frame_width)),
        int(round(height * frame_height))
    )


def scale_rectangle(
    rectangle,
    scale_x,
    scale_y
):
    x, y, width, height = rectangle

    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(width * scale_x))),
        max(1, int(round(height * scale_y)))
    )


def save_rois(
    path,
    rois,
    frame_width,
    frame_height
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        name: normalize_rectangle(
            rectangle,
            frame_width,
            frame_height
        )
        for name, rectangle in rois.items()
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_rois(
    path,
    frame_width,
    frame_height
):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for name in ["work", "checked"]:
        if name not in data:
            raise ValueError(
                f"ROI configuration is missing '{name}'."
            )

    return {
        name: restore_rectangle(
            data[name],
            frame_width,
            frame_height
        )
        for name in ["work", "checked"]
    }


def select_rectangle(
    frame,
    window_name,
    instruction
):
    frame_height, frame_width = frame.shape[:2]

    display_scale = min(
        1.0,
        1280.0 / max(frame_width, 1),
        800.0 / max(frame_height, 1)
    )

    if display_scale < 1.0:
        preview = cv2.resize(
            frame,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        preview = frame.copy()

    cv2.rectangle(
        preview,
        (0, 0),
        (preview.shape[1], 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        preview,
        instruction,
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    selected = cv2.selectROI(
        window_name,
        preview,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    x, y, width, height = selected

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"No valid rectangle selected for {window_name}."
        )

    return (
        int(round(x / display_scale)),
        int(round(y / display_scale)),
        int(round(width / display_scale)),
        int(round(height / display_scale))
    )


def select_two_areas(frame):
    print("\nSelect Area 1 and Area 2.")
    print("Press ENTER or SPACE after each selection.\n")

    return {
        "work": select_rectangle(
            frame,
            "Area 1",
            "Select AREA 1: BUNDLE / SOURCE + INSPECTION"
        ),
        "checked": select_rectangle(
            frame,
            "Area 2",
            "Select AREA 2: CHECKED PANEL"
        )
    }


# ===========================================================================
# Geometry
# ===========================================================================

def rectangles_intersect(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    return not (
        ax + aw < bx
        or bx + bw < ax
        or ay + ah < by
        or by + bh < ay
    )


def expand_rectangle(
    rectangle,
    padding_ratio,
    frame_width,
    frame_height
):
    x, y, width, height = rectangle

    horizontal_padding = int(
        width * padding_ratio
    )

    vertical_padding = int(
        height * padding_ratio
    )

    x1 = max(0, x - horizontal_padding)
    y1 = max(0, y - vertical_padding)

    x2 = min(
        frame_width,
        x + width + horizontal_padding
    )

    y2 = min(
        frame_height,
        y + height + vertical_padding
    )

    return (
        x1,
        y1,
        max(1, x2 - x1),
        max(1, y2 - y1)
    )


def any_hand_near_area(
    hand_rectangles,
    area,
    padding_ratio,
    frame_width,
    frame_height
):
    expanded = expand_rectangle(
        area,
        padding_ratio,
        frame_width,
        frame_height
    )

    return any(
        rectangles_intersect(
            hand_rectangle,
            expanded
        )
        for hand_rectangle in hand_rectangles
    )


def point_in_rectangle(
    point_x,
    point_y,
    rectangle
):
    x, y, width, height = rectangle

    return (
        x <= point_x <= x + width
        and y <= point_y <= y + height
    )


# ===========================================================================
# MediaPipe hand detection
# ===========================================================================

def prepare_mediapipe_image(frame):
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_frame)
    )


def extract_hands(
    result,
    frame_width,
    frame_height
):
    centers = []
    all_points = []
    rectangles = []

    if result is None or not result.hand_landmarks:
        return centers, all_points, rectangles

    for hand_landmarks in result.hand_landmarks:
        points = []

        for landmark in hand_landmarks:
            point_x = int(
                np.clip(
                    landmark.x * frame_width,
                    0,
                    frame_width - 1
                )
            )

            point_y = int(
                np.clip(
                    landmark.y * frame_height,
                    0,
                    frame_height - 1
                )
            )

            points.append((point_x, point_y))

        palm_points = [
            points[index]
            for index in PALM_SAMPLE_INDICES
        ]

        center_x = int(
            np.mean([point[0] for point in palm_points])
        )

        center_y = int(
            np.mean([point[1] for point in palm_points])
        )

        points_array = np.asarray(
            points,
            dtype=np.int32
        )

        centers.append((center_x, center_y))
        all_points.append(points)

        rectangles.append(
            cv2.boundingRect(points_array)
        )

    return centers, all_points, rectangles


def largest_hand_information(hand_rectangles):
    information = {
        "area": 0,
        "width": 0,
        "height": 0
    }

    for _, _, width, height in hand_rectangles:
        area = width * height

        if area > information["area"]:
            information = {
                "area": area,
                "width": width,
                "height": height
            }

    return information


def create_hand_mask(
    frame_shape,
    all_hand_points,
    expansion_ratio,
    minimum_padding
):
    frame_height, frame_width = frame_shape[:2]

    complete_mask = np.zeros(
        (frame_height, frame_width),
        dtype=np.uint8
    )

    for points in all_hand_points:
        points_array = np.asarray(
            points,
            dtype=np.int32
        )

        hull = cv2.convexHull(points_array)

        current_mask = np.zeros_like(
            complete_mask
        )

        cv2.fillConvexPoly(
            current_mask,
            hull,
            255
        )

        _, _, hand_width, hand_height = (
            cv2.boundingRect(points_array)
        )

        padding = max(
            int(minimum_padding),
            int(
                max(hand_width, hand_height)
                * expansion_ratio
            )
        )

        kernel_size = max(
            3,
            padding * 2 + 1
        )

        kernel_size = min(
            kernel_size,
            51
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        current_mask = cv2.dilate(
            current_mask,
            kernel,
            iterations=1
        )

        complete_mask = cv2.bitwise_or(
            complete_mask,
            current_mask
        )

    return complete_mask


def create_tight_hand_mask(
    frame_shape,
    all_hand_points,
    minimum_padding=2
):
    """
    Small hand mask used ONLY to protect panel detection.
    The full dilated mask is too large and erases fabric held in the hand.
    """
    return create_hand_mask(
        frame_shape,
        all_hand_points,
        expansion_ratio=0.02,
        minimum_padding=minimum_padding
    )


# ===========================================================================
# Hand colour and motion tracking
# ===========================================================================

class HandColourTracker:
    def __init__(self):
        self.reference_lab = None

    def reset(self):
        self.reference_lab = None

    def update(
        self,
        lab_frame,
        all_hand_points
    ):
        samples = []
        height, width = lab_frame.shape[:2]

        for points in all_hand_points:
            for index in PALM_SAMPLE_INDICES:
                point_x, point_y = points[index]

                x1 = max(0, point_x - 2)
                y1 = max(0, point_y - 2)

                x2 = min(width, point_x + 3)
                y2 = min(height, point_y + 3)

                sample = lab_frame[y1:y2, x1:x2]

                if sample.size:
                    samples.append(
                        sample.reshape(-1, 3)
                    )

        if samples:
            combined = np.concatenate(
                samples,
                axis=0
            ).astype(np.float32)

            current_reference = np.median(
                combined,
                axis=0
            )

            if self.reference_lab is None:
                self.reference_lab = current_reference
            else:
                self.reference_lab = (
                    0.90 * self.reference_lab
                    + 0.10 * current_reference
                )

        return self.reference_lab


class HandMovementTracker:
    def __init__(
        self,
        frame_width,
        frame_height
    ):
        self.frame_diagonal = max(
            1.0,
            math.hypot(
                frame_width,
                frame_height
            )
        )

        self.previous_centers = []
        self.previous_update_time = None
        self.smoothed_speed = 0.0

    @staticmethod
    def distance(first, second):
        return math.hypot(
            first[0] - second[0],
            first[1] - second[1]
        )

    def reset(self):
        self.previous_centers = []
        self.previous_update_time = None
        self.smoothed_speed = 0.0

    def update(
        self,
        current_centers,
        current_time
    ):
        if self.previous_update_time is None:
            delta_time = 1.0 / 30.0
        else:
            delta_time = max(
                current_time - self.previous_update_time,
                0.001
            )

        self.previous_update_time = current_time

        if current_centers and self.previous_centers:
            distances = []

            for current_center in current_centers:
                nearest_distance = min(
                    self.distance(
                        current_center,
                        previous_center
                    )
                    for previous_center
                    in self.previous_centers
                )

                distances.append(nearest_distance)

            displacement = float(
                np.median(distances)
            )

            current_speed = (
                displacement
                / self.frame_diagonal
                / delta_time
            )

            current_speed = min(
                current_speed,
                5.0
            )

            self.smoothed_speed = (
                0.65 * self.smoothed_speed
                + 0.35 * current_speed
            )

        elif current_centers:
            self.smoothed_speed *= 0.75

        else:
            self.smoothed_speed *= 0.40

        self.previous_centers = list(
            current_centers
        )

        return self.smoothed_speed


# ===========================================================================
# Short-term frame motion
# ===========================================================================

class FrameMotionDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.previous_gray = None

    def reset(self):
        self.previous_gray = None

    def update(self, gray_frame):
        if self.previous_gray is None:
            self.previous_gray = gray_frame.copy()

            return np.zeros_like(
                gray_frame
            )

        difference = cv2.absdiff(
            self.previous_gray,
            gray_frame
        )

        self.previous_gray = gray_frame.copy()

        _, motion_mask = cv2.threshold(
            difference,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )

        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            opening_kernel
        )

        motion_mask = cv2.dilate(
            motion_mask,
            opening_kernel,
            iterations=1
        )

        return motion_mask


# ===========================================================================
# Background-calibrated panel detector
# ===========================================================================

class PanelPresenceDetector:
    """
    White-table panel detector (no background calibration).

    Rule:
      - Table is always WHITE
      - PANEL = fabric of ANY colour (dark, bright, pastel, patterned)
        detected as non-white / colourful / darker than the table
      - Hands are only lightly filtered so fabric colour is kept
      - Count when a panel moves Bundle (AREA 1) -> Checked (AREA 2)
    """

    def __init__(
        self,
        rectangle,
        calibration_samples,
        background_distance,
        minimum_pixel_ratio,
        minimum_span_ratio,
        minimum_dimension_ratio,
        minimum_hand_colour_ratio,
        panel_motion_ratio,
        background_learning_rate,
        white_l_min=175.0,
        white_chroma_max=22.0,
    ):
        self.rectangle = rectangle
        self.calibration_samples = max(3, int(calibration_samples))
        # background_distance kept for CLI compatibility; maps to white sensitivity
        self.background_distance = float(background_distance)
        self.minimum_pixel_ratio = float(minimum_pixel_ratio)
        self.minimum_span_ratio = float(minimum_span_ratio)
        self.minimum_dimension_ratio = float(minimum_dimension_ratio)
        self.minimum_hand_colour_ratio = float(minimum_hand_colour_ratio)
        self.panel_motion_ratio = float(panel_motion_ratio)
        self.background_learning_rate = float(background_learning_rate)

        # White table thresholds in OpenCV LAB (L,a,b all 0..255, neutral a/b = 128)
        self.white_l_min = float(white_l_min)
        self.white_chroma_max = float(white_chroma_max)

        self.samples_collected = 0
        self._learned_l_min = None
        self.previous_centroid = None
        self.smoothed_centroid_motion = 0.0

    @property
    def calibrated(self):
        # White table: no background calibration required
        return True

    @property
    def calibration_progress(self):
        return 1.0

    def reset_background(self):
        # Kept for B key compatibility; does not block detection
        self.samples_collected = 0
        self._learned_l_min = None
        self.previous_centroid = None
        self.smoothed_centroid_motion = 0.0

    def _empty_result(self, frame_shape, status):
        return {
            "detected": False,
            "moving": False,
            "pixel_area": 0,
            "span_area": 0,
            "pixel_ratio": 0.0,
            "span_ratio": 0.0,
            "motion_ratio": 0.0,
            "hand_colour_ratio": 1.0,
            "centroid_motion": 0.0,
            "mask": np.zeros(frame_shape[:2], dtype=np.uint8),
            "status": status,
        }

    def _effective_white_l_min(self):
        # Lower L min slightly if table is off-white / greyish under weak light
        if self._learned_l_min is not None:
            return float(self._learned_l_min)
        # background_distance CLI: higher = stricter white (need brighter)
        # map typical 8..16 -> small adjustment
        adj = max(0.0, (16.0 - self.background_distance) * 1.5)
        return max(130.0, self.white_l_min - adj)

    def update(
        self,
        lab_frame,
        frame_motion_mask,
        hand_mask,
        hand_reference_lab,
        hand_near_area
    ):
        frame_height, frame_width = lab_frame.shape[:2]
        x, y, width, height = self.rectangle

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_width, x + width)
        y2 = min(frame_height, y + height)
        roi_width = x2 - x1
        roi_height = y2 - y1

        if roi_width <= 2 or roi_height <= 2:
            return self._empty_result(lab_frame.shape, "INVALID ROI")

        roi_lab = lab_frame[y1:y2, x1:x2]
        roi_hand = hand_mask[y1:y2, x1:x2]
        roi_motion = frame_motion_mask[y1:y2, x1:x2]

        L = roi_lab[:, :, 0].astype(np.float32)
        A = roi_lab[:, :, 1].astype(np.float32)
        B = roi_lab[:, :, 2].astype(np.float32)
        chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

        white_l = self._effective_white_l_min()

        # WHITE table = bright + low colour (near grey/white only)
        is_white = (L >= white_l) & (chroma <= self.white_chroma_max)

        # PANEL = any different colour fabric on the white table:
        #   - dark fabric (black, navy, grey)  -> low L
        #   - bright / dyed fabric            -> high chroma (any hue)
        #   - pastel fabric                   -> moderate chroma or mid L not pure white
        #   - light coloured fabric           -> fails strict white test
        is_dark = L < (white_l - 25.0)
        is_colourful = chroma >= max(12.0, self.white_chroma_max * 0.45)
        is_not_white = ~is_white

        is_panel = is_not_white | is_dark | is_colourful
        panel_mask = is_panel.astype(np.uint8) * 255

        # Remove only tight skin-like hand pixels (keep coloured fabric)
        if cv2.countNonZero(roi_hand) > 0:
            if hand_reference_lab is not None:
                hand_diff = roi_lab.astype(np.float32) - hand_reference_lab.reshape(1, 1, 3)
                hand_dist = np.sqrt(np.sum(hand_diff * hand_diff, axis=2))
                skin = (hand_dist < 25.0).astype(np.uint8) * 255
                suppress = cv2.bitwise_and(roi_hand, skin)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                suppress = cv2.erode(suppress, k, iterations=1)
                panel_mask = cv2.bitwise_and(panel_mask, cv2.bitwise_not(suppress))
            else:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                tight = cv2.erode(roi_hand, k, iterations=2)
                panel_mask = cv2.bitwise_and(panel_mask, cv2.bitwise_not(tight))

        # Clean mask
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN, k3)
        panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, k7)
        panel_mask = cv2.dilate(panel_mask, k3, iterations=1)

        roi_area = max(1, roi_width * roi_height)
        pixel_area = int(cv2.countNonZero(panel_mask))

        if pixel_area <= 0:
            self.previous_centroid = None
            self.smoothed_centroid_motion *= 0.6
            return self._empty_result(lab_frame.shape, "NO PANEL")

        pts = cv2.findNonZero(panel_mask)
        bx, by, bw, bh = cv2.boundingRect(pts)
        span_area = bw * bh
        pixel_ratio = pixel_area / roi_area
        span_ratio = span_area / roi_area
        width_ratio = bw / max(roi_width, 1)
        height_ratio = bh / max(roi_height, 1)

        moments = cv2.moments(panel_mask, binaryImage=True)
        if moments["m00"] > 0:
            centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        else:
            centroid = (bx + bw / 2.0, by + bh / 2.0)

        if self.previous_centroid is None:
            centroid_motion = 0.0
        else:
            centroid_motion = math.hypot(
                centroid[0] - self.previous_centroid[0],
                centroid[1] - self.previous_centroid[1],
            )
        self.previous_centroid = centroid
        self.smoothed_centroid_motion = (
            0.65 * self.smoothed_centroid_motion + 0.35 * centroid_motion
        )

        moving_pixels = int(cv2.countNonZero(cv2.bitwise_and(panel_mask, roi_motion)))
        motion_ratio = moving_pixels / roi_area

        dimension_ok = (
            width_ratio >= self.minimum_dimension_ratio
            or height_ratio >= self.minimum_dimension_ratio
        )

        raw_detected = (
            pixel_ratio >= self.minimum_pixel_ratio
            and span_ratio >= self.minimum_span_ratio
            and dimension_ok
            and pixel_ratio < 0.98
        )

        centroid_thr = max(0.75, min(roi_width, roi_height) * 0.0025)
        panel_is_moving = raw_detected and (
            motion_ratio >= self.panel_motion_ratio
            or self.smoothed_centroid_motion >= centroid_thr
        )

        global_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        global_mask[y1:y2, x1:x2] = panel_mask

        return {
            "detected": raw_detected,
            "moving": panel_is_moving,
            "pixel_area": pixel_area,
            "span_area": span_area,
            "pixel_ratio": pixel_ratio,
            "span_ratio": span_ratio,
            "motion_ratio": motion_ratio,
            "hand_colour_ratio": 1.0,
            "centroid_motion": self.smoothed_centroid_motion,
            "mask": global_mask,
            "status": (
                "MOVING PANEL"
                if panel_is_moving
                else ("PANEL PRESENT" if raw_detected else "BACKGROUND CHANGE")
            ),
        }


class TransferCounter:
    """
    Count when a coloured panel moves Bundle (AREA 1) -> Checked (AREA 2).

    Edge-triggered (works when both areas keep showing MOVING PANEL):
      WAIT_SOURCE : rising edge of bundle motion  -> ARMED
      ARMED       : rising edge of checked motion -> COUNT +1
      Then short cooldown, ready for the next panel.

    Does not require the panel to fully disappear or the operator to hold still.
    """

    WAIT_SOURCE = "WAITING: PANEL MOVE IN BUNDLE"
    ARMED = "ARMED: MOVE PANEL TO CHECKED"
    COOLDOWN = "COUNTED: COOLDOWN"

    def __init__(
        self,
        starting_count,
        source_confirmation,
        checked_confirmation,
        return_confirmation,
        transfer_timeout,
        count_cooldown=2.5,
        clear_seconds=0.5,
    ):
        self.count = starting_count
        self.bundle_stage_count = 0
        self.checked_stage_count = 0

        self.source_confirmation = float(source_confirmation)
        self.checked_confirmation = float(checked_confirmation)
        self.return_confirmation = float(return_confirmation)
        self.transfer_timeout = float(transfer_timeout)
        self.count_cooldown = float(count_cooldown)
        self.clear_seconds = float(clear_seconds)

        self.state = self.WAIT_SOURCE
        self.evidence_time = 0.0
        self.armed_time = None
        self.last_count_time = None

        # Edge detection: previous frame motion flags
        self._prev_work_moving = False
        self._prev_checked_moving = False

    def reset_evidence(self):
        self.evidence_time = 0.0

    def update_evidence(self, condition, delta_time, decrease_rate=1.5):
        if condition:
            self.evidence_time += delta_time
        else:
            self.evidence_time = max(
                0.0, self.evidence_time - delta_time * decrease_rate
            )
        return self.evidence_time

    def reset_workflow(self):
        self.state = self.WAIT_SOURCE
        self.bundle_stage_count = 0
        self.checked_stage_count = 0
        self.armed_time = None
        self.reset_evidence()
        self._prev_work_moving = False
        self._prev_checked_moving = False

    def reset_all_counts(self):
        self.count = 0
        self.last_count_time = None
        self.reset_workflow()

    def update(
        self,
        current_time,
        delta_time,
        hand_count,
        hand_is_moving,
        hand_in_work_area,
        hand_in_checked_area,
        work_panel_detected,
        work_panel_moving,
        checked_panel_detected,
        checked_panel_moving
    ):
        event = None

        work_moving = bool(work_panel_detected and work_panel_moving)
        checked_moving = bool(checked_panel_detected and checked_panel_moving)

        # Rising edges: motion starts in that box
        work_rise = work_moving and not self._prev_work_moving
        checked_rise = checked_moving and not self._prev_checked_moving

        if self.state == self.WAIT_SOURCE:
            # Arm when panel motion starts (or continues briefly) in bundle
            if work_rise:
                self.evidence_time = max(self.evidence_time, 0.0)
            evidence = self.update_evidence(work_moving, delta_time)

            if work_rise or evidence >= self.source_confirmation:
                if work_moving:
                    self.state = self.ARMED
                    self.bundle_stage_count = 1
                    self.checked_stage_count = 0
                    self.armed_time = current_time
                    self.reset_evidence()
                    # Ignore checked edge on same frame as arm
                    self._prev_checked_moving = checked_moving
                    event = "BUNDLE_ACCEPTED"

        elif self.state == self.ARMED:
            if (
                self.armed_time is not None
                and (current_time - self.armed_time) > self.transfer_timeout
            ):
                self.reset_workflow()
                event = "TIMEOUT"
            else:
                # Count on rising edge of motion in CHECKED after armed
                # Also allow sustained checked motion shortly after arm (fast transfer)
                evidence = self.update_evidence(checked_moving, delta_time)

                trigger = checked_rise or (
                    checked_moving and evidence >= self.checked_confirmation
                )

                if trigger:
                    cooldown_ok = (
                        self.last_count_time is None
                        or (current_time - self.last_count_time)
                        >= self.count_cooldown
                    )
                    if cooldown_ok:
                        self.bundle_stage_count = 0
                        self.checked_stage_count = 1
                        self.count += 1
                        self.last_count_time = current_time
                        self.state = self.COOLDOWN
                        self.armed_time = None
                        self.reset_evidence()
                        event = "CHECKED_COUNTED"
                    else:
                        self.reset_evidence()

        elif self.state == self.COOLDOWN:
            # Fixed cooldown then ready for next panel — do not wait for settle
            if (
                self.last_count_time is not None
                and (current_time - self.last_count_time) >= self.count_cooldown
            ):
                self.bundle_stage_count = 0
                self.checked_stage_count = 0
                self.state = self.WAIT_SOURCE
                self.reset_evidence()
                event = "RETURNED_TO_BUNDLE"

        self._prev_work_moving = work_moving
        self._prev_checked_moving = checked_moving

        return event


# ===========================================================================
# Data storage
# ===========================================================================

def load_count(path):
    path = Path(path)

    if not path.exists():
        return 0

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return max(
            0,
            int(data.get("count", 0))
        )

    except Exception:
        return 0


def save_count(path, count):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = Path(
        str(path) + ".tmp"
    )

    data = {
        "count": int(count),
        "updated_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=4)

    os.replace(
        temporary_path,
        path
    )


def open_csv_log(path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    csv_file = open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    )

    fieldnames = [
        "inspection_count",
        "event",
        "wall_time_utc",
        "video_time_seconds"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    return csv_file, writer


def write_event(
    writer,
    csv_file,
    count,
    event,
    video_time
):
    writer.writerow({
        "inspection_count": count,
        "event": event,
        "wall_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "video_time_seconds": f"{video_time:.3f}"
    })

    csv_file.flush()


# ===========================================================================
# Popup
# ===========================================================================

class PopupNotification:
    def __init__(self):
        self.title = ""
        self.subtitle = ""
        self.colour = (0, 170, 0)
        self.expires_at = -1.0

    def show(
        self,
        title,
        subtitle,
        current_time,
        duration=1.5,
        colour=(0, 170, 0)
    ):
        self.title = title
        self.subtitle = subtitle
        self.colour = colour
        self.expires_at = (
            current_time + duration
        )

    def draw(
        self,
        frame,
        current_time
    ):
        if current_time > self.expires_at:
            return

        frame_height, frame_width = frame.shape[:2]

        popup_width = min(
            680,
            frame_width - 40
        )

        popup_height = 170

        x1 = max(
            10,
            (frame_width - popup_width) // 2
        )

        y1 = max(
            10,
            (frame_height - popup_height) // 2
        )

        x2 = min(
            frame_width - 10,
            x1 + popup_width
        )

        y2 = min(
            frame_height - 10,
            y1 + popup_height
        )

        cv2.rectangle(
            frame,
            (x1 + 6, y1 + 6),
            (x2 + 6, y2 + 6),
            (15, 15, 15),
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.colour,
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            self.title,
            (x1 + 28, y1 + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            self.subtitle,
            (x1 + 28, y1 + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


# ===========================================================================
# UI
# ===========================================================================

class UserInterfaceState:
    def __init__(self):
        self.reset_requested = False
        self.quit_requested = False

        self.reset_button_rectangle = (
            0, 0, 0, 0
        )

        self.quit_button_rectangle = (
            0, 0, 0, 0
        )

    def consume_reset_request(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def consume_quit_request(self):
        requested = self.quit_requested
        self.quit_requested = False
        return requested


def mouse_callback(
    event,
    point_x,
    point_y,
    flags,
    interface
):
    del flags

    if event != cv2.EVENT_LBUTTONUP:
        return

    if point_in_rectangle(
        point_x,
        point_y,
        interface.reset_button_rectangle
    ):
        interface.reset_requested = True

    elif point_in_rectangle(
        point_x,
        point_y,
        interface.quit_button_rectangle
    ):
        interface.quit_requested = True


def draw_area(
    frame,
    rectangle,
    label,
    colour,
    active,
    stage_count
):
    x, y, width, height = rectangle

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        colour,
        5 if active else 2
    )

    label_text = (
        f"{label} | STAGE: {stage_count}"
    )

    text_size, _ = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        2
    )

    label_width = min(
        width,
        text_size[0] + 16
    )

    cv2.rectangle(
        frame,
        (x, max(0, y - 30)),
        (x + label_width, y),
        colour,
        -1
    )

    cv2.putText(
        frame,
        label_text,
        (x + 7, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    badge_width = 55
    badge_height = 52

    badge_x = max(
        x,
        x + width - badge_width
    )

    cv2.rectangle(
        frame,
        (badge_x, y),
        (
            badge_x + badge_width,
            y + badge_height
        ),
        colour,
        -1
    )

    cv2.putText(
        frame,
        str(stage_count),
        (badge_x + 17, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )


def draw_hands(
    frame,
    all_hand_points,
    scale_x,
    scale_y
):
    for hand_index, points in enumerate(
        all_hand_points
    ):
        display_points = [
            (
                int(point[0] * scale_x),
                int(point[1] * scale_y)
            )
            for point in points
        ]

        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(
                frame,
                display_points[start_index],
                display_points[end_index],
                (255, 160, 0),
                2,
                cv2.LINE_AA
            )

        for point in display_points:
            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 255),
                -1
            )

        palm = display_points[0]

        cv2.putText(
            frame,
            f"H{hand_index + 1}",
            (palm[0] + 10, palm[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


def draw_button(
    frame,
    rectangle,
    text,
    colour
):
    x, y, width, height = rectangle

    cv2.rectangle(
        frame,
        (x + 4, y + 4),
        (x + width + 4, y + height + 4),
        (10, 10, 10),
        -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        colour,
        -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (255, 255, 255),
        2
    )

    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    cv2.putText(
        frame,
        text,
        (
            x + (width - text_size[0]) // 2,
            y + (height + text_size[1]) // 2 - 2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_buttons(
    frame,
    interface
):
    frame_width = frame.shape[1]

    button_width = min(
        190,
        max(145, frame_width // 7)
    )

    button_height = 50

    button_x = (
        frame_width
        - button_width
        - 18
    )

    interface.reset_button_rectangle = (
        button_x,
        18,
        button_width,
        button_height
    )

    interface.quit_button_rectangle = (
        button_x,
        80,
        button_width,
        button_height
    )

    draw_button(
        frame,
        interface.reset_button_rectangle,
        "RESET COUNT",
        (0, 0, 200)
    )

    draw_button(
        frame,
        interface.quit_button_rectangle,
        "QUIT APP",
        (45, 45, 45)
    )


def draw_dashboard(
    frame,
    transfer_counter,
    hand_count,
    hand_speed,
    hand_is_moving,
    work_panel,
    checked_panel,
    processing_fps,
    reported_camera_fps
):
    overlay = frame.copy()

    dashboard_width = min(
        frame.shape[1] - 8,
        850
    )

    cv2.rectangle(
        overlay,
        (8, 8),
        (dashboard_width, 310),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.76,
        frame,
        0.24,
        0,
        frame
    )

    lines = [
        (
            f"MAIN INSPECTION COUNT: "
            f"{transfer_counter.count}",
            (0, 255, 0),
            0.80
        ),
        (
            f"BUNDLE STAGE: "
            f"{transfer_counter.bundle_stage_count}   "
            f"CHECKED STAGE: "
            f"{transfer_counter.checked_stage_count}",
            (0, 255, 255),
            0.57
        ),
        (
            f"WORKFLOW: {transfer_counter.state}",
            (255, 255, 255),
            0.50
        ),
        (
            f"HANDS: {hand_count}   "
            f"MOVING: {'YES' if hand_is_moving else 'NO'}   "
            f"SPEED: {hand_speed:.3f}",
            (0, 220, 255),
            0.49
        ),
        (
            "BUNDLE PANEL: "
            f"{work_panel['status']}   "
            f"PIXEL={work_panel['pixel_ratio']:.3f}   "
            f"SPAN={work_panel['span_ratio']:.3f}",
            (
                (0, 255, 255)
                if work_panel["detected"]
                else (180, 180, 180)
            ),
            0.46
        ),
        (
            "CHECKED PANEL: "
            f"{checked_panel['status']}   "
            f"PIXEL={checked_panel['pixel_ratio']:.3f}   "
            f"SPAN={checked_panel['span_ratio']:.3f}",
            (
                (0, 255, 0)
                if checked_panel["detected"]
                else (180, 180, 180)
            ),
            0.46
        ),
        (
            f"PROCESSING FPS: {processing_fps:.1f}   "
            f"CAMERA REPORTED FPS: {reported_camera_fps:.1f}",
            (
                (0, 255, 0)
                if processing_fps >= 50
                else (0, 170, 255)
            ),
            0.47
        ),
        (
            "KEYS: C=COUNT RESET  R=WORKFLOW  "
            "B=BACKGROUND  M=MASKS  Q=QUIT",
            (190, 190, 190),
            0.42
        )
    ]

    text_y = 37

    for text, colour, scale in lines:
        cv2.putText(
            frame,
            text,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            2,
            cv2.LINE_AA
        )

        text_y += 36


def create_debug_view(
    work_panel,
    checked_panel,
    hand_mask
):
    debug_view = np.zeros(
        (
            hand_mask.shape[0],
            hand_mask.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # Red: Area 1 candidate.
    debug_view[:, :, 2] = work_panel["mask"]

    # Green: Area 2 candidate.
    debug_view[:, :, 1] = checked_panel["mask"]

    # Blue: excluded hand.
    debug_view[:, :, 0] = hand_mask

    return debug_view


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Background-calibrated two-area panel counter"
        )
    )

    parser.add_argument(
        "--source",
        default="0"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720
    )

    parser.add_argument(
        "--camera-fps",
        type=float,
        default=60.0,
        help="Requested webcam FPS"
    )

    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Internal processing width"
    )

    parser.add_argument(
        "--hand-detect-every",
        type=int,
        default=2,
        help="Run MediaPipe every N frames"
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_FILE)
    )

    parser.add_argument(
        "--roi-config",
        default=str(DEFAULT_ROI_FILE)
    )

    parser.add_argument(
        "--count-file",
        default=str(DEFAULT_COUNT_FILE)
    )

    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_FILE)
    )

    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_FILE),
        help="SQLite database used by the dashboard"
    )

    parser.add_argument(
        "--station",
        default="STATION-1",
        help="Name of this camera / work station in the database"
    )

    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database logging (CSV only)"
    )

    parser.add_argument(
        "--output",
        default=""
    )

    parser.add_argument(
        "--reselect",
        action="store_true"
    )

    parser.add_argument(
        "--reset-count",
        action="store_true"
    )

    parser.add_argument(
        "--no-display",
        action="store_true"
    )

    parser.set_defaults(
        unmirror_camera=True
    )

    parser.add_argument(
        "--keep-camera-orientation",
        dest="unmirror_camera",
        action="store_false"
    )

    parser.add_argument(
        "--unmirror-camera",
        dest="unmirror_camera",
        action="store_true"
    )

    # Hand configuration.
    parser.add_argument(
        "--hand-motion-threshold",
        type=float,
        default=0.012
    )

    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--hand-expansion",
        type=float,
        default=0.08
    )

    parser.add_argument(
        "--hand-minimum-padding",
        type=int,
        default=4
    )

    # Panel configuration.
    parser.add_argument(
        "--background-calibration-seconds",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--panel-background-distance",
        type=float,
        default=10.0,
        help="White-table sensitivity (lower = treat dimmer pixels as non-white panel)"
    )

    parser.add_argument(
        "--panel-pixel-ratio",
        type=float,
        default=0.01,
        help="Min fraction of ROI that must be coloured fabric (any colour)"
    )

    parser.add_argument(
        "--panel-span-ratio",
        type=float,
        default=0.04,
        help="Min bounding-box area of coloured panel / ROI"
    )

    parser.add_argument(
        "--panel-dimension-ratio",
        type=float,
        default=0.12,
        help="Min width or height of panel blob relative to ROI"
    )

    parser.add_argument(
        "--panel-hand-colour-ratio",
        type=float,
        default=0.04,
        help="Min share of panel pixels that are NOT hand-coloured (low = allow skin-tone fabric)"
    )

    parser.add_argument(
        "--panel-motion-ratio",
        type=float,
        default=0.00030
    )

    parser.add_argument(
        "--background-learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--pixel-difference",
        type=int,
        default=12
    )

    # Workflow timing.
    parser.add_argument(
        "--source-confirm-seconds",
        type=float,
        default=0.08,
        help="Seconds of moving panel in BUNDLE before arming (fast continuous OK)"
    )

    parser.add_argument(
        "--checked-confirm-seconds",
        type=float,
        default=0.10,
        help="Seconds of moving panel in CHECKED before count +1"
    )

    parser.add_argument(
        "--return-confirm-seconds",
        type=float,
        default=0.12,
        help="Seconds before next panel cycle (continuous checking)"
    )

    parser.add_argument(
        "--count-cooldown",
        type=float,
        default=0.65,
        help="Min seconds between counts (continuous checking)"
    )

    parser.add_argument(
        "--transfer-timeout",
        type=float,
        default=30.0
    )

    parser.add_argument(
        "--popup-seconds",
        type=float,
        default=1.5
    )

    args = parser.parse_args()

    args.hand_detect_every = max(
        1,
        args.hand_detect_every
    )

    model_path = ensure_model(
        args.model
    )

    source = parse_video_source(
        args.source
    )

    is_camera = isinstance(
        source,
        int
    )

    capture = open_video_source(
        source,
        args.width,
        args.height,
        args.camera_fps
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    success, first_frame = capture.read()

    if not success or first_frame is None:
        capture.release()
        raise RuntimeError(
            "Could not read the first frame."
        )

    first_frame = correct_camera_orientation(
        first_frame,
        is_camera,
        args.unmirror_camera
    )

    display_height, display_width = (
        first_frame.shape[:2]
    )

    # Lock size from the first frame. Some files report or decode
    # later frames at a different size; force every frame to match.
    target_size = (display_width, display_height)

    reported_camera_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not reported_camera_fps
        or reported_camera_fps <= 1
        or reported_camera_fps > 240
    ):
        reported_camera_fps = (
            args.camera_fps
            if is_camera
            else 25.0
        )

    print(
        f"Camera frame: {display_width} x {display_height}"
    )

    print(
        f"Requested FPS: {args.camera_fps:.1f}"
    )

    print(
        f"Camera-reported FPS: {reported_camera_fps:.1f}"
    )

    # ---------------------------------------------------------------------
    # ROIs
    # ---------------------------------------------------------------------

    roi_path = Path(
        args.roi_config
    )

    if roi_path.exists() and not args.reselect:
        try:
            display_rois = load_rois(
                roi_path,
                display_width,
                display_height
            )

            print(f"Loaded ROIs: {roi_path}")

        except Exception as error:
            print(f"ROI loading failed: {error}")

            if args.no_display:
                raise

            display_rois = select_two_areas(
                first_frame
            )

            save_rois(
                roi_path,
                display_rois,
                display_width,
                display_height
            )
    else:
        if args.no_display:
            raise RuntimeError(
                "ROI file does not exist. "
                "Run once with display enabled."
            )

        display_rois = select_two_areas(
            first_frame
        )

        save_rois(
            roi_path,
            display_rois,
            display_width,
            display_height
        )

    # ---------------------------------------------------------------------
    # Internal processing resolution
    # ---------------------------------------------------------------------

    if (
        args.process_width <= 0
        or display_width <= args.process_width
    ):
        process_width = display_width
        process_height = display_height
    else:
        process_width = args.process_width

        process_height = max(
            1,
            int(
                display_height
                * process_width
                / display_width
            )
        )

    process_scale_x = (
        process_width / display_width
    )

    process_scale_y = (
        process_height / display_height
    )

    display_scale_x = (
        display_width / process_width
    )

    display_scale_y = (
        display_height / process_height
    )

    process_rois = {
        name: scale_rectangle(
            rectangle,
            process_scale_x,
            process_scale_y
        )
        for name, rectangle
        in display_rois.items()
    }

    print(
        f"Internal processing resolution: "
        f"{process_width} x {process_height}"
    )

    # ---------------------------------------------------------------------
    # Count and output files
    # ---------------------------------------------------------------------

    count_path = Path(
        args.count_file
    )

    if args.reset_count:
        starting_count = 0
        save_count(count_path, 0)
    else:
        starting_count = load_count(
            count_path
        )

    csv_file, csv_writer = open_csv_log(
        args.csv
    )

    # Database logging. A database problem must never stop the counter,
    # so every call is protected and the CSV log keeps working regardless.
    database = None

    if not args.no_db:
        try:
            database = QCDatabase(
                args.db,
                args.station,
                source=str(source)
            )
            database.start_session()

            print(
                f"Database: {args.db} | station: {args.station} "
                f"| session: {database.session_id}"
            )

        except Exception as error:
            print(f"Warning: database disabled: {error}")
            database = None

    last_bundle_video_time = None

    def log_db(event_type, running_count, event_video_time,
               transfer_time=None):
        if database is None:
            return

        try:
            database.log_event(
                event_type,
                running_count,
                event_video_time,
                transfer_time
            )
        except Exception as error:
            print(f"Warning: database write failed: {error}")

    output_writer = None

    if args.output:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            reported_camera_fps,
            (display_width, display_height)
        )

        if not output_writer.isOpened():
            print(
                "Warning: Output video could not be opened."
            )

            output_writer = None

    # ---------------------------------------------------------------------
    # Runtime components
    # ---------------------------------------------------------------------

    calibration_fps = min(
        max(reported_camera_fps, 15),
        60
    )

    calibration_samples = int(
        args.background_calibration_seconds
        * calibration_fps
    )

    panel_parameters = {
        "calibration_samples": calibration_samples,
        "background_distance": (
            args.panel_background_distance
        ),
        "minimum_pixel_ratio": (
            args.panel_pixel_ratio
        ),
        "minimum_span_ratio": (
            args.panel_span_ratio
        ),
        "minimum_dimension_ratio": (
            args.panel_dimension_ratio
        ),
        "minimum_hand_colour_ratio": (
            args.panel_hand_colour_ratio
        ),
        "panel_motion_ratio": (
            args.panel_motion_ratio
        ),
        "background_learning_rate": (
            args.background_learning_rate
        )
    }

    work_panel_detector = PanelPresenceDetector(
        rectangle=process_rois["work"],
        **panel_parameters
    )

    checked_panel_detector = PanelPresenceDetector(
        rectangle=process_rois["checked"],
        **panel_parameters
    )

    frame_motion_detector = FrameMotionDetector(
        args.pixel_difference
    )

    hand_colour_tracker = HandColourTracker()

    hand_movement_tracker = HandMovementTracker(
        process_width,
        process_height
    )

    transfer_counter = TransferCounter(
        starting_count=starting_count,
        source_confirmation=(
            args.source_confirm_seconds
        ),
        checked_confirmation=(
            args.checked_confirm_seconds
        ),
        return_confirmation=(
            args.return_confirm_seconds
        ),
        transfer_timeout=(
            args.transfer_timeout
        ),
        count_cooldown=(
            args.count_cooldown
        ),
    )

    popup = PopupNotification()
    interface = UserInterfaceState()

    # ---------------------------------------------------------------------
    # MediaPipe
    # ---------------------------------------------------------------------

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=(
            mp.tasks.vision.RunningMode.VIDEO
        ),
        num_hands=2,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40
    )

    # Cached hand results allow the application to render at a higher FPS
    # than MediaPipe inference.
    cached_hand_centers = []
    cached_hand_points = []
    cached_hand_rectangles = []

    hand_speed = 0.0
    hand_is_moving = False

    frame_number = 0
    previous_time = None
    previous_timestamp_ms = -1

    program_start = time.perf_counter()
    pending_frame = first_frame

    fps_ema = 0.0
    previous_loop_time = time.perf_counter()

    show_debug = False
    running = True

    if not args.no_display:
        cv2.namedWindow(
            MAIN_WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        cv2.setMouseCallback(
            MAIN_WINDOW_NAME,
            mouse_callback,
            interface
        )

    print("\nApplication started.")
    print("White table mode: no background calibration.")
    print("B = reset motion state")
    print("M = debug masks")
    print("C = reset all counts")
    print("R = reset workflow")
    print("Q = quit\n")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as hand_detector:

            while running:
                loop_start = time.perf_counter()

                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                else:
                    success, frame = capture.read()

                    if not success or frame is None:
                        print(
                            "\nEnd of video (played once). "
                            f"Frames processed: {frame_number}"
                        )
                        break

                    frame = correct_camera_orientation(
                        frame,
                        is_camera,
                        args.unmirror_camera
                    )

                # Keep every frame the same size as the first frame
                if (
                    frame.shape[1] != target_size[0]
                    or frame.shape[0] != target_size[1]
                ):
                    frame = cv2.resize(
                        frame,
                        target_size,
                        interpolation=cv2.INTER_LINEAR
                    )

                if is_camera:
                    video_time = (
                        time.perf_counter()
                        - program_start
                    )
                else:
                    video_time = (
                        frame_number
                        / max(reported_camera_fps, 1)
                    )

                if previous_time is None:
                    delta_time = (
                        1.0
                        / max(reported_camera_fps, 1)
                    )
                else:
                    delta_time = max(
                        video_time - previous_time,
                        1.0 / 240.0
                    )

                previous_time = video_time

                # ---------------------------------------------------------
                # Button commands
                # ---------------------------------------------------------

                if interface.consume_quit_request():
                    running = False
                    break

                if interface.consume_reset_request():
                    transfer_counter.reset_all_counts()

                    save_count(
                        count_path,
                        0
                    )

                    write_event(
                        csv_writer,
                        csv_file,
                        0,
                        "COUNT_RESET",
                        video_time
                    )

                    log_db("COUNT_RESET", 0, video_time)

                    popup.show(
                        "ALL COUNTS RESET",
                        "MAIN: 0 | BUNDLE: 0 | CHECKED: 0",
                        video_time,
                        colour=(0, 70, 210)
                    )

                # ---------------------------------------------------------
                # Resize once for all vision operations
                # ---------------------------------------------------------

                if (
                    process_width != display_width
                    or process_height != display_height
                ):
                    process_frame = cv2.resize(
                        frame,
                        (process_width, process_height),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    process_frame = frame.copy()

                gray_frame = cv2.cvtColor(
                    process_frame,
                    cv2.COLOR_BGR2GRAY
                )

                gray_frame = cv2.GaussianBlur(
                    gray_frame,
                    (5, 5),
                    0
                )

                lab_frame = cv2.cvtColor(
                    process_frame,
                    cv2.COLOR_BGR2LAB
                )

                frame_motion_mask = (
                    frame_motion_detector.update(
                        gray_frame
                    )
                )

                # ---------------------------------------------------------
                # MediaPipe at reduced frequency
                # ---------------------------------------------------------

                run_hand_detection = (
                    frame_number
                    % args.hand_detect_every
                    == 0
                )

                if run_hand_detection:
                    timestamp_ms = int(
                        video_time * 1000
                    )

                    timestamp_ms = max(
                        timestamp_ms,
                        previous_timestamp_ms + 1
                    )

                    previous_timestamp_ms = timestamp_ms

                    media_image = prepare_mediapipe_image(
                        process_frame
                    )

                    hand_result = (
                        hand_detector.detect_for_video(
                            media_image,
                            timestamp_ms
                        )
                    )

                    (
                        cached_hand_centers,
                        cached_hand_points,
                        cached_hand_rectangles
                    ) = extract_hands(
                        hand_result,
                        process_width,
                        process_height
                    )

                    hand_speed = (
                        hand_movement_tracker.update(
                            cached_hand_centers,
                            video_time
                        )
                    )

                    hand_is_moving = (
                        len(cached_hand_centers) >= 1
                        and hand_speed
                        >= args.hand_motion_threshold
                    )

                hand_count = len(
                    cached_hand_centers
                )

                hand_information = (
                    largest_hand_information(
                        cached_hand_rectangles
                    )
                )

                del hand_information

                hand_mask = create_hand_mask(
                    process_frame.shape,
                    cached_hand_points,
                    args.hand_expansion,
                    args.hand_minimum_padding
                )

                # Tight mask for panel detection only (does not erase fabric)
                panel_hand_mask = create_tight_hand_mask(
                    process_frame.shape,
                    cached_hand_points,
                    minimum_padding=2
                )

                hand_reference_lab = (
                    hand_colour_tracker.update(
                        lab_frame,
                        cached_hand_points
                    )
                )

                hand_in_work_area = any_hand_near_area(
                    cached_hand_rectangles,
                    process_rois["work"],
                    args.roi_padding,
                    process_width,
                    process_height
                )

                hand_in_checked_area = any_hand_near_area(
                    cached_hand_rectangles,
                    process_rois["checked"],
                    args.roi_padding,
                    process_width,
                    process_height
                )

                # ---------------------------------------------------------
                # Panel detection
                # ---------------------------------------------------------

                work_panel = (
                    work_panel_detector.update(
                        lab_frame=lab_frame,
                        frame_motion_mask=(
                            frame_motion_mask
                        ),
                        hand_mask=panel_hand_mask,
                        hand_reference_lab=(
                            hand_reference_lab
                        ),
                        hand_near_area=(
                            hand_in_work_area
                        )
                    )
                )

                checked_panel = (
                    checked_panel_detector.update(
                        lab_frame=lab_frame,
                        frame_motion_mask=(
                            frame_motion_mask
                        ),
                        hand_mask=panel_hand_mask,
                        hand_reference_lab=(
                            hand_reference_lab
                        ),
                        hand_near_area=(
                            hand_in_checked_area
                        )
                    )
                )

                calibration_complete = (
                    work_panel_detector.calibrated
                    and checked_panel_detector.calibrated
                )

                # ---------------------------------------------------------
                # Workflow
                # ---------------------------------------------------------

                event = None

                if calibration_complete:
                    event = transfer_counter.update(
                        current_time=video_time,
                        delta_time=delta_time,
                        hand_count=hand_count,
                        hand_is_moving=hand_is_moving,
                        hand_in_work_area=(
                            hand_in_work_area
                        ),
                        hand_in_checked_area=(
                            hand_in_checked_area
                        ),
                        work_panel_detected=(
                            work_panel["detected"]
                        ),
                        work_panel_moving=(
                            work_panel["moving"]
                        ),
                        checked_panel_detected=(
                            checked_panel["detected"]
                        ),
                        checked_panel_moving=(
                            checked_panel["moving"]
                        )
                    )

                if event == "BUNDLE_ACCEPTED":
                    last_bundle_video_time = video_time

                    log_db(
                        "BUNDLE_ACCEPTED",
                        transfer_counter.count,
                        video_time
                    )

                    popup.show(
                        "BUNDLE PANEL ACCEPTED",
                        "BUNDLE STAGE COUNT: 1",
                        video_time,
                        1.0,
                        (0, 160, 190)
                    )

                elif event == "CHECKED_COUNTED":
                    save_count(
                        count_path,
                        transfer_counter.count
                    )

                    write_event(
                        csv_writer,
                        csv_file,
                        transfer_counter.count,
                        "SOURCE_TO_CHECKED",
                        video_time
                    )

                    log_db(
                        "PANEL_CHECKED",
                        transfer_counter.count,
                        video_time,
                        (
                            video_time - last_bundle_video_time
                            if last_bundle_video_time is not None
                            else None
                        )
                    )

                    popup.show(
                        "PANEL COUNTED +1",
                        (
                            "MAIN COUNT: "
                            f"{transfer_counter.count}"
                        ),
                        video_time,
                        args.popup_seconds,
                        (0, 170, 0)
                    )

                    print(
                        "Panel counted. Main count: "
                        f"{transfer_counter.count}"
                    )

                elif event == "RETURNED_TO_BUNDLE":
                    log_db(
                        "RETURNED_TO_BUNDLE",
                        transfer_counter.count,
                        video_time
                    )

                    print(
                        "Worker returned to Area 1. "
                        "Ready for the next panel."
                    )

                elif event == "TIMEOUT":
                    last_bundle_video_time = None

                    log_db(
                        "TIMEOUT",
                        transfer_counter.count,
                        video_time
                    )

                    popup.show(
                        "TRANSFER TIMED OUT",
                        "WORKFLOW STAGES RESET",
                        video_time,
                        1.3,
                        (0, 100, 210)
                    )

                # ---------------------------------------------------------
                # FPS
                # ---------------------------------------------------------

                current_loop_time = time.perf_counter()

                instantaneous_fps = 1.0 / max(
                    current_loop_time
                    - previous_loop_time,
                    0.0001
                )

                previous_loop_time = current_loop_time

                if fps_ema <= 0:
                    fps_ema = instantaneous_fps
                else:
                    fps_ema = (
                        0.90 * fps_ema
                        + 0.10 * instantaneous_fps
                    )

                if database is not None:
                    database.heartbeat()

                # ---------------------------------------------------------
                # Draw
                # ---------------------------------------------------------

                draw_area(
                    frame,
                    display_rois["work"],
                    "BUNDLE + INSPECTION",
                    (0, 255, 255),
                    (
                        hand_in_work_area
                        or work_panel["detected"]
                    ),
                    transfer_counter.bundle_stage_count
                )

                draw_area(
                    frame,
                    display_rois["checked"],
                    "CHECKED PANEL",
                    (0, 255, 0),
                    (
                        hand_in_checked_area
                        or checked_panel["detected"]
                    ),
                    transfer_counter.checked_stage_count
                )

                draw_hands(
                    frame,
                    cached_hand_points,
                    display_scale_x,
                    display_scale_y
                )

                draw_dashboard(
                    frame=frame,
                    transfer_counter=transfer_counter,
                    hand_count=hand_count,
                    hand_speed=hand_speed,
                    hand_is_moving=hand_is_moving,
                    work_panel=work_panel,
                    checked_panel=checked_panel,
                    processing_fps=fps_ema,
                    reported_camera_fps=(
                        reported_camera_fps
                    )
                )

                draw_buttons(
                    frame,
                    interface
                )

                if not calibration_complete:
                    progress = min(
                        work_panel_detector.calibration_progress,
                        checked_panel_detector.calibration_progress
                    )

                    popup.show(
                        "CALIBRATING PANEL BACKGROUNDS",
                        (
                            "CLEAR BOTH AREAS - "
                            f"{int(progress * 100)}%"
                        ),
                        video_time,
                        0.15,
                        (160, 80, 0)
                    )

                popup.draw(
                    frame,
                    video_time
                )

                if output_writer is not None:
                    output_writer.write(frame)

                # ---------------------------------------------------------
                # Display and keyboard
                # ---------------------------------------------------------

                if not args.no_display:
                    cv2.imshow(
                        MAIN_WINDOW_NAME,
                        frame
                    )

                    if show_debug:
                        debug_view = create_debug_view(
                            work_panel,
                            checked_panel,
                            hand_mask
                        )

                        debug_view = cv2.resize(
                            debug_view,
                            (display_width, display_height),
                            interpolation=cv2.INTER_NEAREST
                        )

                        cv2.putText(
                            debug_view,
                            "RED=AREA 1 PANEL | "
                            "GREEN=AREA 2 PANEL | "
                            "BLUE=HAND EXCLUSION",
                            (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        cv2.imshow(
                            DEBUG_WINDOW_NAME,
                            debug_view
                        )

                    # For video files, pace close to real-time so it plays
                    # once at normal speed (not double-speed / not looping).
                    if is_camera:
                        wait_ms = 1
                    else:
                        # Aim for source FPS; at least 1 ms
                        wait_ms = max(
                            1,
                            int(round(1000.0 / max(reported_camera_fps, 1.0)))
                        )
                    key = cv2.waitKey(wait_ms) & 0xFF

                    if key == ord("q"):
                        running = False

                    elif key == ord("c"):
                        transfer_counter.reset_all_counts()

                        save_count(
                            count_path,
                            0
                        )

                        write_event(
                            csv_writer,
                            csv_file,
                            0,
                            "COUNT_RESET",
                            video_time
                        )

                        log_db("COUNT_RESET", 0, video_time)

                        popup.show(
                            "ALL COUNTS RESET",
                            "MAIN: 0 | BUNDLE: 0 | CHECKED: 0",
                            video_time,
                            colour=(0, 70, 210)
                        )

                    elif key == ord("r"):
                        transfer_counter.reset_workflow()
                        frame_motion_detector.reset()

                        popup.show(
                            "WORKFLOW RESET",
                            (
                                "MAIN COUNT REMAINS: "
                                f"{transfer_counter.count}"
                            ),
                            video_time,
                            colour=(0, 100, 210)
                        )

                    elif key == ord("b"):
                        work_panel_detector.reset_background()
                        checked_panel_detector.reset_background()
                        frame_motion_detector.reset()
                        transfer_counter.reset_workflow()

                        popup.show(
                            "BACKGROUND RECALIBRATION",
                            "CLEAR BOTH AREAS",
                            video_time,
                            colour=(160, 80, 0)
                        )

                        print(
                            "Background calibration restarted. "
                            "Clear both areas."
                        )

                    elif key == ord("m"):
                        show_debug = not show_debug

                        if not show_debug:
                            try:
                                cv2.destroyWindow(
                                    DEBUG_WINDOW_NAME
                                )
                            except cv2.error:
                                pass

                frame_number += 1

                del loop_start

    except KeyboardInterrupt:
        print("\nApplication stopped by user.")

    finally:
        save_count(
            count_path,
            transfer_counter.count
        )

        capture.release()

        if output_writer is not None:
            output_writer.release()

        csv_file.close()

        if database is not None:
            database.close()

        cv2.destroyAllWindows()

    print("\nApplication closed successfully.")
    print(
        f"Final main count: "
        f"{transfer_counter.count}"
    )
    print(f"Count file: {count_path}")
    print(f"CSV log: {args.csv}")

    if not args.no_db:
        print(f"Database: {args.db}")


if __name__ == "__main__":
    main()
