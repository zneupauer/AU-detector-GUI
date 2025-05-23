"""
feature_extraction.py
---------------------
Feature extraction utilities for facial Action Unit (AU) detection.
"""
import os
import cv2
import dlib
import numpy as np
from scipy.spatial import ConvexHull
import joblib
import time
from typing import Tuple, Optional, Dict, Any

# --- Path setup ---
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, "shape_predictor_68_face_landmarks.dat")
PROJECTION_DIR = os.path.join(MODULE_DIR, "au_projection_le_spectral")

# --- Dlib model loading with error handling ---
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Dlib shape predictor model not found at {MODEL_PATH}. Please download and place it there.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# --- Landmark preprocessing with IOD normalization (66 points) ---
def preprocess_landmarks_iod(raw_points: np.ndarray) -> np.ndarray:
    """
    Normalize facial landmarks using inter-ocular distance (IOD).
    Args:
        raw_points: (68*2,) array of landmark coordinates.
    Returns:
        Normalized (68*2,) array.
    """
    landmarks = raw_points.reshape(68, 2)
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    iod = np.linalg.norm(right_eye - left_eye)
    if iod < 1e-6:
        iod = 1.0
    center = (left_eye + right_eye) / 2
    normalized = (landmarks - center) / iod
    return normalized.flatten()


def extract_landmark_features(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Extract normalized landmark features from coordinates.
    Args:
        image: Face image (unused, for API consistency).
        coords: (68,2) array of landmark coordinates.
    Returns:
        Normalized (68*2,) array.
    """
    return preprocess_landmarks_iod(coords)


def extract_custom_hog_masked(
    image: np.ndarray,
    coords: np.ndarray,
    cell_size: Tuple[int, int] = (18, 16),
    num_bins: int = 59
) -> np.ndarray:
    """
    Extract HOG features from masked face region using convex hull of landmarks.
    Args:
        image: Face image (BGR).
        coords: (68,2) array of landmark coordinates.
        cell_size: Size of HOG cell.
        num_bins: Number of orientation bins.
    Returns:
        HOG feature vector.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_points, 255)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    gx = cv2.Sobel(masked_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(masked_gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    orientation_range = np.linspace(0, 180, num_bins + 1)
    features = []
    for (x, y) in coords:
        x, y = int(x), int(y)
        x1, x2 = x - cell_size[0] // 2, x + cell_size[0] // 2
        y1, y2 = y - cell_size[1] // 2, y + cell_size[1] // 2
        if x1 < 0 or y1 < 0 or x2 >= masked_gray.shape[1] or y2 >= masked_gray.shape[0]:
            features.extend([0] * num_bins)
            continue
        patch_mag = magnitude[y1:y2, x1:x2]
        patch_angle = angle[y1:y2, x1:x2]
        patch_mask = mask[y1:y2, x1:x2]
        if patch_mask.sum() == 0:
            features.extend([0] * num_bins)
            continue
        patch_mag = patch_mag * (patch_mask > 0)
        hist, _ = np.histogram(patch_angle, bins=orientation_range, weights=patch_mag, density=True)
        features.extend(hist)

    return np.array(features).flatten()


def extract_hog_features(
    image: np.ndarray,
    landmarks: np.ndarray,
    cell_size: Tuple[int, int] = (18, 16),
    num_bins: int = 59
) -> np.ndarray:
    """
    Wrapper for HOG feature extraction.
    Args:
        image: Face image (BGR).
        landmarks: (68,2) array of landmark coordinates.
        cell_size: Size of HOG cell.
        num_bins: Number of orientation bins.
    Returns:
        HOG feature vector.
    """
    return extract_custom_hog_masked(image, landmarks, cell_size, num_bins)


def extract_features_from_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Extract landmark and HOG features from an image file.
    Args:
        image_path: Path to image file.
    Returns:
        Tuple of (landmark features, HOG features, message), or (None, None, message) if failed.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read {image_path}")
        return None, None, "Nepodarilo sa načítať obrázok."
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("❌ No face detected")
        return None, None, "Na obrázku nebola detegovaná žiadna tvár."
    d = faces[0]
    h_img, w_img = gray.shape
    # Check if the face is fully within the image bounds
    if d.left() <= 0 or d.top() <= 0 or d.right() >= w_img or d.bottom() >= h_img:
        print("❌ Whole face not visible")
        return None, None, "Celá tvár musí byť viditeľná na obrázku."
    timestamp = int(time.time() * 1000)
    shape = predictor(gray, d)
    face_chip = dlib.get_face_chip(image, shape, size=256, padding=0.4)
    shape_chip = predictor(face_chip, dlib.rectangle(0, 0, 256, 256))
    coords = np.array([[p.x, p.y] for p in shape_chip.parts()])
    landmarks_feat = extract_landmark_features(face_chip, coords)
    hog_feat = extract_hog_features(face_chip, coords)
    return landmarks_feat, hog_feat, None


def extract_features_from_image_array(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Extract landmark and HOG features from an image array (BGR).
    Args:
        image: Image array (BGR).
    Returns:
        Tuple of (landmark features, HOG features, message), or (None, None, message) if failed.
    """
    if image is None:
        print(f"❌ Failed to read image array")
        return None, None, "Nepodarilo sa načítať obrázok."
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("❌ No face detected")
        return None, None, "Na obrázku nebola detegovaná žiadna tvár."
    d = faces[0]
    h_img, w_img = gray.shape
    if d.left() <= 0 or d.top() <= 0 or d.right() >= w_img or d.bottom() >= h_img:
        print("❌ Whole face not visible")
        return None, None, "Celá tvár musí byť viditeľná na obrázku."
    shape = predictor(gray, d)
    face_chip = dlib.get_face_chip(image, shape, size=256, padding=0.4)
    gray_face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)
    shape_chip = predictor(gray_face_chip, dlib.rectangle(0, 0, 256, 256))
    coords_chip = np.array([[p.x, p.y] for p in shape_chip.parts()])
    landmarks_feat = extract_landmark_features(face_chip, coords_chip)
    hog_feat = extract_hog_features(face_chip, coords_chip)
    return landmarks_feat, hog_feat, None


def get_projected_hog_per_au(
    hog_raw: np.ndarray,
    aus: Any
) -> Dict[str, np.ndarray]:
    """
    For each AU, load the scaler and projection matrix, project HOG features, and return a dict AU->projected HOG.
    Args:
        hog_raw: Raw HOG feature vector.
        aus: Iterable of AU numbers.
    Returns:
        Dict mapping AU name (e.g. 'AU1') to projected HOG feature array.
    """
    hog_feat_per_au = {}
    for au in aus:
        scaler_path = os.path.join(PROJECTION_DIR, f"AU{au}_scaler.pkl")
        W_path = os.path.join(PROJECTION_DIR, f"AU{au}_W.npy")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(f"Scaler for AU{au} not found at {scaler_path}")
        if not os.path.isfile(W_path):
            raise FileNotFoundError(f"Projection matrix for AU{au} not found at {W_path}")
        scaler = joblib.load(scaler_path)
        W = np.load(W_path)
        hog_reshaped = hog_raw.reshape(1, -1)
        hog_scaled = scaler.transform(hog_reshaped)[0]
        hog_proj = hog_scaled @ W.T
        hog_feat_per_au[f"AU{au}"] = hog_proj.flatten()
    return hog_feat_per_au


