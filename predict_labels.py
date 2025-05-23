"""
predict_labels.py
-----------------
Prediction utilities for Action Unit (AU) detection using HOG and landmark features.
"""

import warnings
import numpy as np


def predict_aus(au,landmarks_feat, hog_feat_per_au, models, scalers):
    """
    Predict binary and multiclass presence of a given Action Unit (AU)
        using both HOG and landmark features.
    Args:
        au: Integer AU identifier (e.g., 1, 2, 4, ...)
        landmarks_feat: 1D array of normalized facial landmarks
        hog_feat_per_au: Dict of AU keys -> projected HOG features
        models: Dict of model dictionaries for each method type
        scalers: Dict of scaler dictionaries for each method type
    Returns:
        Tuple (landmark_binary, hog_binary, landmark_multiclass, hog_multiclass)
    """

    def predict_output(features, model, scaler):
        """
        Helper function to scale features and make a prediction.
        Suppresses feature name warnings when using NumPy input.
        """
        if model and scaler:
            features = np.array(features).flatten()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                scaled = scaler.transform([features])
            return model.predict(scaled)[0]
        return "?"


    au_key = f"AU{au}"

    # HOG binary
    model = models["results_hog_binary"].get(au_key)
    scaler = scalers["results_hog_binary"].get(au_key)
    hog_bin_pred = predict_output(hog_feat_per_au[au_key], model, scaler)

    # Landmark binary
    model = models["results_landmark_binary"].get(au_key)
    scaler = scalers["results_landmark_binary"].get(au_key)
    lm_bin_pred = predict_output(landmarks_feat, model, scaler)

    # HOG multiclass
    model = models["results_hog_multiclass"].get(au_key)
    scaler = scalers["results_hog_multiclass"].get(au_key)
    hog_mc_pred = predict_output(hog_feat_per_au[au_key], model, scaler)

    # Landmark multiclass
    model = models["results_landmark_multiclass"].get(au_key)
    scaler = scalers["results_landmark_multiclass"].get(au_key)
    lm_mc_pred = predict_output(landmarks_feat, model, scaler)

    return lm_bin_pred, hog_bin_pred, lm_mc_pred, hog_mc_pred
