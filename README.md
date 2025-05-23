# AU Detector GUI

This project is a simple visualization tool created as part of a bachelor thesis focused on automatic facial expression analysis. It provides a graphical user interface (GUI) for visualizing the predictions of pre-trained models that detect facial Action Units (AUs). The tool uses libraries such as dlib, OpenCV, and scikit-learn to extract facial features from a static image and display the outputs of trained classifiers in an intuitive format.

## Requirements

- Python 3.8+
- The following Python libraries:
  - numpy
  - opencv-python
  - dlib
  - Pillow
  - scikit-learn
  - scipy
  - joblib
  - tkinter (usually included with Python)

## Installation

1. **Clone the repository and navigate to the project directory.**

2. **Install dependencies:**

   You can use the following command to install all required libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Or install them individually:

   ```bash
   pip install numpy opencv-python dlib Pillow scikit-learn scipy joblib
   ```

   > **Note:**
   > - On macOS, you may need to install CMake and Boost for dlib:
   >   ```bash
   >   brew install cmake boost
   >   ```
   > - On Windows, use pre-built wheels for dlib if you encounter issues.

3. **Download the dlib shape predictor model:**

   Download `shape_predictor_68_face_landmarks.dat` from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place it in the same directory as the Python scripts.

4. **Place the model folders:**

   Ensure the following folders (with trained models and scalers) are in the same directory as the scripts:
   - `results_landmark_binary`
   - `results_hog_binary`
   - `results_landmark_multiclass`
   - `results_hog_multiclass`
   - `au_projection_le_spectral` 

## Usage

You can run the main GUI application in several ways:

### 1. From the Terminal

```bash
python main.py
```

### 2. Using the “Play” Button in Your IDE

- **VS Code:**  
  - Open `main.py` in the editor.
  - Click the green “Run” (play) button in the top right, or right-click in the editor and select “Run Python File in Terminal”.

- **PyCharm:**  
  - Open `main.py` in the editor.
  - Click the green play button next to the line `if __name__ == "__main__":` or at the top right of the window.
  - Make sure your interpreter is set to your virtual environment if you created one.

- **Other IDEs:**  
  - Open `main.py` and look for a “Run” or “Play” button, or right-click and select “Run”.

---

Let me know if you want this added directly to your `README.md`!


## Notes
- The application should work on Windows, macOS, and Linux, provided all dependencies are installed.
- For any issues with dlib installation, refer to the [dlib installation guide](http://dlib.net/compile.html).
- If you use an IDE, make sure your interpreter is set to the virtual environment you created.

## Troubleshooting
- **dlib install errors:**  
  - On macOS, install CMake and Boost with Homebrew.
  - On Windows, use a pre-built wheel if you have issues compiling.
- **Missing model files:**  
  - Ensure all required folders and the shape predictor `.dat` file are present in your project directory.

