"""
main.py
-------
Main GUI application for AU (Action Unit) detection using facial images 
"""
import os
import platform
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import feature_extraction
import predict_labels
from model_loader import all_models, all_scalers
import atexit

# List of Action Units and their descriptions
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
AU_DESCRIPTIONS = {
    1: "Zdvíhanie vnútornej časti obočia",
    2: "Zdvíhanie vonkajšej časti obočia",
    4: "Stiahnutie obočia nadol",
    5: "Zdvíhanie horného viečka",
    6: "Zdvíhanie líc",
    9: "Zmŕštenie nosa",
    12: "Ťahanie kútikov pier nahor",
    15: "Zníženie kútikov pier",
    17: "Zdvíhanie brady",
    20: "Roztiahnutie pier",
    25: "Oddelenie pier",
    26: "Spadnutie sánky"
}

def center_window(window):
    """Center the given Tkinter window on the screen."""
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"+{x}+{y}")

# Delete all captured_face.jpg files on exit (cleanup for temp files)
def cleanup_captured_photos():
    """Delete the temporary captured_face.jpg file on exit."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    photo_path = os.path.join(script_dir, "captured_face.jpg")
    if os.path.exists(photo_path):
        try:
            os.remove(photo_path)
        except Exception:
            pass
atexit.register(cleanup_captured_photos)

class AUDetectorGUI:
    """
    Main GUI class for AU detection.
    Handles all UI logic, camera/photo handling, and result display.
    """
    def __init__(self, root):
        """Initialize the GUI, set up main menu, and window close protocol."""
        self.root = root
        self.root.title("AU Detektor")
        self.image_path = None
        self.cap = None
        self.camera_running = False
        self.models = all_models
        self.scalers = all_scalers
        self.current_frame = None
        self.setup_main_menu()
        # Ensure cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_main_menu(self):
        """Set up the main menu layout with buttons and image/results areas."""
        self.clear_root()
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill="both")
        # Two columns: left for image, right for results
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.main_frame.columnconfigure(0, weight=0, minsize=620)  # 600 for image + padding
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        # Center button block horizontally in left_frame
        self.left_frame.columnconfigure(0, weight=1)
        button_block = tk.Frame(self.left_frame)
        button_block.grid(row=0, column=0, pady=(0, 10), sticky="")
        tk.Button(button_block, text="📁 Nahrať obrázok", command=self.upload_image, width=20).pack(pady=3)
        tk.Button(button_block, text="📷 Spustiť kameru", command=self.start_camera, width=20).pack(pady=3)
        self.capture_button = tk.Button(button_block, text="📸 Odfotiť", command=self.capture_photo, width=20, state=tk.DISABLED)
        self.capture_button.pack(pady=3)
        tk.Button(button_block, text="❌ Ukončiť", command=self.root.quit, width=20).pack(pady=3)
        # Image label (left)
        self.image_label = tk.Label(self.left_frame, bd=0, highlightthickness=0)
        self.image_label.grid(row=1, column=0, pady=5, sticky="n")
        # Results frame (right)
        self.results_frame = tk.Frame(self.right_frame)
        self.results_frame.pack(expand=True, fill="both")
        # Message on right by default
        tk.Label(
            self.results_frame,
            text="Výsledky AU sa zobrazia po nahraní alebo odfotení obrázka.",
            font=("Arial", 12)
        ).pack(expand=True, fill="both", anchor="center")

    def clear_root(self):
        """Remove all widgets from the root window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def get_face_rect(self, cv_img):
        """Detect the first face in the image and return its bounding box (x, y, w, h)."""
        import dlib
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            d = faces[0]
            return (d.left(), d.top(), d.width(), d.height())
        return None

    def upload_image(self):
        """Handle image upload, show image, and process AU detection."""
        if self.camera_running:
            self.stop_camera()
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = os.path.abspath(file_path)
            image = cv2.imread(self.image_path)
            if image is not None:
                face_rect = self.get_face_rect(image)
                self.show_image(image, face_rect=face_rect, draw_face_rect=True)
                self.process_image()
                
            else:
                self.clear_results()
                tk.Label(self.results_frame, text="Nepodarilo sa načítať obrázok.", fg="red").pack()

    def start_camera(self):
        """Start the camera preview and enable the capture button."""
        self.camera_running = True
        self.clear_results()
        self.capture_button.config(state=tk.NORMAL)  # Enable the capture button
        self.show_camera_frame()
     
     

    def stop_camera(self):
        """Stop the camera preview and disable the capture button."""
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capture_button.config(state=tk.DISABLED)  # Disable the capture button

    def show_camera_frame(self):
        """Continuously update the camera preview in the GUI."""
        if not self.camera_running:
            return
        if self.cap is None:
            if platform.system() == "Darwin":
                self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            elif platform.system() == "Windows":
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Do NOT detect face or draw rectangle during live preview
            self.show_image(frame)
        self.root.after(10, self.show_camera_frame)

    def capture_photo(self):
        """Capture the current camera frame, save, and process AU detection."""
        if not self.camera_running:
            return
        if self.current_frame is not None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.image_path = os.path.join(script_dir, "captured_face.jpg")
            cv2.imwrite(self.image_path, self.current_frame)
            self.stop_camera()
            face_rect = self.get_face_rect(self.current_frame)
            self.show_image(self.current_frame, face_rect=face_rect, draw_face_rect=True)
            self.process_image()

    def show_image(self, cv_img, face_rect=None, draw_face_rect=False):
        """Display an image in the left panel, optionally drawing a rectangle around the detected face."""
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # Draw rectangle if face_rect is provided and draw_face_rect is True
        if face_rect is not None and draw_face_rect:
            x, y, w, h = face_rect
            import cv2 as _cv2
            cv_img_rgb = _cv2.rectangle(cv_img_rgb.copy(), (x, y), (x + w, y + h), (0, 255, 0), 3)
        base_size = 600
        img = Image.fromarray(cv_img_rgb)
        w, h = img.size
        # Resize while keeping aspect ratio
        if w > h:
            new_w = base_size
            new_h = int(h * base_size / w)
        else:
            new_h = base_size
            new_w = int(w * base_size / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Get the background color from the Tkinter widget
        bg_color = self.left_frame.cget("bg")  # or self.root.cget("bg")
        # Convert color name to RGB
        self.root.update_idletasks()  # Ensure color is resolved
        rgb = self.root.winfo_rgb(bg_color)
        rgb = tuple([int(v/256) for v in rgb])  # Convert 16-bit to 8-bit per channel
        background = Image.new("RGB", (base_size, base_size), rgb)
        offset = ((base_size - new_w) // 2, (base_size - new_h) // 2)
        background.paste(img, offset)
        imgtk = ImageTk.PhotoImage(image=background)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk, width=base_size, height=base_size)
        

    def process_image(self):
        """Extract features, predict AUs, and display results in the right panel."""
        self.clear_results()
        result = feature_extraction.extract_features_from_image(self.image_path)
        if len(result) == 3:
            landmarks_feat, hog_feat_raw, message = result
        else:
            landmarks_feat, hog_feat_raw = result
            message = None
        if message:
            tk.Label(self.results_frame, text=message, fg="red").pack()
            return
        if landmarks_feat is None or hog_feat_raw is None:
            tk.Label(self.results_frame, text="No face or features detected.", fg="red").pack()
            return
        hog_feat_per_au = feature_extraction.get_projected_hog_per_au(hog_feat_raw, AUS)
        # Vertically center the AU results table in the right panel
        container = tk.Frame(self.results_frame)
        container.pack(expand=True, fill="both")
        table = tk.Frame(container)
        table.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(table, text="Predikcia AU", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=5, pady=5)
        columns = ["AU", "HOG BINARY", "LANDMARKS BINARY", "HOG MULTICLASS", "LANDMARKS MULTICLASS"]
        for col_idx, col_name in enumerate(columns):
            tk.Label(table, text=col_name, font=("Arial", 10, "bold")).grid(row=1, column=col_idx, padx=5, pady=3)
        for idx, au_num in enumerate(AUS):
            au = f"AU{au_num}"
            lm_bin_pred, hog_bin_pred, lm_mc_pred, hog_mc_pred = predict_labels.predict_aus(au_num, landmarks_feat, hog_feat_per_au, all_models, all_scalers)
            hog_mc_val = int(hog_mc_pred) if str(hog_mc_pred).isdigit() else 0
            lm_mc_val = int(lm_mc_pred) if str(lm_mc_pred).isdigit() else 0
            tk.Label(
                table,
                text=f"{au} - {AU_DESCRIPTIONS.get(au_num, '')}",
                font=("Arial", 10, "bold"),
                anchor="w",
                width=28  # fixed width for AU description
            ).grid(row=idx+2, column=0, padx=5, pady=3, sticky="w")
            self._render_binary(table, hog_bin_pred, idx+2, 1)
            self._render_binary(table, lm_bin_pred, idx+2, 2)
            self._render_bar(table, hog_mc_val, idx+2, 3)
            self._render_bar(table, lm_mc_val, idx+2, 4)

    def clear_results(self):
        """Remove all widgets from the results frame (right panel)."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def _render_binary(self, frame, pred, row, col):
        """Render a binary prediction (check/cross) in the AU results table."""
        if pred == 1:
            tk.Label(frame, text="✔", bg="#4CAF50", fg="white", width=12, font=("Arial", 12, "bold")).grid(row=row, column=col, padx=5, pady=3)
        elif pred == 0:
            tk.Label(frame, text="✗", bg="#F44336", fg="white", width=12, font=("Arial", 12, "bold")).grid(row=row, column=col, padx=5, pady=3)
        else:
            tk.Label(frame, text=pred, bg="white", fg="#003366", width=12).grid(row=row, column=col, padx=5, pady=3)

    def _render_bar(self, frame, value, row, col):
        """Render a progress bar for multiclass AU prediction in the AU results table."""
        bar = ttk.Progressbar(frame, length=80, maximum=5)
        bar.grid(row=row, column=col, padx=5, pady=8, sticky='w')
        bar['value'] = value
        tk.Label(frame, text=f"{value}/5").grid(row=row, column=col, sticky='e')

    def on_close(self):
        """Cleanup and close the application window."""
        # Try to delete the photo before closing
        script_dir = os.path.dirname(os.path.abspath(__file__))
        photo_path = os.path.join(script_dir, "captured_face.jpg")
        if os.path.exists(photo_path):
            try:
                os.remove(photo_path)
            except Exception:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # Set initial window size to 80% of the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    root.geometry(f"{window_width}x{window_height}")
    # Set a minimum window size to prevent shrinking after showing results
    min_width = 1400
    min_height = 700
    root.minsize(min_width, min_height)
    app = AUDetectorGUI(root)
    center_window(root)
    root.mainloop()