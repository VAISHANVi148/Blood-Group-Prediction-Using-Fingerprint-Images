# -*- coding: utf-8 -*-
"""
Blood Group Detection System using CNN and Fingerprint Image Validation
Author: Vaishnavi (Final Version with All Image Format Support)
"""

import tkinter as tk
from tkinter import ttk, LEFT, END, messagebox
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import sqlite3
import os
from keras.models import load_model
from datetime import datetime
from skimage.feature import local_binary_pattern

# ========== Load Models Safely ==========
def safe_load_model(model_path):
    if not os.path.exists(model_path):
        messagebox.showerror("Model Missing", f"Model file not found:\n{model_path}")
        raise FileNotFoundError(f"{model_path} not found.")
    return load_model(model_path)

try:
    fingerprint_model = safe_load_model('fingerprint_identifier_model.h5')
    blood_model = safe_load_model('Group.h5')
except FileNotFoundError:
    exit()

# ========== Globals ==========
fn = ""
img_label = gray_label = binary_label = result_label = None

# ========== GUI Setup ==========
root = tk.Tk()
root.configure(background="skyblue")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{w}x{h}+0+0")
root.title("Blood Group Detection System")

# Background
try:
    bg_img = Image.open("back4.jpg").resize((2000, 1000))
    bg_img_tk = ImageTk.PhotoImage(bg_img)
    bg_label = tk.Label(root, image=bg_img_tk)
    bg_label.place(x=0, y=0)
except Exception as e:
    print("Background image error:", e)

# Title
tk.Label(root, text="Blood Group Detection System", font=('times', 30, 'bold'),
         width=70, height=1, bg="#152238", fg="white").place(x=0, y=0)

# ========== Store Prediction ==========
def store_prediction(image_path, prediction, confidence):
    with sqlite3.connect("prediction_history.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                predicted_group TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("INSERT INTO predictions (image_path, predicted_group, confidence, timestamp) VALUES (?, ?, ?, ?)",
                       (image_path, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ========== View History ==========
def show_history():
    history_window = tk.Toplevel(root)
    history_window.title("Prediction History")
    history_window.geometry("800x400")
    tree = ttk.Treeview(history_window, columns=("Image Path", "Prediction", "Confidence", "Timestamp"), show='headings')
    for col in tree["columns"]:
        tree.heading(col, text=col)
    tree.pack(fill=tk.BOTH, expand=True)

    with sqlite3.connect("prediction_history.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, predicted_group, confidence, timestamp FROM predictions")
        for row in cursor.fetchall():
            tree.insert('', END, values=row)

# ========== Update Result Label ==========
def update_label(msg):
    global result_label
    if result_label:
        result_label.destroy()
    result_label = tk.Label(root, text=msg, width=70, font=("bold", 18), bg='white', fg='darkgreen', justify=LEFT)
    result_label.place(x=300, y=650)

# ========== Prediction Logic ==========
def test_model_proc(filepath):
    IMAGE_SIZE = 64
    img = Image.open(filepath).resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype('float32') / 255.0
    prediction = blood_model.predict(img)
    confidence = float(np.max(prediction)) * 100
    label_idx = np.argmax(prediction)
    blood_groups = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]
    return blood_groups[label_idx], confidence

# ========== Fingerprint Validator ==========

from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.measure import shannon_entropy

def is_valid_fingerprint(image_path, edge_thresh=0.015, lbp_thresh=0.3, entropy_range=(2.5, 8.0)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Basic sanity check
    if img is None or img.shape[0] < 64 or img.shape[1] < 64:
        return False

    # Resize and CLAHE
    img = cv2.resize(img, (64, 64))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Multiple Gabor Filters to simulate ridge orientation
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_sum = np.zeros_like(img_clahe, dtype=np.float32)
    for theta in orientations:
        g_kernel = cv2.getGaborKernel((15, 15), 3, theta, 8, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img_clahe, cv2.CV_32F, g_kernel)
        gabor_sum += filtered

    gabor_sum = np.uint8(np.clip(gabor_sum, 0, 255))
    edges = cv2.Canny(gabor_sum, 50, 150)

    # Edge density check
    edge_density = np.sum(edges > 0) / (64 * 64)

    # LBP texture analysis
    lbp = local_binary_pattern(img_clahe, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
    lbp_score = np.sum(lbp_hist[:10]) / np.sum(lbp_hist)

    # Entropy check (to remove very noisy or too smooth images)
    entropy = shannon_entropy(img_clahe)

    # Final decision logic
    if (edge_density > edge_thresh and 
        lbp_score > lbp_thresh and 
        entropy_range[0] < entropy < entropy_range[1]):
        return True
    else:
        return False


# ========== Predict Button ==========
def test_model():
    global fn
    if not fn:
        update_label("⚠️ Please Select Image For Prediction.")
        return

    if not is_valid_fingerprint(fn):
        update_label("❌ Invalid image. Select a fingerprint.")
        messagebox.showerror("Invalid", "Choose a valid fingerprint image.")
        return

    update_label("⏳ Testing in Progress...")

    try:
        for btn in root.winfo_children():
            if isinstance(btn, tk.Button):
                btn.config(state='disabled')

        root.update()
        start = time.time()
        prediction, confidence = test_model_proc(fn)
        store_prediction(fn, prediction, confidence)
        end = time.time()

        msg = f"✅ Predicted: {prediction} | Confidence: {confidence:.2f}%\n⏱ Time: {end - start:.2f} sec"
        update_label(msg)
        messagebox.showinfo("Success", "Prediction Done!.")

    except Exception as e:
        update_label("❌ Error during prediction.")
        messagebox.showerror("Error", str(e))

    finally:
        for btn in root.winfo_children():
            if isinstance(btn, tk.Button):
                btn.config(state='normal')
        fn = ""

# ========== Load Image ==========
def openimage():
    global fn, img_label
    filename = askopenfilename(
        title='Select fingerprint image',
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
    )

    if not filename:
        messagebox.showinfo("Cancelled", "No image selected.")
        return

    fn = os.path.abspath(filename)
    try:
        img = Image.open(fn).resize((250, 250))
        imgtk = ImageTk.PhotoImage(img)

        if img_label:
            img_label.destroy()
        img_label = tk.Label(root, text='Original', font=('times new roman', 20, 'bold'),
                             image=imgtk, compound='bottom')
        img_label.image = imgtk
        img_label.place(x=400, y=100)

        globals()['img_label'] = img_label
        update_label("✅ Image loaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ========== Preprocess Image ==========
def convert_grey():
    global fn
    if not fn:
        messagebox.showerror("Error", "Select an image first.")
        return

    IMAGE_SIZE = 200
    img_cv = cv2.imread(fn)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def update_img_label(image_array, label_ref_name, label_text, x):
        global gray_label, binary_label
        img = Image.fromarray(image_array)
        imgtk = ImageTk.PhotoImage(img)
        lbl = tk.Label(root, text=label_text, font=('times new roman', 20, 'bold'),
                       image=imgtk, compound='bottom', height=250, width=250, bg='white')
        lbl.image = imgtk
        lbl.place(x=x, y=100)

        if label_ref_name == 'gray':
            if gray_label:
                gray_label.destroy()
            gray_label = lbl
        elif label_ref_name == 'binary':
            if binary_label:
                binary_label.destroy()
            binary_label = lbl

    update_img_label(gray, 'gray', 'Gray', 690)
    update_img_label(binary, 'binary', 'Binary', 1000)
     
# ========== Clear ==========
def clear_screen():
    global img_label, result_label, gray_label, binary_label, fn
    for widget in [img_label, result_label, gray_label, binary_label]:
        if widget:
            widget.destroy()
    img_label = result_label = gray_label = binary_label = None
    fn = ""

# ========== Exit ==========
def exit_app():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

# ========== Buttons ==========
button_actions = [
    ("Select Image", openimage, 100),
    ("Preprocess Image", convert_grey, 170),
    ("Predict Blood Group", test_model, 240),
    ("View History", show_history, 310),
    ("Clear Screen", clear_screen, 380),
    ("Exit", exit_app, 450)
]

for text, command, y in button_actions:
    tk.Button(root, text=text, command=command, width=20, height=1,
              font=('times', 15, 'bold'), bg="white", fg="black").place(x=50, y=y)

root.mainloop()






