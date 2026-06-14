# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 02:47:23 2025
@author: vaish
GUI_Master_old1 updated with accurate fingerprint validation
"""

import tkinter as tk
from tkinter import ttk, LEFT, END, messagebox
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import sqlite3
from keras.models import load_model
from datetime import datetime
import os

# Load the trained CNN model for fingerprint detection
fp_identifier_model = load_model('fingerprint_identifier_model.h5')  # Binary classifier: 1=fingerprint, 0=non-fingerprint

# Function to validate if image is a fingerprint using CNN
from keras.preprocessing.image import img_to_array
from PIL import Image as PILImage

def is_valid_fingerprint(image_path):
    try:
        img = PILImage.open(image_path).convert('RGB').resize((64, 64))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = fp_identifier_model.predict(img)
        return prediction[0][0] > 0.5  # 1 = fingerprint, 0 = non-fingerprint
    except Exception as e:
        print(f"Validation error: {e}")
        return False

filename = None

class BloodGroupDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.configure(background="skyblue")
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.root.title("Blood Group Detection System")

        image_path = os.path.join(os.path.dirname(__file__), 'bb.jpg')
        if os.path.exists(image_path):
            image2 = Image.open(image_path).resize((1800, 1000))
            self.background_image = ImageTk.PhotoImage(image2)
            self.background_label = tk.Label(root, image=self.background_image)
            self.background_label.image = self.background_image
            self.background_label.place(x=0, y=0)
        else:
            self.background_label = tk.Label(root, bg="skyblue")
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
            print("❌ Warning: Background image 'bb.jpg' not found.")

        self.lbl = tk.Label(root, text="Blood Group Detection System", 
                          font=('times', 30, 'bold'), width=60, height=1, 
                          bg="#152238", fg="white")
        self.lbl.place(x=0, y=0)

        self.fn = ""
        self.img_label = None
        self.gray_label = None
        self.binary_label = None
        self.result_label = None

        self.init_db()
        self.create_ui()

    # rest of the class stays the same ...


    def init_db(self):
        conn = sqlite3.connect("prediction_history.db")
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
        conn.commit()
        conn.close()

    def store_prediction(self, image_path, prediction, confidence):
        conn = sqlite3.connect("prediction_history.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (image_path, predicted_group, confidence, timestamp) 
            VALUES (?, ?, ?, ?)
        """, (image_path, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

    def test_model_proc(self, filepath):
        IMAGE_SIZE = 64
        model = load_model('Group.h5')
        img = Image.open(filepath).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype('float32') / 255.0
        prediction = model.predict(img)
        confidence = float(np.max(prediction)) * 100
        label_idx = np.argmax(prediction)
        blood_groups = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]
        return blood_groups[label_idx], confidence

    def update_label(self, msg):
        if self.result_label:
            self.result_label.destroy()
        self.result_label = tk.Label(self.root, text=msg, width=80, 
                                    font=("bold", 18), bg='white', 
                                    fg='darkgreen', justify=LEFT)
        self.result_label.place(x=300, y=650)

    def test_model(self):
        if not self.fn:
            self.update_label("Please Select Image For Prediction....")
            return

        if not is_valid_fingerprint(self.fn):
            self.update_label("❌ Rejected: Not a valid fingerprint image.")
            messagebox.showerror("Invalid Image", "Selected image is not a valid fingerprint.")
            return

        self.update_label("Model Testing in Progress...")
        self.disable_buttons()

        try:
            start = time.time()
            prediction, confidence = self.test_model_proc(self.fn)
            self.store_prediction(self.fn, prediction, confidence)
            end = time.time()
            msg = f"✅ Predicted: {prediction} | Confidence: {confidence:.2f}%\n⏱ Execution Time: {end - start:.2f} seconds"
            self.update_label(msg)
            messagebox.showinfo("Success", "Prediction stored successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_label("❌ Error during prediction.")
        finally:
            self.enable_buttons()

    def openimage(self):
        filename = askopenfilename(
            title='Select image for Analysis', 
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not filename:
            return

        if not is_valid_fingerprint(filename):
            messagebox.showerror("Invalid Image", "The selected file is not a valid fingerprint image.")
            return

        self.fn = filename
        self.display_image(filename)

    def display_image(self, filename):
        img = Image.open(filename).resize((250, 250))
        imgtk = ImageTk.PhotoImage(img)

        if self.img_label:
            self.img_label.destroy()

        self.img_label = tk.Label(
            self.root, text='Original', 
            font=('times new roman', 20, 'bold'), 
            image=imgtk, compound='bottom'
        )
        self.img_label.image = imgtk
        self.img_label.place(x=300, y=100)

    def convert_grey(self):
        if not self.fn:
            messagebox.showerror("Error", "Please select an image first.")
            return

        IMAGE_SIZE = 200
        img_cv = cv2.imread(self.fn)
        if img_cv is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self.display_processed_images(gray, binary)

    def display_processed_images(self, gray_img, binary_img):
        for label in [self.gray_label, self.binary_label]:
            if label:
                label.destroy()

        im_gray = Image.fromarray(gray_img)
        imgtk_gray = ImageTk.PhotoImage(im_gray)
        self.gray_label = tk.Label(
            self.root, text='Gray', 
            font=('times new roman', 20, 'bold'), 
            image=imgtk_gray, compound='bottom', 
            height=250, width=250, bg='white'
        )
        self.gray_label.image = imgtk_gray
        self.gray_label.place(x=580, y=100)

        im_binary = Image.fromarray(binary_img)
        imgtk_binary = ImageTk.PhotoImage(im_binary)
        self.binary_label = tk.Label(
            self.root, text='Binary', 
            font=('times new roman', 20, 'bold'), 
            image=imgtk_binary, compound='bottom', 
            height=250, width=250, bg='white'
        )
        self.binary_label.image = imgtk_binary
        self.binary_label.place(x=880, y=100)

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Prediction History")
        history_window.geometry("800x600")

        frame = tk.Frame(history_window)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=("Image Path", "Prediction", "Confidence", "Timestamp"), show='headings')
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.heading("Image Path", text="Image Path")
        tree.heading("Prediction", text="Prediction")
        tree.heading("Confidence", text="Confidence")
        tree.heading("Timestamp", text="Timestamp")

        tree.column("Image Path", width=200)
        tree.column("Prediction", width=100)
        tree.column("Confidence", width=100)
        tree.column("Timestamp", width=150)

        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        conn = sqlite3.connect("prediction_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, predicted_group, confidence, timestamp FROM predictions ORDER BY timestamp DESC")

        for row in cursor.fetchall():
            short_path = row[0] if len(row[0]) < 50 else f"...{row[0][-47:]}"
            tree.insert('', END, values=(short_path, row[1], f"{row[2]:.2f}%", row[3]))

        conn.close()

    def clear_screen(self):
        for label in [self.img_label, self.result_label, self.gray_label, self.binary_label]:
            if label:
                label.destroy()
        self.img_label = self.result_label = self.gray_label = self.binary_label = None
        self.fn = ""

    def exit_app(self):
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.destroy()

    def disable_buttons(self):
        for btn in self.root.winfo_children():
            if isinstance(btn, tk.Button):
                btn.config(state='disabled')
        self.root.update()

    def enable_buttons(self):
        for btn in self.root.winfo_children():
            if isinstance(btn, tk.Button):
                btn.config(state='normal')
        self.root.update()

    def create_ui(self):
        buttons = [
            ("Select Image", self.openimage, 100),
            ("Preprocess Image", self.convert_grey, 170),
            ("Predict Blood Group", self.test_model, 240),
            ("View History", self.show_history, 310),
            ("Clear Screen", self.clear_screen, 380),
            ("Exit", self.exit_app, 450)
        ]

        for text, cmd, y in buttons:
            tk.Button(
                self.root, text=text, command=cmd, 
                width=20, height=1, 
                font=('times', 15, 'bold'), 
                bg="white", fg="black"
            ).place(x=50, y=y)

if __name__ == "__main__":
    root = tk.Tk()
    app = BloodGroupDetectionApp(root)
    root.mainloop()
