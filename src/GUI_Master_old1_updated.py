
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 02:47:23 2025

@author: vaish

Enhanced GUI_Master_old1 with Rejection Log and Detailed Feedback
"""

# Required Libraries
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

# Global variable for filename
filename = None

def is_valid_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return False

    img = cv2.resize(img, (300, 300))

    # Edge Detection
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # Gabor Filtering
    gabor = cv2.getGaborKernel((21, 21), 4.0, np.pi / 2, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(img, cv2.CV_8UC3, gabor)
    gabor_response = np.mean(filtered)

    # Ridge Orientation Histogram
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(sobely, sobelx)
    orientation_std = np.std(orientation)

    # Log rejected samples for debugging
    if not (edge_density > 0.012 and gabor_response > 18 and orientation_std < 2.0):
        with open("rejection_log.txt", "a") as f:
            f.write(f"{os.path.basename(image_path)} → Rejected | EdgeDensity: {edge_density:.4f}, Gabor: {gabor_response:.2f}, OrientationSTD: {orientation_std:.2f}\n")

    return edge_density > 0.012 and gabor_response > 18 and orientation_std < 2.0

# All other GUI code and class remains same except update the error message below

# Inside any place where is_valid_fingerprint is checked and failed, use this line:
# Updated Error Message Example
# messagebox.showerror("Invalid Image", "The image does not meet fingerprint characteristics (e.g., low ridge texture or poor clarity). Try a clearer fingerprint.")

# The rest of your GUI and class code goes here (unchanged except for this update)

