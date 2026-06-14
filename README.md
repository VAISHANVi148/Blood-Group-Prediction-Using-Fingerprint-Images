#  Blood Group Detection Using Fingerprint Images

## 📌 Project Overview

Blood Group Detection Using Fingerprint Images is a Deep Learning and Image Processing based application developed to predict a person's blood group from fingerprint images.

The system uses Convolutional Neural Networks (CNN) for blood group classification and includes a separate Fingerprint Validation Model that verifies whether the uploaded image is a valid fingerprint before prediction.

The application provides a user-friendly Tkinter GUI with registration, login, image preprocessing, blood group prediction, confidence score display, and prediction history management using SQLite.

---

#  Key Features

✅ User Registration and Login System

✅ CNN-Based Blood Group Prediction

✅ CNN-Based Fingerprint Validation

✅ Automatic Rejection of Non-Fingerprint Images

✅ Fingerprint Image Preprocessing

✅ Grayscale and Binary Image Conversion

✅ Blood Group Prediction with Confidence Score

✅ Prediction History Storage using SQLite Database

✅ User-Friendly Tkinter GUI

✅ Support for PNG, JPG, JPEG, and BMP Images

---

# Technologies Used

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)

### GUI Development
- Tkinter

### Image Processing
- OpenCV
- Pillow (PIL)

### Database
- SQLite

### Data Analysis & Visualization
- NumPy
- Matplotlib
- Scikit-Learn

---

# 📂 Project Structure

```
Blood-Group-Detection-System
│
├── GUI_Main.py
├── GUI_Master_old.py
├── fingerprint_validator.py
├── Login.py
├── Registration.py
│
├── CNN_Model1.py
├── CNN_Model2.py
├── Model_CNN.py
│
├── README.md
│
├── images
│   ├── login_page.png
│   ├── main_gui.png
│   ├── preprocessing.png
│   └── prediction_result.png
│
└── database
    └── prediction_history.db
```

---

#  System Workflow

### Step 1
User logs into the system.

### Step 2
User uploads a fingerprint image.

### Step 3
Fingerprint Validation Model checks whether the uploaded image is a valid fingerprint.

### Step 4
If the image is valid, preprocessing is performed.

### Step 5
The Blood Group CNN Model predicts the blood group.

### Step 6
The predicted blood group and confidence score are displayed.

### Step 7
Prediction details are stored in the SQLite database.

---

#  Supported Blood Groups

The system can predict the following blood groups:

- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

---

#  CNN Architecture

### Blood Group Classification Model

Input Size:

```
64 × 64 × 3
```

Architecture:

```
Conv2D (32 Filters)
↓
MaxPooling2D
↓
Conv2D (64 Filters)
↓
MaxPooling2D
↓
Flatten
↓
Dense (128 Units)
↓
Dropout (0.5)
↓
Dense (8 Classes)
↓
Softmax
```

---

#  Fingerprint Validation Module

Before blood group prediction, the uploaded image is validated using a dedicated CNN-based Fingerprint Identifier Model.

### Validation Results

| Image Type | Result |
|------------|---------|
| Fingerprint | Accepted |
| Face Image | Rejected |
| Animal Image | Rejected |
| Vehicle Image | Rejected |
| Landscape Image | Rejected |

This prevents incorrect blood group predictions from invalid images.

---

#  Model Performance

| Metric | Value |
|----------|----------|
| Test Accuracy | 88.78% |
| Classification Type | Multi-Class (8 Classes) |
| Input Image Size | 64×64 RGB |
| Model Type | CNN |

---

# Project Screenshots

### Home Page 
<img width="1892" height="1007" alt="home2" src="https://github.com/user-attachments/assets/2889e00f-0d2c-4eaa-8730-9e2ee508d051" />

---

### Login Page
<img width="862" height="832" alt="login" src="https://github.com/user-attachments/assets/d7dc2db6-bc2b-4256-9c80-caca46b77802" />

---

### Signup Page
<img width="741" height="791" alt="singup2" src="https://github.com/user-attachments/assets/80791b71-ef82-412a-9fef-0faa0e6d7a53" />

---

### Main GUI
<img width="1895" height="897" alt="mainGUI" src="https://github.com/user-attachments/assets/76833a23-a0b3-4a5c-8ec8-d3b6f2578484" />

---

### Prediction History
<img width="777" height="707" alt="predictionHistory" src="https://github.com/user-attachments/assets/39fe6c23-402a-4f22-99dc-44b3045730d4" />

---

### ROC-AUC Graph

<img width="2000" height="1600" alt="roc_auc_graph" src="https://github.com/user-attachments/assets/083b7d74-a336-48bc-a4a4-ea54a0bf4501" />

---

### Precision-Recall Graph
<img width="386" height="278" alt="precision_recall_curve" src="https://github.com/user-attachments/assets/5543ba92-9d31-4244-980c-ff7a602abe16" />

---

### Model Accuracy Graph 
<img width="2000" height="1200" alt="model_accuracy_graph" src="https://github.com/user-attachments/assets/d9c6937c-249a-4ead-963b-31fc7643c092" />




---

#  Download Sample Images

Sample fingerprint images used for testing:

🔗 **Sample Images Link:**  
https://drive.google.com/drive/folders/1XUxa8Jxsgfyqlwu7Ot3TZhDHsfdOsxjA?usp=drive_link

---

🔗 **Sample Videos Link:** 
https://drive.google.com/drive/folders/1xqEIs5hs-3zIoKUZo5mgDmKK2M0zjL3Z?usp=drive_link

---





#  Download Trained Models

Due to GitHub file size limitations, trained models are hosted separately.

### Blood Group Prediction Model / Fingerprint Validation Model

🔗 **Group.h5 / Fingerprint_identifier_model.h5 Download Link:**  
https://drive.google.com/drive/folders/1souo_U66zPxI0v-kEl_hIO5Qv8Kx6J9l?usp=drive_link


---

#  Download Database File


🔗 **Database File Link:**  
https://drive.google.com/drive/folders/1UDcMroqvT6MrEcT8mT9EqPGfBo0weJJH?usp=drive_link

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/VAISHNAVi148/Blood-Group-Detection-System.git
```

## Install Required Libraries

```bash
pip install tensorflow
pip install keras
pip install numpy
pip install pillow
pip install opencv-python
pip install matplotlib
pip install scikit-learn
pip install tkvideo
```

---

#  Run the Application

```bash
python GUI_Main.py
```

---

#  Database Information

The application stores prediction history using SQLite.

Stored Information:

- Image Path
- Predicted Blood Group
- Confidence Score
- Date and Time



---

#  Future Enhancements

- Web-Based Version
- Cloud Deployment
- Mobile Application
- Real-Time Fingerprint Scanner Integration
- Advanced CNN Architectures
- Higher Prediction Accuracy
- User Dashboard and Analytics

---

#  Author

**Vaishnavi Suryavanshi**

Bachelor of Technology (Computer Science Engineering)

Project Title:

**Blood Group Detection Using Fingerprint Images**

---

# ⭐ Support

If you found this project useful, please consider giving this repository a ⭐ Star.

Thank you for visiting this project!
