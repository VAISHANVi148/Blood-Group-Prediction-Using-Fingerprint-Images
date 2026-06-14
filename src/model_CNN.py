import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# PARAMETERS
img_rows, img_cols = 64, 64
img_channels = 3
dataset_path = 'C:\\Users\\vaish\\Documents\\Detection_of_blood_grp_using_fingerprint_img_100%_Code[1]\\Detection of blood grp using fingerprint img 100% Code\\training'

# Load images and labels
X = []
y = []
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img = Image.open(img_path).resize((img_rows, img_cols)).convert('RGB')
                X.append(np.array(img))
                y.append(folder)
            except:
                print("Error loading:", img_path)

X = np.array(X)
y = np.array(y)

# Normalize
X = X.astype('float32') / 255.0

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y_cat = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, img_channels)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

# Save model
model.save("bloodgroup_classifier.h5")

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Plot accuracy/loss
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.legend()
plt.show()





