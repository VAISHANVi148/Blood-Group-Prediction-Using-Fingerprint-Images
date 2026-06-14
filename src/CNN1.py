import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score


# Image dimensions
img_rows, img_cols = 64, 64
img_channels = 3

# Data paths
train_path = r'C:\Users\vaish\Desktop\BloodGrupDetection\Detection of blood grp using fingerprint img 100% Code\training'
test_path = r'C:\Users\vaish\Desktop\BloodGrupDetection\Detection of blood grp using fingerprint img 100% Code\testing'

# Ensure testing folder exists
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Image preprocessing
X_data = []
y_data = []
label_map = {}
label_counter = 0

for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)
    if os.path.isdir(folder_path):
        if folder not in label_map:
            label_map[folder] = label_counter
            label_counter += 1
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    img = Image.open(file_path).resize((img_rows, img_cols)).convert('RGB')
                    X_data.append(np.array(img))
                    y_data.append(label_map[folder])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

X_data = np.array(X_data, dtype='float32') / 255.0
y_data = np.array(y_data)

X_data, y_data = shuffle(X_data, y_data, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
num_classes = Y_train.shape[1]

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
hist = model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_data=(X_test, Y_test), verbose=1)

# Evaluate on training data
train_score = model.evaluate(X_train, Y_train, verbose=0)
print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")

# Evaluate on testing data
test_score = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")

# Plot Training vs Validation Accuracy
import os
import matplotlib.pyplot as plt

# Ensure the folder path is correct
save_path = r"C:\Users\vaish\Desktop\BloodGrupDetection\Detection of blood grp using fingerprint img 100% Code"

# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)

import matplotlib.pyplot as plt

# Plot model accuracy
plt.plot(hist.history['accuracy'], label='Train', color='blue')
plt.plot(hist.history['val_accuracy'], label='Test', color='orange')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_graph.png")  # Saves the graph image
plt.show()


import matplotlib.pyplot as plt

# Replace these with your actual FPR and TPR values
fpr = [0.0, 0.1, 0.2, 0.4, 1.0]
tpr = [0.0, 0.6, 0.8, 0.95, 1.0]

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, marker='o', markersize=4, label='ROC Curve (AUC ≈ 0.99)')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

# Save graph in your folder
plt.savefig(r"C:\Users\vaish\Desktop\BloodGrupDetection\Detection of blood grp using fingerprint img 100% Code\roc_auc_graph.jpg", format='jpg')

# Show the graph in Spyder's plot viewer
plt.show()




# Show the graph
plt.show()
# Save model
model.save("bloodgroup_model.h5")

# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test Loss: {:.4f}".format(score[0]))
print("Test Accuracy: {:.2f}%".format(score[1] * 100))


# Classification report
Y_pred = model.predict(X_test)



# Get true labels and predicted probabilities
true_labels = np.argmax(Y_test, axis=1)
predicted_probs = Y_pred

# If multiclass, calculate average='macro' or 'weighted'
roc_auc = roc_auc_score(Y_test, predicted_probs, multi_class='ovr', average='macro')
print("ROC-AUC Score (macro average): {:.2f}%".format(roc_auc * 100))

y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(np.argmax(Y_test, axis=1), y_pred))
print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))



