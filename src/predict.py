from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load model
model = load_model('blood_group_model.h5')

# Image path
img_path = 'test_fingerprint.jpg'  # Replace with your image path

# Load and preprocess image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Map class index to label
class_labels = ['A', 'AB', 'B', 'O']  # Update based on your train_gen.class_indices
print(f"Predicted blood group: {class_labels[predicted_class]}")
