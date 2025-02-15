import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ✅ Load class names from labels.txt
labels_file = "E:/image labelling/labels.txt"
with open(labels_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)
print(f"Loaded {num_classes} class labels: {class_names}")

# ✅ Define dataset path
dataset_path = "E:/image labelling/images_dataset_mlsc"
image_size = (64, 64)
batch_size = 32

# ✅ Load dataset (Train + Validation from the same folder)
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

# ✅ Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ✅ Define CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# ✅ Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ✅ Print model summary
model.summary()

# ✅ Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=25)

# ✅ Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ✅ Save trained model
model.save("image_classifier_model.h5")

# ✅ Load and predict on a test image
image_path = "E:/image_labelling/images_dataset_mlsc/sehajjot.JPG"
image = load_img(image_path, color_mode="grayscale", target_size=(64, 64))
image = img_to_array(image) / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Add batch dimension

# ✅ Predict
prediction = model.predict(image)
predicted_label_index = np.argmax(prediction)

# ✅ Get class name from labels.txt
predicted_label = class_names[predicted_label_index]

# ✅ Print results
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Probabilities: {prediction[0]}")
