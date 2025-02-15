import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import cv2
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("dataset.csv")

# Separate labels and features
X = df.iloc[:, 1:].values  # Feature columns (4 features)
y = df.iloc[:, 0].values   # Label column (first column)

# Normalize features (CNNs work better with values between 0 and 1)
X = (X-X.min()) / (X.max()-X.min())

# One-hot encode labels (if you have multiple classes)
num_classes = 13  # Ensure num_classes includes the highest label
y = to_categorical(y, num_classes)

# Reshape X to match CNN input shape (Assuming 4 features → (2,2,1) shape)
X = X.reshape(-1, 2, 2, 1)  # Reshaping to (4x4) "image"

# Split dataset into train & test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

# Define CNN model
model = keras.Sequential([
    keras.layers.Conv2D(64, (2, 2), activation="relu", input_shape=(2, 2, 1), padding="same"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),  # This reduces (2,2,64) → (1,1,64)
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)


# Train the model
history = model.fit(X, y, epochs=100, batch_size=64)


# Evaluate on test data
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Make predictions
# predictions = model.predict(X_test)
# print("Sample Prediction:", np.argmax(predictions[0]))  # Predicted label




# test on a image sample
image = cv2.imread("E:/image labelling/reference_images/test_img.JPG", cv2.IMREAD_GRAYSCALE)  # Load as grayscale (update if needed)
image = cv2.resize(image, (2, 2))  # Resize to match the input size of your model
image = image / 255.0  # Normalize (optional)
image = image.reshape(1, 2, 2, 1)  # Reshape for CNN input (batch_size, height, width, channels)
print(image)

# Predict
prediction = model.predict(image)
predicted_label = np.argmax(prediction)
# plt.imshow(image)

# Print results
print(f"Random Image: {image}")
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Probabilities: {prediction[0]}")