import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Nadam
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define paths
train_dir = r"C:/Users/user/Desktop/TRAIN_MODEL/dataset/train"
test_dir = r"C:/Users/user/Desktop/TRAIN_MODEL/dataset/test"
model_save_path = r"C:/Users/user/Desktop/TRAIN_MODEL/waste_classification_model.h5"

# Image parameters
img_size = (128, 128)  # Resize images to 128x128
channels = 3  # Ensure all images have 3 channels (RGB)

# Function to load images into memory
def load_images_from_folder(folder, label, img_size, channels):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')  # Convert to RGB (3 channels)
            img = img.resize(img_size)  # Resize to the specified dimensions
            img_array = np.array(img) / 255.0  # Normalize pixel values
            if img_array.shape == (img_size[0], img_size[1], channels):  # Check shape
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return np.array(images), np.array(labels)

# Load organic and inorganic images
organic_images, organic_labels = load_images_from_folder(os.path.join(train_dir, "organic"), 1, img_size, channels)
inorganic_images, inorganic_labels = load_images_from_folder(os.path.join(train_dir, "inorganic"), 0, img_size, channels)

# Combine the data
X = np.vstack((organic_images, inorganic_images))
y = np.hstack((organic_labels, inorganic_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the image data for SMOTE
n_samples, height, width, channels = X_train.shape
X_train_flattened = X_train.reshape(n_samples, -1)  # Shape: (n_samples, height * width * channels)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_flattened, y_train)

# Reshape the data back to its original shape
X_resampled = X_resampled.reshape(-1, height, width, channels)

# Check the class distribution after SMOTE
print("Class distribution after SMOTE:", np.unique(y_resampled, return_counts=True))

# Build the strict ANN model
def build_ann_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),  # Flatten the image data
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Hidden layer
        BatchNormalization(),  # Normalize the activations
        Dropout(0.5),  # Dropout for regularization
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Hidden layer
        BatchNormalization(),  # Normalize the activations
        Dropout(0.5),  # Dropout for regularization
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Hidden layer
        BatchNormalization(),  # Normalize the activations
        Dropout(0.5),  # Dropout for regularization
        Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])
    return model

# Learning rate warmup and cyclic learning rate scheduler
def lr_schedule(epoch):
    warmup_epochs = 10
    max_lr = 0.001
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs  # Linear warmup
    else:
        return max_lr * (0.5 ** (epoch // 20))  # Cyclic decay

# Compile the model with Nadam optimizer
model = build_ann_model((height, width, channels))
optimizer = Nadam(learning_rate=0.0001)  # Nadam optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(
    X_resampled, y_resampled,
    batch_size=32,
    epochs=100,  # Increase the number of epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save the model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training and validation loss and accuracy
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Visualize training history
plot_training_history(history)

# Additional evaluation metrics
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("F1 Score:", f1_score(y_test, y_pred_classes))
print("Precision:", precision_score(y_test, y_pred_classes))
print("Recall:", recall_score(y_test, y_pred_classes))