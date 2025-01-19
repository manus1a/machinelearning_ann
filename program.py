import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Initialize the main window
root = tk.Tk()
root.title("Waste Classification Predictor")
root.geometry("600x400")

# Global variables
model = None
image_path = None

# Function to load the model
def load_model():
    global model
    model_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
    if model_path:
        model = tf.keras.models.load_model(model_path)
        messagebox.showinfo("Model Loaded", "Model has been successfully loaded!")

# Function to load the image
def load_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if image_path:
        img = Image.open(image_path)
        img = img.resize((150, 150), Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

# Function to make a prediction
def predict():
    if model is None:
        messagebox.showerror("Error", "Please load a model first!")
        return
    if image_path is None:
        messagebox.showerror("Error", "Please select an image first!")
        return

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = "Organic" if prediction > 0.5 else "Inorganic"

    # Display the result
    result_label.config(text=f"Predicted Class: {predicted_class}")

# GUI Components
# Load Model Button
load_model_button = tk.Button(root, text="Load Model", command=load_model)
load_model_button.pack(pady=10)

# Load Image Button
load_image_button = tk.Button(root, text="Load Image", command=load_image)
load_image_button.pack(pady=10)

# Image Display Label
image_label = tk.Label(root)
image_label.pack(pady=10)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 14))
result_label.pack(pady=10)

# Run the GUI
root.mainloop()