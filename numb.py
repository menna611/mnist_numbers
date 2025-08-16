import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("mnist_cnn.h5")

st.title("🖊️ MNIST Digit Classifier")
st.write("Draw or upload a digit (0-9) and let the model predict it.")

# Upload option
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = ImageOps.invert(image)                 # invert (white digit on black background)
    image = image.resize((28, 28))                 # resize
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Processed Image", width=150)
    prediction = model.predict(img_array)
    st.success(f"Predicted digit: {np.argmax(prediction)}")
# Drawing option
else:
    st.write("Draw a digit below:")
    canvas = st.canvas(width=280, height=280, background_color="white", drawing_mode="freedraw")
    if canvas.image_data is not None:
        image = Image.fromarray(canvas.image_data).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        st.image(image, caption="Processed Image", width=150)
        prediction = model.predict(img_array)
        st.success(f"Predicted digit: {np.argmax(prediction)}")