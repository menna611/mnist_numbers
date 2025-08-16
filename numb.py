import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import cv2
from streamlit_drawable_canvas import st_canvas

# -------------------------
# Load or Train Model
# -------------------------
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("mnist_cnn.h5")
    except:
        # Train quickly if no model found
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
        model.save("mnist_cnn.h5")
    return model

model = load_trained_model()

st.title("✍️ MNIST Digit Recognizer")
st.write("Draw a digit (0–9) in the box below and let the model guess!")

# -------------------------
# Drawing Canvas
# -------------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28,28))
    img = img.reshape(1,28,28,1).astype("float32")/255.0

    # Prediction
    pred = model.predict(img)
    st.subheader(f"Prediction: {np.argmax(pred)}")
    st.bar_chart(pred[0])





