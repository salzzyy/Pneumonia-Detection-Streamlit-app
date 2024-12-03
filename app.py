import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Custom CSS to set the background and text styles
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #f8c291, #6a89cc);  /* Gradient background */
        color: black;
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
    }

    div.stButton > button:first-child {
        background-color: #38ada9;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        height: 50px;
        width: 200px;
    }

    div.stButton > button:first-child:hover {
        background-color: #b8e994;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths and constants
MODEL_PATH = "models/pneumonia_model.h5"

# Title and description
st.title("Pneumonia Detection App")
st.write(
    "Upload a chest X-ray image, and the app will predict if it shows signs of pneumonia."
)

# Display an image (optional)
st.image(
    "image.png", width=100, caption="Pneumonia Detection App", use_column_width=True
)


@st.cache_resource
def load_trained_model(path):
    """Load the trained model with caching to optimize performance."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


# Load model
model = load_trained_model(MODEL_PATH)
st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an X-ray image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

# Styled button for prediction
if st.button("Predict Pneumonia"):
    if uploaded_file is not None:
        st.write("Processing your uploaded file...")
        # Add your prediction logic here
    else:
        st.write("Please upload an image file first!")


# Function to preprocess the input image
def preprocess_image(image):
    """Resize and normalize the image for the model."""
    try:
        image = image.convert("RGB")  # Ensure RGB format
        img_array = np.array(image.resize((150, 150))) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")


if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize the image to match the model input shape
        img_array = np.array(image.resize((150, 150))) / 255.0  # Normalize to [0, 1]

        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 150, 150, 1)

        # Validate input dimensions
        if img_array.shape[1:] != model.input_shape[1:]:
            raise ValueError(
                f"Input image shape {img_array.shape[1:]} does not match model input shape {model.input_shape[1:]}."
            )

        # Make prediction
        prediction = model.predict(img_array)

        # Interpret prediction
        predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        confidence = (
            prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        )
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.write("Please upload an image file.")
