import streamlit as st
import requests
from PIL import Image
import io

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000/predict"

st.title("ECG Classification App")
st.write("Upload an ECG image to predict the patient's condition.")

# Image upload (allowing PNG, JPG, and JPEG)
uploaded_file = st.file_uploader("Choose an ECG image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG Image", use_column_width=True)

        # Send the image to the backend
        if st.button("Predict"):
            try:
                # Convert image to bytes and send to backend
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="PNG")
                image_bytes.seek(0)

                response = requests.post(
                    BACKEND_URL,
                    files={"file": image_bytes}
                )

                # Handle response
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Predicted Class: {data['predicted_class']}")
                    st.info(f"Confidence Score: {data['confidence_score']:.2f}%")
                    st.write("Confidence scores for all class above 10%:")
                    for label, score in data['class_confidences'].items():
                        st.write(f"  - {label}: {score:.2f}%")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    except Exception as e:
        st.error(f"Invalid image file: {e}")
