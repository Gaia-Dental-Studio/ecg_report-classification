from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('cnn_model-v2.h5')

# Class from the training process
classes = [
    'Myocardial Infarction Patients',
    'Patient that have History of MI',
    'Patient that have abnormal heartbeat',
    'Normal person'
]

def preprocess_image(image):
    """
    Preprocess a single image for inference.
    Args:
        image (BytesIO): Image file in BytesIO format.
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    # Load the image with a target size
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        # Convert file to BytesIO and preprocess
        image_bytes = BytesIO(file.read())
        preprocessed_image = preprocess_image(image_bytes)
        
        # Predict using the model
        predictions = model.predict(preprocessed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes[predicted_index]
        confidence_score = predictions[0][predicted_index]
        
        # Return the prediction and confidence score
        return jsonify({
            "predicted_class": predicted_class,
            "confidence_score": float(confidence_score),
            "confidence_scores": {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
