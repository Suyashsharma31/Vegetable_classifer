from flask import Flask, request, jsonify , render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('best_vegetable_model.h5')

def image_processing(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = image_processing(image)
        predictions = model.predict(processed_image)
        prediction = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return jsonify({'class': class_names[prediction], 'confidence': f"{confidence*100:.2f}%"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
