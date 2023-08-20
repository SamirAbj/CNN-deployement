import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json

# Load the first model 
with open('model/model_architecture1.json', 'r') as f:
    model_architecture1 = f.read()

# Create the model from the loaded architecture
model1 = model_from_json(model_architecture1)

# Load the model weights
model1.load_weights('model/model_weights1.h5')



# Load the second model 
with open('model/model_architecture2.json', 'r') as f:
    model_architecture2 = f.read()

# Create the model from the loaded architecture
model2 = model_from_json(model_architecture2)

# Load the model weights
model2.load_weights('model/model_weights2.h5')


# Load the third model 
with open('model/model_architecture3.json', 'r') as f:
    model_architecture3= f.read()

# Create the model from the loaded architecture
model3 = model_from_json(model_architecture3)

# Load the model weights
model3.load_weights('model/model_weights3.h5')



from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your models (model_1, model_2, model_3)
# Load your input_shape

@app.route('/')
def index():
    return render_template('index.html')

input_shape=(224,224,3)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        if image_file:
            image = Image.open(image_file)
            image = image.resize((input_shape[0], input_shape[1]))
            image_rgb = image.convert("RGB")
            image_array = np.array(image_rgb)

            # Preprocess the image
            preprocessed_image = image_array.reshape((1,) + input_shape) / 255.0

            # Predict using each model
            predictions = []
            for model in [model1, model2, model3]:
                prediction = model.predict(preprocessed_image)
                predictions.append(prediction)

            # Interpret predictions
            threshold = 0.5  # You can adjust the threshold as needed
            store_predictions = []

            for i, prediction in enumerate(predictions):
                if prediction[0][0] > threshold:  # Compare individual prediction value
                    store_predictions.append(f"Store {i+1}")

            return jsonify({'predicted_stores': store_predictions})

        return jsonify({'error': 'No image file found'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)








