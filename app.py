from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
model = load_model('model.h5')  # Path to your trained Keras .h5 model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Load & preprocess image
            img = Image.open(filepath).convert('RGB').resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0][0]
            prediction = 'Cat 🐱' if pred > 0.5 else 'Dog 🐶'


            # remove uploaded image after prediction
            os.remove(filepath)

    return render_template('project.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
