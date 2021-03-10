

import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from fastai.vision.all import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

def parent_label_multi(o):
    return [Path(o).parent.name]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    results = []
    if filename != '':
        path = Path()
        learn_inf = load_learner(path/'invasive_plants_II/fmodel.pkl', cpu=True)
        img = PILImage.create(uploaded_file)
        labels, prediction, probability = learn_inf.predict(img)
        for idx,label in enumerate(learn_inf.dls.vocab):
            results.append(f"category: {label}, prediction: {prediction[idx]}, probablity: {probability[idx]:.2f}")
        print(results)
        
    return render_template('index.html', results=results)
    
if __name__ == '__main__':
   app.run(debug = True)

