

from flask import Flask, render_template, request, redirect, url_for, abort
from fastai.vision.all import *
import logging
from learner_utils import parent_label_multi,load_infer 


app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

global learn_inf


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    results = []
    img = PILImage.create(uploaded_file)
    labels, prediction, probability = learn_inf.predict(img)
    for idx,label in enumerate(learn_inf.dls.vocab):
        results.append(f"category: {label}, prediction: {prediction[idx]}, probablity: {probability[idx]:.2f}")
    print(results)
    return render_template('index.html', results=results)
    
if __name__ == '__main__':
   learn_inf = load_infer()
   app.run(debug = False)

