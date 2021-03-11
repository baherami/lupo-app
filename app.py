

from flask import Flask, render_template, request, redirect, url_for, abort
from fastai.vision.all import *
import logging


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
    global learn_inf
    labels, prediction, probability = learn_inf.predict(img)
    for idx,label in enumerate(learn_inf.dls.vocab):
        results.append([label, prediction[idx],f"{probability[idx]:.2f}"])
    return render_template('index.html', results=results)

def parent_label_multi(o):
    return [Path(o).parent.name]

if __name__ == '__main__':
    path = Path()
    global learn_inf
    learn_inf = load_learner(path/'invasive_plants_II/fmodel.pkl', cpu=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)
