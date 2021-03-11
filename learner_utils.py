from fastai.vision.all import Path, load_learner
import logging

def parent_label_multi(o):
    return [Path(o).parent.name]
def load_infer():
    path = Path()
    return load_learner(path/'invasive_plants_II/fmodel.pkl', cpu=True)
