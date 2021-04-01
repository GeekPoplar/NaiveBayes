import os

import numpy as np

from classification.naive_bayes import NaiveBayes

nb = NaiveBayes()
ROOT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
nb.load_model(os.path.join(MODEL_DIR, "nba.model.json"))
nb.forecast(
    x_validation=np.array(['0.51', '0.7', '5.0', '7.0', '1.0', '17.0']),
    y="PF"
)

nb.forecast(
    x_validation=np.array(['0.47', '0.8', '3.0', '2.0', '0.0', '11.0']),
    y="SG"
)
