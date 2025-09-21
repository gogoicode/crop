import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open(r"C:\Users\aayan\OneDrive\Desktop\MCA1_Code\pjt\crop\model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    features = pd.DataFrame([float_feature], columns=columns)
    prediction  = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is: {}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
