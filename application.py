import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import ridge regression model and standard scaler pickle files

try:
    ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    ridge_model = None
    scaler = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.form
        features = [
            float(data["Temperature"]),
            float(data["RH"]),
            float(data["Ws"]),
            float(data["Rain"]),
            float(data["FFMC"]),
            float(data["DMC"]),
            float(data["ISI"]),
            float(data["Classes"]),
            float(data["Region"]),
        ]

        # Convert features to numpy array and reshape for scaler
        features_array = np.array(features).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(features_array)

        # Predict using the ridge model
        prediction = ridge_model.predict(scaled_features)

        return render_template("predict.html", prediction=prediction[0])
    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
