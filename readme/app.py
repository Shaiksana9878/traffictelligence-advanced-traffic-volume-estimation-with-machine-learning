# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

import numpy as np 
import pickle
import matplotlib.pyplot as plt 
import time 
import pandas 
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and encoder
with open(r"C:\Users\madhu\madhu aiml\.ipynb_checkpoints\flask\model.pkl", 'rb') as file:
    model = pickle.load(file)

with open(r"C:\Users\madhu\madhu aiml\.ipynb_checkpoints\flask\encoder.pkl", 'rb') as file:
    scale = pickle.load(file)
print("Model's feature names:", model.feature_names_in_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    print("Form data:", request.form)
    
    # Create a dictionary to store form values, converting keys to lowercase
    form_data = {key.lower(): float(value) for key, value in request.form.items()}
    print("Processed form data:", form_data)
    
    # Use the model's feature names
    feature_names = model.feature_names_in_
    print("Model's feature names:", feature_names)
    
    # Create input_feature list in the correct order
    input_feature = []
    for feature in feature_names:
        if feature in form_data:
            input_feature.append(form_data[feature])
        else:
            print(f"Warning: {feature} not found in form data")
            input_feature.append(0)  # or some default value
    
    print("Number of input features:", len(input_feature))
    print("Input features:", input_feature)
    
    # Create DataFrame
    data = pandas.DataFrame([input_feature], columns=feature_names)
    print("DataFrame:", data)
    
    # Make prediction
    prediction = model.predict(data)
    
    text = "Estimated Traffic Volume is: "
    return render_template("output.html",result = text + str(prediction) + "units")

if __name__ == "__main__":
    print("Model's feature names:", model.feature_names_in_)
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)


# %%

@app.route("/")

# * running the app

# %%



