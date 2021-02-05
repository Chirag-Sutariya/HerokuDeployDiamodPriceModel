# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 22:00:45 2021

@author: admin
"""


# import Flask class from flask library

from flask import Flask, render_template, request, jsonify
import numpy as np
from pickle import load

# instanciate app object of Flask class
app = Flask(__name__)

model = load(open('diamond_price_model.pkl','rb'))

# define function to return string in first or home or root page using decorator and route
@app.route("/")
def home():
    return render_template('index.html')
    

@app.route("/predict", methods=["POST"])
def predict():
    raw_input = [float(x) for x in request.form.values()]
    input_vector = np.array(raw_input).reshape(1,-1)
    prediction = model.predict(input_vector)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Price of Diamond sould be $ {}'.format(output))


if __name__=='__main__':
    app.run(debug=True)