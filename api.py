from itsdangerous import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS,cross_origin
import pickle

app=Flask(__name__)

CORS(app)

model = pickle.load(open("training2.pkl","rb"))


@app.route("/predict",methods=["POST"])
def predict():
    json_=request.json
    query_df=pd.DataFrame(json_)
    prediction=model.predict(query_df)
    return jsonify({"Prediction":list(prediction.astype("str"))})

if __name__=="__main__":
    app.run(debug=True)