#from itsdangerous import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
#from flask_cors import CORS,cross_origin
import pickle
import numpy as np

app=Flask(__name__)

#CORS(app)

model = pickle.load(open("training3.pkl","rb"))
@app.route('/')
def home():
    return render_template('tampilan.html')
data=['LIVED','DIED']
@app.route("/predict",methods=["POST"])
def predict():
    #json_=request.json
    #query_df=pd.DataFrame(json_)
    int_features=[float(x) for x in request.form.values()]
    #prediction=model.predict(query_df)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('tampilan.html', prediction_text='Status {}'.format(data[int(prediction[0])-1]))

if __name__=="__main__":
    app.run(debug=True)