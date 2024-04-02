from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from flask import Flask,request,jsonify,render_template



'''

if __name__=="__main__":
    logging.info("The execution has started")
    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging_info("custom Exception")
        raise CustomException(e,sys)

'''







## Web API--->>
    
from flask import Flask,request,jsonify,render_template 
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor






app  = Flask(__name__)


## import pkl file----------------->>>>>>>>>
process_model = pickle.load(open("notebook/preprocessor.pkl","rb"))
standard_scaler = pickle.load(open("notebook/scaler2.pkl","rb"))


## Route for home page ---------->>>>>>>>>>>
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
         Pregnancies= float(request.form.get("Pregnancies"))
         Glucose= float(request.form.get("Glucose"))
         BloodPressure = float(request.form.get("BloodPressure"))
         SkinThickness = float(request.form.get("SkinThickness"))
         Insulin = float(request.form.get("Insulin"))
         BMI = float(request.form.get("BMI"))
         DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
         Age= str(request.form.get("Age"))
         
         
         

         new_data_scaled = standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
         result = process_model.predict(new_data_scaled)

         return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)






