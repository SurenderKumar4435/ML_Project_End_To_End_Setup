from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.tree import DecisionTreeRegressor

app  = Flask(__name__)


## import pkl file----------------->>>>>>>>>
reg_model = pickle.load(open("model/reg.pkl","rb"))
standard_scaler = pickle.load(open("model/scaler.pkl","rb"))


## Route for home page ---------->>>>>>>>>>>
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
         Order_ID = float(request.form.get("Order_ID"))
         Quantity_Ordered= float(request.form.get("Quantity_Ordered"))
         Price_Each = float(request.form.get("Price_Each"))
         Month = float(request.form.get("Month"))
         Day = float(request.form.get("Day"))
         Year = float(request.form.get("Year"))
         Hours= float(request.form.get("Hours"))
         Minute = float(request.form.get("Minute"))
         A= str(request.form.get("A"))
         B = str(request.form.get("B"))
         C= str(request.form.get("C"))
         D = str(request.form.get("D"))
         
         
         

         new_data_scaled = standard_scaler.transform([[Order_ID,Quantity_Ordered,Price_Each,Month,Day,Year,Hours,Minute,A,B,C,D]])
         result = reg_model.predict(new_data_scaled)

         return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)

