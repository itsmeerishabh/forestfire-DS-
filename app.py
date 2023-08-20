import pickle
from flask import Flask,request,jsonify,render_template
import numpy 
import pandas
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#import both the pickle file (standardscalar and ridge model)

Ridge_model     = pickle.load(open("pickle/Ridge_file.pkl","rb"))
standard_scaler = pickle.load(open("pickle/scaler_file.pkl","rb"))


#route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods = ["GET","POST"])
def predict_datapoint():
    if request.method=='POST': #after submit the form
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]])
        result=Ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')  #for the GET request in get reqst only page will view so in start call fum 
        #this line will work




if __name__=="__main__":
    app.run(host="0.0.0.0")
