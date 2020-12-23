from flask import Flask,render_template,request,redirect,jsonify
import numpy as np
import xgboost
import pickle

with open('ml_model.pkl','rb') as file:
    classifier=pickle.load(file)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html',home_active='active')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        num_preg=request.form.get('Pregnancies')
        glucose_conc=request.form.get('Glucose')
        diastolic_bp=request.form.get('BloodPressure')
        thickness=request.form.get('SkinThickness')
        insulin=request.form.get('InsulinLevel')
        bmi=request.form.get('BodyMassIndex')
        dpf=request.form.get('DiabetesPedigreeFunction')
        age=request.form.get('Age')

        data=np.array([[int(num_preg),int(glucose_conc),int(diastolic_bp),int(thickness),int(insulin),float(bmi),float(dpf),int(age)]])
        prediction=classifier.predict(data)

        context={
            'num_preg':num_preg,
            'glucose_conc':glucose_conc,
            'diastolic_bp':diastolic_bp,
            'thickness':thickness,
            'insulin':insulin,
            'bmi':bmi,
            'dpf':dpf,
            'age':age,
            'pred':prediction
        }        

        return render_template('prediction.html',context=context,pred_active='active')

    elif request.method=='GET':
        return redirect('/')

@app.route('/api')
def api_help():
    return render_template('api.html',api_active='active')

@app.route('/api/<int:num_preg>/<int:glucose_conc>/<int:diastolic_bp>/<int:thickness>/<int:insulin>/<float:bmi>/<float:dpf>/<int:age>')
def api_pred(num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,dpf,age):
    data=np.array([[int(num_preg),int(glucose_conc),int(diastolic_bp),int(thickness),int(insulin),float(bmi),float(dpf),int(age)]])
    prediction=classifier.predict(data)

    result={
            'num_preg':num_preg,
            'glucose_conc':glucose_conc,
            'diastolic_bp':diastolic_bp,
            'thickness':thickness,
            'insulin':insulin,
            'bmi':bmi,
            'dpf':dpf,
            'age':age,
            'pred':bool(prediction[0])
        }

    return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True)