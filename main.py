import pickle
import pandas as pd
from flask import Flask,render_template,request
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)
model=  pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    my_predictors = []
    parameters=['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male']
    age = request.form.get('age')
    my_predictors.append(age)
    male = request.form.get('gender')
    my_predictors.append(male)
    cigsPerDay = request.form.get('cigperday')
    my_predictors.append(cigsPerDay)
    sysBP = request.form.get('sysbp')
    my_predictors.append(sysBP)
    diaBP = request.form.get('diabp')
    my_predictors.append(diaBP)
    totChol = request.form.get('chol') 
    my_predictors.append(totChol)
    prevalentHyp =request.form.get('hypertensive')
    my_predictors.append(prevalentHyp)
    diabetes = request.form.get('diabetes')
    my_predictors.append(diabetes)
    glucose = request.form.get('glucose')
    my_predictors.append(glucose)
    BPMeds = request.form.get('bpdabai')
    my_predictors.append(BPMeds)

    my_data = dict(zip(parameters, my_predictors))
    my_df = pd.DataFrame(my_data, index=[0])
    scaler = MinMaxScaler(feature_range=(0,1)) 
   
    # assign scaler to column:
    my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)
    my_y_pred = model.predict(my_df)

    if my_y_pred == 1:
        output ='The patient will develop a Heart Disease.'
    if my_y_pred == 0:
        output= 'The patient will not develop a Heart Disease.'
    
    print(output)
    return render_template('index.html', prediction_text = f'RESULT:{output}')


if __name__=='__main__':
    app.run()