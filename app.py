from flask import Flask,render_template,request
import pickle
import numpy as np


model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('in.html')


@app.route('/predict',methods=['POST'])
def predict():
    Gender = float(request.form.get('Gender'))
    Age = float(request.form.get('Age'))
    Race_Ethnicity = float(request.form.get('Race_Ethnicity'))
    Education_Level_Adults = float(request.form.get('Education_Level_Adults'))
    Marital_Status = float(request.form.get('Marital_Status'))
    Occupation_Status = float(request.form.get('Occupation_Status'))
    Marijuana_Use = float(request.form.get('Marijuana_Use'))
    Sleep_Trouble = float(request.form.get('Sleep_Trouble'))
    Alcohol = float(request.form.get('Alcohol'))
    Diet_Quality_Score = int(request.form.get('Diet_Quality_Score'))
    Health_Score = float(request.form.get('Health_Score'))
    Coffee_Tea = float(request.form.get('Coffee_Tea'))


    #Prediction
    result=model.predict(np.array([
        Gender,
        Age,
        Race_Ethnicity,
        Education_Level_Adults,
        Marital_Status,
        Occupation_Status,
        Marijuana_Use,
        Sleep_Trouble,
        Alcohol,
        Diet_Quality_Score,
        Health_Score,
        Coffee_Tea]).reshape(1, -1))


    return render_template('in.html', result=result)





if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)




