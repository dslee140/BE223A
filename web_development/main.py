from flask import Flask, render_template, request, make_response
from flask import Markup
from flask_bootstrap import Bootstrap
from flask_misaka import Misaka
import os
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired




class NameForm(FlaskForm):
    name = StringField('What is your name?', validators=[DataRequired()])
    submit = SubmitField('Submit')


content = ""
with open("readme.md", "r") as f:
     content = f.read()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'iridescent'
bootstrap = Bootstrap(app)
Misaka(app) # To use markdown in the template

@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        form.name.data = ''
    return render_template('index.html', form=form, name=name)


@app.route('/user/<name>')
def user(name):
    return render_template('user.html',name=name)


@app.route('/home')
def home():
    return render_template('home.html',text=content)

@app.route('/charts')
def chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    return render_template('chart.html', values=values, labels=labels)

@app.route('/calendar')
def calendar():
    return render_template("calendar.html")

# Demo form

import numpy as np
import xgboost as xgb
import pickle

# Modalities in the order they appear in the one-hot encoding
modalities = np.array(['CR', 'CT', 'DX', 'MG', 'MR', 'NM', 'OT', 'PR', 'PT', 'RF', 'RG', 'SR', 'US', 'XA'])
modalities_choices = [(modal, modal) for modal in modalities] # For the label appearing full names of the modality can be found
# Orgcodes in the order they appear in the one-hot encoding
orgcodes = np.array(['ASM', 'AWW', 'CCHS', 'CKHC', 'JSMO', 'MBIP', 'MP', 'MP1', 'MP1P', 'MP2P', 'MP3',
 'RCPN', 'RICP', 'SHC', 'SMH', 'SMO', 'SMWG', 'WWH'])
orgcodes_choices = [(org, org) for org in orgcodes]
feature_names = ['Age', 'Weekday', 'OrgCode_ASM', 'OrgCode_AWW', 'OrgCode_CCHS', 'OrgCode_CKHC', 'OrgCode_JSMO', 'OrgCode_MBIP', 'OrgCode_MP', 'OrgCode_MP1', 'OrgCode_MP1P', 'OrgCode_MP2P', 'OrgCode_MP3', 'OrgCode_RCPN', 'OrgCode_RICP', 'OrgCode_SHC', 'OrgCode_SMH', 'OrgCode_SMO', 'OrgCode_SMWG', 'OrgCode_WWH', 'Modality_CR', 'Modality_CT', 'Modality_DX', 'Modality_MG', 'Modality_MR', 'Modality_NM', 'Modality_OT', 'Modality_PR', 'Modality_PT', 'Modality_RF', 'Modality_RG', 'Modality_SR', 'Modality_US', 'Modality_XA', 'hour', 'Gender']

class DemoForm(FlaskForm):
    name = StringField('Name of the patient', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices = [('1', 'Male'), ('0', 'Female')])
    modality = SelectField('Exam Modality', choices = modalities_choices)
    orgcode = SelectField('Organization Code', choices = orgcodes_choices)
    #date = DateField('What is the date?', validators=[DataRequired()])
    submit = SubmitField('Submit')

def preprocess_user_data(age, gender, modality, orgcode, day, hour):
    # Process the one-hot encoded data
    orgcode_array = (orgcodes == orgcode)
    modality_array = (modalities == modality)
    # Process continuous data
    age_array = np.array([age])
    day_array = np.array([day])
    hour_array = np.array([hour])
    gender_array = np.array([gender])

    X_array = np.concatenate([age_array, day_array, orgcode_array, modality_array, hour_array, gender_array])
    X_array = np.array([X_array]).astype(float)
    return X_array

def process_demo_form(form):
    age = form.age.data
    gender = int(form.gender.data)
    modality = form.modality.data
    orgcode = form.orgcode.data
    return age, gender, modality, orgcode

import seaborn as sns

def predict_week(age, gender, modality, orgcode, days = range(0,7), hours = range(8, 24)):


    model =pickle.load(open("./data/models/XGBoostMidtermModel.dat", "rb"))

    # Initialize prediction table
    Y_predict = np.zeros((len(hours), len(days)))

    i = 0
    for hour in hours:
        j = 0
        for day in days:
            X_test = preprocess_user_data(age, gender, modality, orgcode, day, hour)
            xgdmat_test = xgb.DMatrix(data = X_test, feature_names = feature_names)
            Y_predict[i, j] = model.predict(data = xgdmat_test)
            j = j + 1
        i = i + 1

    days_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours_format = [str(hour) + ":00" for hour in hours]
    df = pd.DataFrame(Y_predict, index=hours_format, columns=days_names)

    cm = sns.light_palette("red", as_cmap=True)
#    dfs.set_table_styles([
#    {'selector': 'tr :first-child',
#     'props': [('display', 'none')]}
#     ])
    s = df.style.set_table_attributes('class = "table table-hover"').background_gradient(cmap=cm).render()

    #html_code = s.to_html(classes = ["table", "table-hover", "heat-map"])

    return s




# Live demo route
@app.route('/livedemo', methods=['GET', 'POST'])
def livedemo():
    form = DemoForm()
    Y_predict = None
    predicted = False
    if form.validate_on_submit():
        age, gender, modality, orgcode = process_demo_form(form)
        #day, hour  = 1, 17
        #X_test = preprocess_user_data(age, gender, modality, orgcode, day, hour)
        # Load ML model
        #model =pickle.load(open("./data/models/XGBoostMidtermModel.dat", "rb"))
        # Predict
        #xgdmat_test = xgb.DMatrix(data = X_test, feature_names = feature_names)
        #Y_predict = model.predict(data = xgdmat_test)
        #prob = Y_predict


        Y_predict = predict_week(age, gender, modality, orgcode)
        predicted = True
        print(Y_predict)

        return render_template('livedemo.html', form = form, html_code = Y_predict, predicted = predicted)
    return render_template('livedemo.html', form = form, html_code = Y_predict, predicted = predicted)


#@app.route('/dropdown')
#def chart():
#    labels = ["January","February","March","April","May","June","July","August"]
#    values = [10,9,8,7,6,4,7,8]
#    return render_template('chart.html', values=values, labels=labels)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
