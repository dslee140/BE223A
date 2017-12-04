from flask import Flask, render_template, request, make_response
from flask import Markup, redirect
from flask_bootstrap import Bootstrap
from flask_misaka import Misaka
import os
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired
# Demo form
import numpy as np
import xgboost as xgb
import pickle

info=pd.read_csv('../withLabel.csv')
#TODO This will be modified as long as database is set up
info['date'], info['time'] = info['CompletedDTTM_D'].str.split(' ', 1).str
info=info[['Modality','Age','OrgCode','Anatomy','date','Labels']]
features = ['Modality','Age','OrgCode','Anatomy']
feature_tup = [(feature, feature) for feature in features]


#class ChartForm(FlaskForm):
#    orgcode = SelectField('Organization Code', choices = orgcodes_choices)
#    submit = SubmitField('Submit')

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

@app.route('/BarCharts')
def bar_chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    return render_template('barChart.html', values=values, labels=labels)

@app.route('/LineCharts')
def line_chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    return render_template('lineChart.html', values=values, labels=labels)

@app.route('/PieCharts')
def pie_chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    colors = [ "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA","#ABCDEF", "#DDDDDD", "#ABCABC"  ]
    return render_template('pieChart.html', set=zip(values, labels, colors))

def preproc_stacked(info, feature='Modality'):
    labels = info[feature].unique()
    show = []
    noshow = []

    for f in labels:
        completed_count = list(info[info[feature] == f]['Labels'] == 0).count(True)
        noshow_count = list(info[info[feature] == f]['Labels'] == 1).count(True)
        show.append(completed_count)
        noshow.append(noshow_count)

    data = dict()
    data['Show'] = show
    data['NoShow'] = noshow
    stack = []
    for key in data:
        stack.append(key)

    return labels, data, stack

class FeatForm(FlaskForm):
    feature = SelectField('Features', choices = feature_tup)
    submit = SubmitField('Submit')

@app.route('/StackedCharts', methods=['GET', 'POST'])
def stacked_chart():
    form = FeatForm()

    if form.validate_on_submit():
        feature = form.feature.data
        #feature = request.args.get("selector", 'Modality')
        labels, data, stack = preproc_stacked(info, feature)
        return render_template('StackedChart.html', labels=labels, data=data, stack=stack, form=form)
    labels, data, stack = preproc_stacked(info, features[0])
    return render_template('StackedChart.html', labels=labels, data=data, stack=stack, form=form)

@app.route('/calendar')
def calendar():
    return render_template("calendar.html")





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

# Live demo route
@app.route('/livedemo', methods=['GET', 'POST'])
def livedemo():
    form = DemoForm()
    prob = None
    if form.validate_on_submit():
        age = form.age.data
        gender = int(form.gender.data)
        modality = form.modality.data
        orgcode = form.orgcode.data
        X_test = preprocess_user_data(age, gender, modality, orgcode, 1, 17)
        # Load ML model
        model =pickle.load(open("./data/models/XGBoostMidtermModel.dat", "rb"))
        # Predict
        xgdmat_test = xgb.DMatrix(data = X_test, feature_names = feature_names)
        Y_predict = model.predict(data = xgdmat_test)
        prob = Y_predict
        #name = form.name.data
        #form.name.data = ''
    return render_template('livedemo.html', form = form, prob = prob)

#@app.route('/dropdown')
#def chart():
#    labels = ["January","February","March","April","May","June","July","August"]
#    values = [10,9,8,7,6,4,7,8]
#    return render_template('chart.html', values=values, labels=labels)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
