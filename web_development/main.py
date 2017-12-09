from flask import Flask, render_template, request, make_response, jsonify
from flask import Markup, redirect
from flask_bootstrap import Bootstrap
from flask_misaka import Misaka
import os
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired
import livedemo as lvdm
import markdown

# Demo form
import numpy as np
#import xgboost as xgb
import pickle

#info=pd.read_csv('../withLabel.csv')
#TODO This will be modified as long as database is set up
#info['date'], info['time'] = info['CompletedDTTM_D'].str.split(' ', 1).str
#info=info[['Modality','Age','OrgCode','Anatomy','date','Labels']]
#features = ['Modality','Age','OrgCode','Anatomy']
#feature_tup = [(feature, feature) for feature in features]



#class ChartForm(FlaskForm):
#    orgcode = SelectField('Organization Code', choices = orgcodes_choices)
#    submit = SubmitField('Submit')

class NameForm(FlaskForm):
    name = StringField('What is your name?', validators=[DataRequired()])
    submit = SubmitField('Submit')


content = ""

#with open("12.6 Algorithm Refinement.md", "r") as f:
#     content = f.read()


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
    contents = content
    contents = Markup(markdown.markdown(contents))
    return render_template('home.html', content = contents)


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
    labels = sorted(info[feature].unique())
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

#class FeatForm(FlaskForm):
#    feature = SelectField('Features', choices = feature_tup)
#    submit = SubmitField('Submit')

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


# Live demo route
#@app.route('/livedemo', methods=['GET', 'POST'])
#def livedemo():
#    # form instance
#    form = lvdm.DemoForm()
    # Initialize variables
#    prediction_html_table = None
#    predicted = False
#    if form.validate_on_submit():
        # Extract form data
#        age, gender, modality, orgcode = lvdm.process_demo_form(form)
        # Send data to get prediction table
#        prediction_html_table = lvdm.predict_week(age, gender, modality, orgcode)
#        predicted = True


@app.route('/modalities_json')
def modalities_json():
    '''
    Gets orgcode returns modalities
    '''
    orgcode = request.args.get('orgcode')
    # Retrieve data from database
    # Let's set up just a dict for now
    org_to_mod = {
    'Choose' : ['Choose'],
    'ASM' : ['Choose','CR', 'CT', 'DX', 'MG', 'MR', 'NM', 'OT', 'PR', 'PT', 'RF', 'RG', 'SR', 'US', 'XA'],
    'AWW' : ['Choose', 'CR', 'CT', 'DX']
    }
    modalities = org_to_mod[orgcode]
    return jsonify(result = modalities )

@app.route('/departments_json')
def departments_json():
    '''
    Gets orgcode and modality returns departments
    '''
    orgcode = request.args.get('orgcode')
    modality = request.args.get('modality')

    departments = get_department(orgcode, modality)

    return jsonify(result = departments )

def get_department(orgcode, modality):
    ### query database
    ### use this list instead
    departments = ['Choose', 'CT', 'PCT', 'GEN']
    return departments

@app.route('/calendar_json')
def calendar_json():
    orgcode = request.args.get('orgcode')
    modality = request.args.get('modality')
    # Hard-code the initial date
    initial_date = '3246-12-10'

    table = {
        'columns_names' : ["", "A", "B"],
        'row_names': [ "1", "2", "3"],
        'rows' : [
                    [
                        {"status": 0, "exam_id":12345, "probability": 0.421},
                        {"status": 1, "exam_id":54321, "probability": 0.433}
                    ],
                    [
                        {"status": 1, "exam_id":1253, "probability": 0.768},
                        {"status": 1, "exam_id":1363, "probability": 0.977}
                    ],
                    [
                        {"status":2, "exam_id":None, "probability": 0.123},
                        {"status":2, "exam_id":54684, "probability": 0.465721}
                    ]
        ]
    }
    return jsonify(result = table )

@app.route('/_patient_json')
def patient_json():
    exam_id = request.args.get('exam_id')
    print(exam_id)
    # Look in database for the exam_id and pull information of the patient
    patient_info = {
        'name': "John",
        'telephone': 1112223333,
        'email': 'john@gmail.com',
        'gender': 'Male',
        'age': 83
    }
    return jsonify(result = patient_info)

# Retrieve data from database
# Let's set up just a list for now
orgcodes = np.array(['Choose','ASM', 'AWW'])
orgcodes_choices = [(org, org) for org in orgcodes]
class FiltersForm(FlaskForm):
    orgcode = SelectField('Organization Code', choices = orgcodes_choices)
    modality = SelectField('Exam Modality', choices = [('Choose', 'Choose')])
    departmentcode = SelectField('Department Code', choices = [('Choose', 'Choose')])

class PatientForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices = [('1', 'Male'), ('0', 'Female')])
    submit = SubmitField('Submit')

@app.route('/dashboard')
def dashboard():
    form = FiltersForm()
    patient_form = PatientForm()
    return render_template('dashboard.html' , form =form, patient_form = patient_form)

@app.route('/_render_calendar')
def calendar_data():
    orgcode = 'ASM' #orgcode = request.args.get('orgcode')
    modality = 'CR' #modality = request.args.get('modality')
    timeslots, days = find_timeslots(orgcode, modality)
    return jsonify(timeslots = timeslots, days = days)

def find_timeslots(orgcode, modality):
    # Retrieve data from database
    # Let's set up just a list for now
    timeslots = ["08:00", "10.00", "15:00"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    return timeslots, days

def render_calendar(orgcode, modality):

    return None

#@app.route('/dropdown')
#def chart():
#    labels = ["January","February","March","April","May","June","July","August"]
#    values = [10,9,8,7,6,4,7,8]
#    return render_template('chart.html', values=values, labels=labels)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
