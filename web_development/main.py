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
import livedemo as lvdm

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


# Live demo route
@app.route('/livedemo', methods=['GET', 'POST'])
def livedemo():
    # form instance
    form = lvdm.DemoForm()
    # Initialize variables
    prediction_html_table = None
    predicted = False
    if form.validate_on_submit():
        # Extract form data
        age, gender, modality, orgcode = lvdm.process_demo_form(form)
        # Send data to get prediction table
        prediction_html_table = lvdm.predict_week(age, gender, modality, orgcode)
        predicted = True
    return render_template('livedemo.html', form = form, html_table = prediction_html_table, predicted = predicted)


#@app.route('/dropdown')
#def chart():
#    labels = ["January","February","March","April","May","June","July","August"]
#    values = [10,9,8,7,6,4,7,8]
#    return render_template('chart.html', values=values, labels=labels)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
