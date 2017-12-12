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
from app_calendar import *
from datetime import datetime,timedelta
import markdown
from timeslot_chart import *
from read_db import filt_shows

# Demo form
import numpy as np
#import xgboost as xgb
import pickle

content = ""
with open("./data/ML_Analysis.md", "r") as f:
     content = f.read()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'iridescent'
bootstrap = Bootstrap(app)
Misaka(app) # To use markdown in the template


@app.route('/Analytics')
def analytics():
    contents = content
    contents = Markup(markdown.markdown(contents))
    return render_template('home.html',content = contents)
    #return render_template('ML_Analysis.html')

@app.route('/for_orgcode_charts_json')
def for_orgcode_charts(feature='OrgCode'):
    show_df, noshow_df = filt_shows()
    show_labels = [0] * len(show_df.index) + [1] * len(noshow_df.index)
    info = pd.concat([show_df, noshow_df], axis=0)
    info['Labels'] = show_labels
    labels = sorted(info[feature].unique())
    show = []
    noshow = []

    for f in labels:
        completed_count = list(info[info[feature] == f]['Labels'] == 0).count(True)
        noshow_count = list(info[info[feature] == f]['Labels'] == 1).count(True)
        show.append(completed_count)
        noshow.append(noshow_count)
    data = dict()

    percent_show = []
    percent_noshow = []
    for i, s in enumerate(show):
        total = s + noshow[i]
        if total == 0:
            percent_show.append(s)
            percent_noshow.append(noshow[i])
        else:
            percent_show.append(s/total)
            percent_noshow.append(noshow[i]/total)

    data['Show'] = percent_show
    data['NoShow'] = percent_noshow

    stack = []
    for key in data:
        stack.append(key)

    data['stack'] = stack
    data['labels'] = labels
    return jsonify(result = data)

@app.route('/for_modality_charts_json')
def for_modality_charts(feature='Modality'):
    show_df, noshow_df = filt_shows()
    show_labels = [0] * len(show_df.index) + [1] * len(noshow_df.index)
    info = pd.concat([show_df, noshow_df], axis=0)
    info['Labels'] = show_labels
    orgcode = request.args.get('orgcode')
    if orgcode == 'Choose':
        data = dict()
    else:
        data = dict()

        labels = sorted(info[feature].unique())
        show = []
        noshow = []

        for f in labels:
            info_temp = info[info['OrgCode'] == orgcode]
            completed_count = list(info_temp[info_temp[feature] == f]['Labels'] == 0).count(True)

            noshow_count = list(info_temp[info_temp[feature] == f]['Labels'] == 1).count(True)
            show.append(completed_count)
            noshow.append(noshow_count)

        percent_show = []
        percent_noshow = []
        for i, s in enumerate(show):
            total = s + noshow[i]
            if total == 0:
                percent_show.append(s)
                percent_noshow.append(noshow[i])
            else:
                percent_show.append(s/total)
                percent_noshow.append(noshow[i]/total)

        data['Show'] = percent_show
        data['NoShow'] = percent_noshow
        stack = []
        for key in data:
            stack.append(key)

        data['stack'] = stack
        data['labels'] = labels

    return jsonify(result = data)

@app.route('/calendar')
def calendar():
    return render_template("calendar.html")




@app.route('/modalities_json')
def modalities_json():
    '''
    Gets orgcode returns modalities
    '''
    orgcode = request.args.get('orgcode')
    if orgcode == 'Choose':
        modalities = ['Choose']
    else:
        modalities = ['Choose']+list(hospital_map[orgcode].keys())
    return jsonify(result = modalities )

@app.route('/departments_json')
def departments_json():
    '''
    Gets orgcode and modality returns departments
    '''
    orgcode = request.args.get('orgcode')
    modality = request.args.get('modality')
    departments = ['Choose']+hospital_map[orgcode][modality]
    return jsonify(result = departments )

@app.route('/calendar_json')
def calendar_json():
    orgcode = request.args.get('orgcode')
    modality = request.args.get('modality')
    dept = request.args.get('departmentcode')
    # The initial date on the week calendar hard-coded, to be updated when data are current
    initial_date = datetime(3246,12,10)
    table_data, days, ts_times = generate_timeslots(orgcode, modality, dept, initial_date)

    ts_data = timeslots_for_charts(orgcode, modality, dept, initial_date)

    prev_week1 = initial_date - timedelta(days = 7)
    ts_data2 = timeslots_for_charts(orgcode, modality, dept, prev_week1)

    columns_names = [' ']+ days
    table = {
        'columns_names' : columns_names,
        'row_names': ts_times,
        'rows' : table_data,
        'ts_data': ts_data,
        'ts_data2': ts_data2
    }
    return jsonify(result = table)


# Retrieve data from hospital map
orgcodes = ['Choose']+list(hospital_map.keys())
orgcodes_choices = [(org, org) for org in orgcodes]
class FiltersForm(FlaskForm):
    orgcode = SelectField('Organization Code', choices = orgcodes_choices)
    modality = SelectField('Exam Modality', choices = [('Choose', 'Choose')])
    departmentcode = SelectField('Department Code', choices = [('Choose', 'Choose')])


@app.route('/')
def dashboard():
    form = FiltersForm()
    return render_template('dashboard.html', form =form)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
