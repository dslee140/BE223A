import numpy as np
import xgboost as xgb

from flask_wtf import FlaskForm # to create the form
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired

import pickle # to load the model
import pandas as pd
import seaborn as sns # for the heatmap on the table

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
    '''
    Takes features before one-hot encoding. Encodes the categorical variables.
    Returns the features as a vector ready to input in ML model.
    '''
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
    '''
    Extracts the form input into variables
    '''
    age = form.age.data
    gender = int(form.gender.data)
    modality = form.modality.data
    orgcode = form.orgcode.data
    return age, gender, modality, orgcode

def predict_week(age, gender, modality, orgcode, days = range(0,7), hours = range(8, 24)):
    '''
    Takes all features of the ML model. Days and hours are given as lists. It returns a
    table of the probabilities predicted by the model for all combinations of days and hours.
    The table is in html format.
    '''

    # Loading the model
    model =pickle.load(open("./data/models/XGBoostMidtermModel.dat", "rb"))

    # Initialize prediction table
    Y_predict = np.zeros((len(hours), len(days)))

    # Predict for each cell in the table
    i = 0
    for hour in hours:
        j = 0
        for day in days:
            X_test = preprocess_user_data(age, gender, modality, orgcode, day, hour)
            xgdmat_test = xgb.DMatrix(data = X_test, feature_names = feature_names)
            Y_predict[i, j] = model.predict(data = xgdmat_test)
            j = j + 1
        i = i + 1
    # Column headers to appear on the table
    days_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Rows index to appear on the table
    hours_format = [str(hour) + ":00" for hour in hours]
    # Create Dataframe
    df = pd.DataFrame(Y_predict, index=hours_format, columns=days_names)
    # Colormap for the table
    cm = sns.light_palette("red", as_cmap=True)
    # Render dataframe to html table using bootstrap classes and heatmap (Colormap)
    html_table = df.style.set_table_attributes('class = "table table-hover"').background_gradient(cmap=cm).render()

    return html_table
