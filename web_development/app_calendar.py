import datetime as dt
import pandas as pd
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from database_functions import *


def day_timeslots(orgcode, modality, dept, date_time, all_data = True):
    '''
    Queries the database for specific orgcode, modality, departmentcode and a single date. Returns
    filtered data as a pandas.DataFrame.
    Arguments:
    ---------
    orgcode: str
    modality: str
    dept: str
    date_time: Datetime.Datetime
    Returns:
    -------
    day_info: pandas.DataFrame
    '''
    query_day = str(date_time.date())
    query = "SELECT apt.Exam_ID, apt.OrgCode, apt.Modality, pat.Age, pat.Patient_ID, pat.Gender, apt.datetime "
    if all_data:
        query += "FROM appointments apt "
    else:
        query += "FROM vresults vr "
        query += "JOIN appointments apt ON apt.Exam_ID = vr.Exam_ID "
    query += "JOIN patients pat ON pat.Patient_ID = apt.Patient_ID "
    query += "WHERE apt.Orgcode='%s' AND apt.Modality='%s' AND apt.DepartmentCode = '%s' " %(orgcode,modality,dept)
    query = query +  "AND `datetime` BETWEEN '"+query_day+"' and '"+query_day+" 23:59:59'"
    info_day = query_data("./data/db/223ADB3_v.db", query)
    # Make the datetime column as datetime objects
    date_time= parse_datetime(info_day.pop('datetime'), dt_format = '%Y-%m-%d  %H:%M:%S')
    df= pd.DataFrame({'datetime':date_time})
    day_info=pd.concat([info_day, df], axis=1)

    return day_info


def parse_datetime(raw_datetime, dt_format= '%m/%d/%Y %H:%M'):
    '''
    Takes a pandas.Series with strings and converts them to datetime when in dt_format.
    The output is a python list.
    Arguments:
    ---------
    raw_datetime: pandas.Series
    dt_format: str
    Returns:
    -------
    datetime_list: list
    '''
    raw_datetime_list = raw_datetime.tolist()
    datetime_list = []
    n_data = raw_datetime.shape[0]
    for i in range(0,n_data):
        try:
            dt_temp = dt.datetime.strptime(raw_datetime_list[i], dt_format)
        except:
            dt_temp = np.nan
        # append list
        datetime_list.append(dt_temp)

    return datetime_list

#function to return Examid, Probability, Status for time slots in a week
def generate_timeslots(orgcode, modality, dept, dt_initial, threshold = 0.5):
    '''
    Gives data of timeslots of a week starting at dt_initial. The data are
    status, exam_id, probability, patient_id, gender, age. status is 0
    when probability <= 0.5, 1 when probability > 0.5 and 2 when the time
    slot is not occupied
    Arguments:
    ---------
    orgcode: str
    modality: str
    dept: str
    dt_initial: Datetime.Datetime
    threshold: float
    Returns:
    -------
    table_data: dict
    days: list
    ts_times: list
    '''
    ts_duration = dt.timedelta(minutes = 30)
    n_ts = 31
    n_days = 7
    time_initial = dt.time(8,0)
    # initialize data array
    table_data = []
    # initialize days list
    days = []
    # iterating over days of the week
    for i in range(0,n_days):
        dt_temp = dt_initial + dt.timedelta(days = i)
        dt_next = dt_temp + dt.timedelta(days = 1)
        days.append(dt_temp.strftime("%A") + '\n'+dt_temp.strftime("%m")+'/'+dt_temp.strftime("%d") )

        info_day=day_timeslots(orgcode, modality, dept, dt_temp, all_data = False)


        #info_day = info_filt.loc[(info_filt.datetime >= dt_temp) & (info_filt.datetime < dt_next)]
        # initialize row
        row = []
        # iterating over timeslots
        for j in range(0,n_ts):
            dt_ts = dt.datetime.combine(dt_temp.date(),time_initial) + j*ts_duration
            dt_ts_end = dt_ts + ts_duration
            info_ts = info_day.loc[(info_day.datetime >= dt_ts) & (info_day.datetime < dt_ts_end)]
            if info_ts.empty:
                exam_id = None
                probability  = None
                patient_id = None
                age = None
                gender = None
                status = 2
            else:
                exam_id = int(info_ts.iloc[0]['Exam_ID']) # Pick only the first examid
                probability = predict_probability(exam_id)
                patient_id = info_ts.iloc[0]['Patient_ID'][0:10]
                age = int(info_ts.iloc[0]['Age'])
                gender = info_ts.iloc[0]['Gender']
                if gender == 'M':
                    gender = 'Male'
                if gender == 'F':
                    gender = 'Female'
                if gender == 'U':
                    gender = 'Unknown'
                if probability > threshold:
                    status = 1
                else:
                    status = 0
            data_dict = {
                "status": status, "exam_id":exam_id, "probability": probability,
                "patient_id":patient_id, "gender":gender, "age":age
            }
            row.append(data_dict)
        table_data.append(row)
    table_data = list(map(list, zip(*table_data))) # To transpose it
    ts_times = []

    for j in range(0,n_ts):
        dt_ts = dt.datetime.combine(dt_initial.date(),time_initial) + j*ts_duration
        time = dt_ts.strftime("%H") + ':'+dt_ts.strftime("%M")
        ts_times.append(time)
    return table_data, days, ts_times


def predict_probability(exam_id):
    '''
    Querying the database and finding the probability for a given exam_id
    Arguments:
    ---------
    exam_id: str
    Returns:
    -------
    probability: float
    '''
    query = "SELECT Probabilities FROM vresults  WHERE Exam_ID = %s" %exam_id
    info_day = query_data("./data/db/223ADB3_v.db", query)
    probability = info_day.loc[0]['Probabilities']
    return probability

# Testing
#if __name__ == '__main__':
#    datetime_list = parse_datetime(pd.Series(['12/10/2017 19:22']))
#    print(datetime_list)
