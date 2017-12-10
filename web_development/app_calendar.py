import datetime as dt
import pandas as pd
import numpy as np

def initialize_data():
    info=pd.read_csv('./data/be223a_dataset.csv')
    date_time=parse_datetime(info.ScheduledDTTM_D)
    info1 = info[['Exam ID', 'OrgCode', 'Modality', 'DepartmentCode']]
    df= pd.DataFrame({'datetime':date_time})
    info=pd.concat([info1, df], axis=1)
    return info

def parse_datetime(raw_datetime):
    '''
    Takes a list or pandas.Series with strings and converts them to datetime when in this format '%m/%d/%Y %H:%M'.
    The output is a python list
    '''
    raw_datetime_list = raw_datetime.tolist()
    datetime_list = []
    n_data = raw_datetime.shape[0]
    for i in range(0,n_data):
        try:
            dt_temp = dt.datetime.strptime(raw_datetime_list[i], '%m/%d/%Y %H:%M')
        except:
            dt_temp = np.nan
        # append list
        datetime_list.append(dt_temp)

    return datetime_list

#function to return Examid, Probability, Status for time slots in a week
def generate_timeslots(orgcode, modality, dept, dt_initial, threshold = 0.5):
    info = initialize_data()
    # Filter orgcode, modality, dept
    info_filt = info.loc[(info.OrgCode == orgcode) & (info.Modality == modality) & (info.DepartmentCode == dept)]
    # Time slots intervals duration to check
    ts_duration = dt.timedelta(minutes = 30)
    n_ts = 10
    n_days = 7
    time_initial = dt.time(8,0)
    # initialize data array
    table_data = []
    # initialize days list
    days = []
    # iterating over days of the week
    for i in range(0,7):
        dt_temp = dt_initial + dt.timedelta(days = i)
        dt_next = dt_temp + dt.timedelta(days = 1)
        days.append(dt_temp.strftime("%A") + '\n'+dt_temp.strftime("%m")+'/'+dt_temp.strftime("%d") )
        info_day = info_filt.loc[(info_filt.datetime >= dt_temp) & (info_filt.datetime < dt_next)]
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
                status = 2
            else:
                exam_id = int(info_ts.iloc[0]['Exam ID']) # Pick only the first examid
                probability = predict_probability(exam_id)
                if probability > threshold:
                    status = 1
                else:
                    status = 0
            data_dict = {"status": status, "exam_id":exam_id, "probability": probability}
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
    ## Should be done from the model
    return np.random.random(1)[0]
