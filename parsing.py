from parsing_icd9 import *
import pandas as pd
import numpy as np
import scipy as sp
import glob as glob
from datetime import datetime
import time
from bisect import bisect 

def parse_datetime(raw_datetime, dtformat):
    """ Parses input raw date and time string into Python datetime object, given the format of raw datetime string
    raw_datetime: string of raw date and time.
    dtformat: string to specify format of raw_datetime. 
    
    returns day of week, day of year, hour in 24-hr format, Python datetime object.     
    """    
    if len(raw_datetime)<5:
        return np.nan, 365, np.nan, np.nan    
    datetime_obj = datetime.strptime(raw_datetime,dtformat) #'%m/%d/%Y %H:%M'
    return datetime_obj.weekday(), datetime_obj.timetuple().tm_yday, datetime_obj.hour, datetime_obj


def bizhour(hh):
    """ Helper function for get_descr_bizhour to map a given hour to "business status", AM, PM, or OFF.  
    hh: hour in 24-hr format
    
    returns business status, AM, PM, or OFF.
    """
    biz = ['OFF','AM','PM','OFF']
    breakpoints = [8, 12, 17]
    return biz[bisect(breakpoints, hh)]

def get_descr_bizhour(hhmat):
    """ Uses helper bizhour to map an array of hours to "busines status", AM, PM, or OFF.
    hhmat: Numpy array of hours
    
    returns Numpy array of business status, AM, PM, or OFF.
    """
    return np.array(list(map(bizhour, hhmat)))

def parse_weather(weatherfilename, featurelist):
    """ Parses the raw weather file given the list of features, and text file of weather file. . 
    weatherfilename: name of text file containing weather info.
    featurelist: name of features of interest in the weather file. 
    
    returns Pandas dataframe of row as day of year [1,365], and column the weather features. 
    """
    dummylines=2
    i=0
    count = 0
    weatherlist = [[] for i in range(len(featurelist))]

    with open(weatherfilename) as f:
        for l in f.readlines():        
            if count > 0:
                if dummylines:
                    dummylines-=1
                    continue
                count-=1
                weatherlist[i-1] += [float(j) for j in l.split()[1:-1]]
            else:
                if i >= len(featurelist):
                    break
                if featurelist[i] in l:
                    i+=1
                    count = 12
                    dummylines = 2
    df = pd.DataFrame(np.array(weatherlist).T,columns=featurelist)  
    narray = np.array([[np.nan], [np.nan], [np.nan], [np.nan]]).T
    df=df.append(pd.DataFrame(narray, columns=featurelist),ignore_index=True)
    #df2 = pd.concat([df.loc[:59], pd.DataFrame(df.iloc[59]).transpose(), df.loc[60:]],axis=0)
    return df

def query_weather(doy, weatherdf):
    """ Querys the weather given day of year, and weather dataframe from parse_weather. 
    doy: Numpy array of day of the year
    weatherdf: weather dataframe from parse_weather
    
    returns Pandas dataframe of weather for each day of year. 
    """
    return weatherdf.iloc[doy-1].reset_index(drop=True)

def get_label(cancel_list, valid_reason):
    """ Creates label of patient no-show given a list of cancellation status for appointment data and valid no-show criteria. 
    cancel_list: Pandas dataframe of cancelation status of appointment data 
    valid_reason: Valid criteria of no-show
    
    returns binary Numpy array of shows (0) and no-shows (1). 
    """
    ct = cancel_list.value_counts()

    labels = np.zeros(cancel_list.shape)
    toinclude = np.array(valid_reason)
    for reason in toinclude: 
        labels += cancel_list ==reason
    labels=(labels>0).astype(int) #Cancel == 1
    return labels

def rename_columns(df):
    """ Renames column whitespace with underscore (_). 
    df: original dataframe 
    
    returns dataframe with renamed columns 
    """
    colnames = df.columns.tolist()
    for i, cn in enumerate(colnames):
        colnames[i] = cn.replace(' ', '_')
    df.columns = colnames
    return df

def parse_patient(patientlist):
    """ Parses the list of patient given the raw data
    patientlist: The raw data relevant to patient, i.e., patient ID, age, and gender.
    
    returns pandas list of unique patient list with ID, age, and gender
    """
    features_pt = patientlist.drop_duplicates(subset=patientlist.columns[0])
    return features_pt

def parsing(data_raw_fname, encoding, dtformat,
            exam_id = 'Exam ID', pt_id = 'Patient ID', age = 'Age', gender = 'Gender'):
    """ Parses the raw data for patient, appointment, and weather database. 
    data_raw_fname: Name of raw data file. 
    encoding: File encoding.
    dtformat: The date and time format in the raw data file. 
    exam_id, pt_id, age, gender: The feature names in the raw data for the respective features. 
    
    returns 3 Pandas parsed dataframes, respectively, for  patient, appointment, and weather. 
    """
    a=time.time()
    print('Reading %s'%data_raw_fname)

    data_raw = pd.read_csv(data_raw_fname, encoding = encoding)
    raw_datetime = data_raw['ScheduledDTTM_D']

    num_samples = raw_datetime.shape[0]

    weekday = np.zeros(num_samples)
    timeofday = np.zeros(num_samples)
    ddofyr = np.zeros(num_samples)
    dtobjs = np.zeros(num_samples, dtype=object)
    tdata = np.zeros([num_samples,3])


    featurelist = ['Minimum Temperature', 'Maximum Temperature', 'Average Temperature', 'Precipitation']
    pt_featurelist = [pt_id, age, gender]
    patientlist = data_raw[pt_featurelist]
    
    features_pt = parse_patient(patientlist)    
    
    icd9_grp = parse_icd9(data_raw['icd9'])

    for i,rd in enumerate(data_raw['ScheduledDTTM_D']):
        weekday[i],ddofyr[i],timeofday[i], dtobjs[i]=parse_datetime(rd,dtformat)

    weathermaster = parse_weather('CA045115.txt', featurelist)
    weathermaster['Dayofyear'] = weathermaster.index
    weathermaster.columns = ['mintemp', 'maxtemp', 'avtemp', 'precip', 'Dayofyear']
    weatherdf = query_weather(ddofyr,weathermaster)
    bizdescr = get_descr_bizhour(timeofday)


    label = get_label(data_raw['ReasonDesc'], ['CANCELLED BY PT', 'PT NO SHOW'])
    features_exam = pd.concat([
        data_raw[[exam_id, pt_id]+['OrgCode','Modality','Anatomy','SubSpecialty']],
        pd.DataFrame({'Weekday':weekday, 'Timeofday':bizdescr, 'Dayofyear':ddofyr,'Datetime Obj':dtobjs,'Label':label, 'ICD Group':icd9_grp})
                         ],axis=1)
    
    print('Processed in %.3f seconds.'% (time.time()-a))
    return rename_columns(features_pt), rename_columns(features_exam), rename_columns(weathermaster)

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    encoding = sys.argv[2]
    dtformat = sys.argv[3]
    exam_id = sys.argv[4]
    pt_id = sys.argv[5]
    age = sys.argv[6]
    gender = sys.argv[7]
    pt, appt, weather = parsing(fname, encoding, dtformat, exam_id, pt_id, age, gender)
