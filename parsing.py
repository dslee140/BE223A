import pandas as pd
import numpy as np
import scipy as sp
import glob as glob
from datetime import datetime
import time
from bisect import bisect 

def parse_datetime(raw_datetime):
    if len(raw_datetime)<5:
        return np.nan, 365, np.nan, np.nan    
    datetime_obj = datetime.strptime(raw_datetime,'%m/%d/%Y %H:%M')
    return datetime_obj.weekday(), datetime_obj.timetuple().tm_yday, datetime_obj.hour, datetime_obj


def bizhour(hh):
    biz = ['OFF','AM','PM','OFF']
    breakpoints = [8, 12, 17]
    return biz[bisect(breakpoints, hh)]

def get_descr_bizhour(hhmat):
    return np.array(list(map(bizhour, hhmat)))

def load_weather(weatherfilename, featurelist):
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
    return weatherdf.iloc[doy-1].reset_index(drop=True)

def get_label(cancel_list, valid_reason):
    ct = cancel_list.value_counts()

    labels = np.zeros(cancel_list.shape)
    toinclude = np.array(valid_reason)
    for reason in toinclude: 
        labels += cancel_list ==reason
    labels=(labels>0).astype(int) #Cancel == 1
    return labels



def parsing(data_raw_fname):
    a=time.time()
    print('Reading %s'%data_raw_fname)
    #data_raw_fname = 'be223a_dataset.csv'
    data_raw = pd.read_csv(data_raw_fname)
    raw_datetime = data_raw['ScheduledDTTM_D']

    num_samples = raw_datetime.shape[0]

    weekday = np.zeros(num_samples)
    timeofday = np.zeros(num_samples)
    ddofyr = np.zeros(num_samples)
    dtobjs = np.zeros(num_samples, dtype=object)
    tdata = np.zeros([num_samples,3])


    featurelist = ['Minimum Temperature', 'Maximum Temperature', 'Average Temperature', 'Precipitation']


    for i,rd in enumerate(data_raw['ScheduledDTTM_D']):
        weekday[i],ddofyr[i],timeofday[i], dtobjs[i]=parse_datetime(rd)
        #tdata[i,:], dtobjs[i]=parse_datetime(rd)


    weathermaster = load_weather('CA045115.txt', featurelist)
    weatherdf = query_weather(ddofyr,weathermaster)
    bizdescr = get_descr_bizhour(timeofday)


    label = get_label(data_raw['ReasonDesc'], ['CANCELLED BY PT', 'PT NO SHOW'])
    features = pd.concat([
        data_raw[['Gender','Age','OrgCode','Modality','Anatomy','SubSpecialty']],
        pd.DataFrame({'Weekday':weekday, 'Timeofday':bizdescr,'Datetime Obj':dtobjs,'Label':label}),
        weatherdf
                         ],axis=1)

    print('Processed in %.3f seconds.'% (time.time()-a))
    return features

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    parsing(fname)