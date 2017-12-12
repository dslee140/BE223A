import pandas as pd
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from database_functions import *

def filt_shows():
    '''
    Queries the database for Orgcode, Modality and Label Returns
    filtered data as a pandas.DataFrame.

    Arguments:
    ---------
    None
    Returns:
    -------
    data: pandas.DataFrame
    '''

    query = "SELECT `OrgCode`, `Modality`"
    query = query + " FROM appointments WHERE Label='%d'" %(0)
    show = pd.DataFrame(query_data("./data/db/223ADB3.db", query))

    query = "SELECT `OrgCode`, `Modality`"
    query = query + " FROM appointments WHERE Label='%d'" %(1)
    noshow = pd.DataFrame(query_data("./data/db/223ADB3.db", query))

    return show, noshow

# Test
#if __name__ == '__main__':
#    show_df, noshow_df = filt_shows()
#    #print(show)
#    #print(noshow)
#
#    data = pd.concat([show_df, noshow_df], axis=0)
#    show_labels = [1] * len(show_df.index) + [0] * len(noshow_df.index)
#    info = pd.concat([data, show_labels], axis=1)
#
#    print(info)

"""
    feature = 'OrgCode'
    labels = sorted(data[feature].unique())


    for f in labels:
        completed_count = show_df[show_df[feature] == f].count()
        print(completed_count)
"""
