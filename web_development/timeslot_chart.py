import datetime as dt
import pandas as pd
import numpy as np
from app_calendar import initialize_data

def timeslots_for_charts(orgcode, modality, dept, dt_initial):
    """
    Generate timeslot with full slots and slots that was taken

    :param orgcode: String of Organization code
    :param modality: String of Modality
    :param dept: String of department code
    :param dt_initial: The initial date parsed with pandas.datetime
    :return: A dictionary with full_slots, slot_taken and date_list with lists
    """

    info = initialize_data()
    info_filt = info.loc[(info.OrgCode == orgcode) & (info.Modality == modality) & (info.DepartmentCode == dept)]
    n_days = 7
    dt_prev = dt_initial - dt.timedelta(n_days)
    info_days = info_filt.loc[(info_filt.datetime >= dt_prev) & (info_filt.datetime <= dt_initial)]

    # Initialize lists and date list in a week
    date_list = []
    date_in_week = [dt_prev + dt.timedelta(n) for n in range(n_days+1)]
    slot_counts = []

    for i, single_date in enumerate(dt_prev + dt.timedelta(n) for n in range(n_days)):

        date_list.append(single_date.strftime("%A") + '\n' + single_date.strftime("%m") + '/' + single_date.strftime("%d"))
        info_i = info_days.loc[(info_days.datetime >= single_date) & (info_days.datetime < date_in_week[i+1])]
        slot_counts.append(int(len(info_i.index)))

    full_slots = np.repeat(np.max(slot_counts), n_days)
    subtract_slots = slot_counts - full_slots
    subtract_slots = [int(f) for f in subtract_slots]
    full_slots = [int(f) for f in full_slots]

    ts_data = dict()
    ts_data['full_slots'] = full_slots
    ts_data['slots_taken'] = slot_counts
    ts_data['date_list'] = date_list

    return ts_data

if __name__ == '__main__':
    orgcode = 'WWH'
    modality = 'CT'
    dept = 'CT'
    dt_initial = dt.datetime(3246, 11, 27)
    ts_data = timeslots_for_charts(orgcode, modality, dept, dt_initial)
    print(ts_data)
