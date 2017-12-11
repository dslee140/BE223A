import numpy as np
import pandas as pd

def icd9_group(terms, label, low, high):
    lst = np.arange(low, high+1, 1).astype(str).tolist()
    for i in lst:
        terms[i] = label
    return terms

def icd9_term(): 
    terms = {}
    conditionlist = [('blood', 280, 289), ('circulatory', 390, 459),('congenital', 740, 759),('digestive', 520, 579),
    ('enmi', 240, 279),('genitourinary', 580, 629), ('illdefined', 780, 799),('injury', 800, 999),
    ('mental', 290, 319), ('musculoskeletal', 710, 739),('neoplasms', 140, 239),('obstetrics', 630, 679),
    ('perinatal', 760, 779),('respiratory', 460, 519),('skin', 680, 709) ]

    for tt in conditionlist:
        terms = icd9_group(terms, tt[0],tt[1],tt[2])
    terms['nan'] = 'nan'
    return terms

def parse_icd9(icdlist):
    terms = icd9_term()
#    icdlist = df['icd9']
    icdlist = icdlist.astype(str)
    icdlist = icdlist.str.partition('.')[0]
    icdlist = icdlist.str.partition('|')[0]
    icdgrplist = [terms.get(i, 'supplemental') for i in icdlist]
    return icdgrplist 