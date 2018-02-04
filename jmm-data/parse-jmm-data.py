# -*- coding: utf-8 -*-
"""
Created on Tue Feb 3 14:33:51 2018

Load the Parkinson detection dataset from C4.5 format
into Pandas dataframe format.

Provide a mechanism to export to ARFF format for Weka.

Modified from jtay data parser and pangyanham work on his blog.

@author: mcgarrah@gmail.com
"""

import pandas as pd
import numpy as np

# Preprocess with parkinson dataset

# http://blog.pangyanhan.com/posts/2017-02-15-analysis-of-the-adult-data-set-from-uci-machine-learning-repository.ipynb.html
park = pd.read_csv('./parkinsons.data', header=None)
#park = pd.read_csv('./parkinsons.data', header=None, delimiter=r"\s+",)
"""
From: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
name - ASCII subject name and recording number
MDVP:Fo(Hz) - Average vocal fundamental frequency
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
MDVP:Flo(Hz) - Minimum vocal fundamental frequency
MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several  measures of variation in fundamental frequency
MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
NHR,HNR - Two measures of ratio of noise to tonal components in the voice
status - Health status of the subject (one) - Parkinson's, (zero) - healthy
RPDE,D2 - Two nonlinear dynamical complexity measures
DFA - Signal fractal scaling exponent
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation 
"""
park.columns = [
    'name',
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'status',
    'RPDE', 'D2',
    'DFA',
    'spread1', 'spread2', 'PPE'
]
# park.columns = ['age', 'employer', 'fnlwt', 'edu', 'edu_num', 'marital', 'occupation', 'relationship', 'race', 'sex',
#                  'cap_gain', 'cap_loss', 'hrs', 'country', 'income']
# Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
# print(park.ix[park.cap_gain > 0].cap_loss.abs().max())
# print(park.ix[park.cap_loss > 0].cap_gain.abs().max())
# park['cap_gain_loss'] = park['cap_gain'] - park['cap_loss']
# park = park.drop(['fnlwt', 'edu', 'cap_gain', 'cap_loss'], 1)
# park['income'] = pd.get_dummies(park.income)
# print(park.groupby('occupation')['occupation'].count())
# print(park.groupby('country').country.count())
# # http://scg.sdsu.edu/dataset-adult_r/
# replacements = {'Cambodia': ' SE-Asia',
#                 'Canada': ' British-Commonwealth',
#                 'China': ' China',
#                 'Columbia': ' South-America',
#                 'Cuba': ' Other',
#                 'Dominican-Republic': ' Latin-America',
#                 'Ecuador': ' South-America',
#                 'El-Salvador': ' South-America ',
#                 'England': ' British-Commonwealth',
#                 'France': ' Euro_1',
#                 'Germany': ' Euro_1',
#                 'Greece': ' Euro_2',
#                 'Guatemala': ' Latin-America',
#                 'Haiti': ' Latin-America',
#                 'Holand-Netherlands': ' Euro_1',
#                 'Honduras': ' Latin-America',
#                 'Hong': ' China',
#                 'Hungary': ' Euro_2',
#                 'India': ' British-Commonwealth',
#                 'Iran': ' Other',
#                 'Ireland': ' British-Commonwealth',
#                 'Italy': ' Euro_1',
#                 'Jamaica': ' Latin-America',
#                 'Japan': ' Other',
#                 'Laos': ' SE-Asia',
#                 'Mexico': ' Latin-America',
#                 'Nicaragua': ' Latin-America',
#                 'Outlying-US(Guam-USVI-etc)': ' Latin-America',
#                 'Peru': ' South-America',
#                 'Philippines': ' SE-Asia',
#                 'Poland': ' Euro_2',
#                 'Portugal': ' Euro_2',
#                 'Puerto-Rico': ' Latin-America',
#                 'Scotland': ' British-Commonwealth',
#                 'South': ' Euro_2',
#                 'Taiwan': ' China',
#                 'Thailand': ' SE-Asia',
#                 'Trinadad&Tobago': ' Latin-America',
#                 'United-States': ' United-States',
#                 'Vietnam': ' SE-Asia',
#                 'Yugoslavia': ' Euro_2'}
# park['country'] = park['country'].str.strip()
# park = park.replace(to_replace={'country': replacements,
#                                   'employer': {' Without-pay': ' Never-worked'},
#                                   'relationship': {' Husband': 'Spouse', ' Wife': 'Spouse'}})
# park['country'] = park['country'].str.strip()
# print(park.groupby('country').country.count())
# for col in ['employer', 'marital', 'occupation', 'relationship', 'race', 'sex', 'country']:
#     park[col] = park[col].str.strip()
#
# park = pd.get_dummies(park)

print("Are there null values?")
print(park.isnull().values.any())

print("Print a dataframe or two...")
print(park.head())

park = park.rename(columns=lambda x: x.replace('-', '_'))

park.to_hdf('datasets.hdf', 'parkinsons', complib='blosc', complevel=9)
park.to_csv('datasets-parkinsons.csv')



# parkinsons telemonitoring dataset

updrs = pd.read_csv('./parkinsons_updrs.data', header=None)
"""
From: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names

subject# - Integer that uniquely identifies each subject
age - Subject age
sex - Subject gender '0' - male, '1' - female
test_time - Time since recruitment into the trial. The integer part is the 
number of days since recruitment.
motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated
total_UPDRS - Clinician's total UPDRS score, linearly interpolated
Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP - Several measures of 
variation in fundamental frequency
Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA - 
Several measures of variation in amplitude
NHR,HNR - Two measures of ratio of noise to tonal components in the voice
RPDE - A nonlinear dynamical complexity measure
DFA - Signal fractal scaling exponent
PPE - A nonlinear measure of fundamental frequency variation 
"""
updrs.columns = [
    'subject#',
    'age',
    'sex',
    'test_time',
    'motor_UPDRS',
    'total_UPDRS',
    'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE',
    'DFA',
    'PPE'
]

print("Are there null values?")
print(updrs.isnull().values.any())

print("Print a dataframe or two...")
print(updrs.head())

updrs = updrs.rename(columns=lambda x: x.replace('-', '_'))

updrs.to_hdf('datasets.hdf', 'updrs', complib='blosc', complevel=9)
updrs.to_csv('datasets-updrs.csv')



# # How to merge datasets for training and valid
# # Madelon
# madX1 = pd.read_csv('./madelon_train.data', header=None, sep=' ')
# madX2 = pd.read_csv('./madelon_valid.data', header=None, sep=' ')
# madX = pd.concat([madX1, madX2], 0).astype(float)
# madY1 = pd.read_csv('./madelon_train.labels', header=None, sep=' ')
# madY2 = pd.read_csv('./madelon_valid.labels', header=None, sep=' ')
# madY = pd.concat([madY1, madY2], 0)
# madY.columns = ['Class']
# mad = pd.concat([madX, madY], 1)
# mad = mad.dropna(axis=1, how='all')
# mad.to_hdf('datasets.hdf', 'madelon', complib='blosc', complevel=9)
# mad.to_csv('datasets-madelon.csv')