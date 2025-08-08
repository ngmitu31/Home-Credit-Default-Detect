import numpy as np
import pandas as pd 
import os
import sys

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the parent directory
parent_directory = os.path.dirname(current_directory)

# Add the parent directory to sys.path
sys.path.append(parent_directory)

from Tools.helper import *
from Tools.pipeline import *
from Tools.tuning import *


#Import Preprocessing data after
bureau = pd.read_csv(r"Final_Project\Kaggle_Output\Bureau_aggregrate_OH.csv")
pos = pd.read_csv(r"Final_Project\Kaggle_Output\Pos_aggregrate_OH.csv")
prev = pd.read_csv(r"Final_Project\Kaggle_Output\Prev_aggregrate_OH.csv")
credit = pd.read_csv(r"Final_Project\Kaggle_Output\Credit_aggregrate_OH.csv")
install = pd.read_csv(r"Final_Project\Kaggle_Output\Install_aggregrate_OH.csv")
train = pd.read_csv(r"Final_Project\Kaggle_Output\application_train_OH.csv")
test = pd.read_csv(r"Final_Project\Kaggle_Output\application_train_OH.csv")

#Merging all table into application 
train = train.merge(bureau, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_bureau'))
train = train.merge(credit, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_credit'))
train = train.merge(install,  how = 'left', on = 'SK_ID_CURR', suffixes=('', '_install'))
train = train.merge(pos, how = 'left', on = 'SK_ID_CURR', suffixes = ('', '_pos'))
train = train.merge(prev, how = 'left', on = 'SK_ID_CURR', suffixes = ('', '_prev'))

test = test.merge(bureau, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_bureau'))
test = test.merge(credit, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_credit'))
test = test.merge(install,  how = 'left', on = 'SK_ID_CURR', suffixes=('', '_install'))
test = test.merge(pos, how = 'left', on = 'SK_ID_CURR', suffixes = ('', '_pos'))
test = test.merge(prev, how = 'left', on = 'SK_ID_CURR', suffixes = ('', '_prev'))

#Checking missing column train_columns = set(train.columns)
test_columns = set(test.columns)
train_columns = set(train.columns)

# Identify the missing column(s)
missing_columns = test_columns - train_columns
print("Missing columns in training set:", missing_columns)

#Fillna 
train = fill_missing_values(train)
test = fill_missing_values(test)

#Check non-numeric if still havve
print(check_non_numeric_columns(train))
print(check_non_numeric_columns(test))

#If still have non-numeric column 
train = one_hot_encode_auto(train)
test = one_hot_encode_auto(test)

#Predict_Probability
submission = train_and_predict_submission_with_scaling_and_feature_selection(train, test, 'TARGET')
submission.to_csv(r"Final_Project\Ouput_Data\Prediction\Prediction_Proba.csv")