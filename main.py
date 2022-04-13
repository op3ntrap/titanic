import joblib
from sklearnex import patch_sklearn
import pandas as pd

patch_sklearn()

from dataclasses import dataclass


@dataclass
class DB(object):
    path = 'C:/Users/risan/PycharmProjects/scientificProject'
    data_master = joblib.load(path+'/outputs/master_training_data')
    data_master.sort_index(inplace=True)
    training_data = data_master.iloc[0:891].copy()
    evaluation_data = data_master.iloc[891:].copy()
    training_db = pd.read_csv(path+'/data/Kaggle/titanic/train.csv', index_col=0)
    training_data['Survived'] = training_db.Survived
    

d = DB()
