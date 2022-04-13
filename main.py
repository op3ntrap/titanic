import joblib
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from dataclasses import dataclass


@dataclass
class DB(object):
    path = 'C:/Users/risan/PycharmProjects/scientificProject'
    data_master = joblib.load(path + '/outputs/master_training_data')
    data_master.sort_index(inplace=True)
    training_data = joblib.load(path + '/outputs/train_csv_mod')
    evaluation_data = data_master.iloc[891:].copy()
    training_db = pd.read_csv(path + '/data/Kaggle/titanic/train.csv', index_col=0)
    training_data['Survived'] = training_db.Survived
    eval_data = pd.read_csv(path + '/data/Kaggle/titanic/test.csv', index_col=0)
    evaluation_data['Survived'] = eval_data.Survived
    joblib.dump(evaluation_data, path + '/outputs/test_csv_mod')

    def dump_training_data(self, x):
        joblib.dump(x, self.path + '/outputs/train_csv_mod')

    def dump_eval_data(self, x):
        joblib.dump(x, self.path + '/outputs/test_csv_mod')


d = DB()
