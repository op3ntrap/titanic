import joblib
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from dataclasses import dataclass
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class DB(object):
    path = 'C:/Users/risan/PycharmProjects/scientificProject'

    # data_master = joblib.load(path + '/outputs/master_training_data')
    # data_master.sort_index(inplace=True)
    # training_data = joblib.load(path + '/outputs/train_csv_mod')
    # evaluation_data = data_master.iloc[891:].copy()
    # training_db = pd.read_csv(path + '/data/Kaggle/titanic/train.csv', index_col=0)
    # training_data['Survived'] = training_db.Survived
    # eval_data = pd.read_csv(path + '/data/Kaggle/titanic/test.csv', index_col=0)
    # evaluation_data['Survived'] = eval_data.Survived
    # joblib.dump(evaluation_data, path + '/outputs/test_csv_mod')

    def dump_training_data(self, x):
        joblib.dump(x, self.path + '/outputs/train_csv_mod')

    def dump_eval_data(self, x):
        joblib.dump(x, self.path + '/outputs/test_csv_mod')


d = DB()


def feature_one_hot(df, f):
    f_one_hot = pd.get_dummies(df[f], prefix=str(f))
    df = pd.concat([df, f_one_hot], axis=1)
    return df.copy()


def transform_data(df: pd.DataFrame):
    # One Hot Encoding of Embarked
    df = feature_one_hot(df, 'Embarked')
    df.Cabin = df.Cabin.fillna('U')
    df.Cabin = df.Cabin.apply(lambda val: val[0])
    df = feature_one_hot(df, 'Cabin')
    df = feature_one_hot(df, 'Age_Bin')
    df = feature_one_hot(df, 'salutation')
    df = feature_one_hot(df, 'Sex')
    df = feature_one_hot(df, 'Pclass')
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df = feature_one_hot(df, 'Family_Size')
    df.Fare.fillna(method='ffill', inplace=True)
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    fares = kbd.fit_transform(np.array(df.Fare).reshape(-1, 1))
    df.Fare = pd.Series(fares.ravel(), index=df.index)
    df = feature_one_hot(df, 'Fare')

    # df = feature_one_hot(df,'')
    df.drop(
        ['Sex', 'Age', 'Name', 'Ticket', 'Cabin', 'Embarked', 'salutation', 'Pclass', 'Fare', 'Family_Size', 'SibSp',
         'Parch','Age_Bin'],
        axis=1,
        inplace=True)
    return df


x = joblib.load(DB.path + '/outputs/train_csv_mod')
# y = x.Survival
x = transform_data(x)
# x.drop(['Survival'], axis=1, inplace=True)
x.to_csv(DB.path+'/outputs/train_csv.csv')
print(x.columns.size)



