import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.metrics import precision_score
import joblib


# import xgboost as xgb


def get_kbd():
    return KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')


def val_dump(clf, x_train, x_test, y_train, y_test):
    joblib.dump(clf, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/age__classifier')
    joblib.dump(x_train, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/train_age')
    joblib.dump(x_test, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/x_test')
    joblib.dump(y_train, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/y_train')
    joblib.dump(y_test, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/y_test')


@dataclass
class DataObject(object):
    @staticmethod
    def find_salutation(row):
        if 'mr.' in row.Name:
            row.salutation = "MR"
        elif 'dr.' in row.Name:
            row.salutation = "DR"
        elif 'mrs.' in row.Name:
            row.salutation = "MRS"
        elif 'miss.' in row.Name:
            row.salutation = "MISS"
        elif 'master.' in row.Name:
            row.salutation = "MASTER"
        else:
            row.salutation = 'NA'
        return row

    @staticmethod
    def has_relatives(row):
        if row.SibSp + row.Parch > 1:
            row.has_relatives = 1
        else:
            row.has_relatives = 0
        return row

    training_data_path = 'C:/Users/risan/PycharmProjects/scientificProject/data/Kaggle/titanic/train.csv'
    test_data_path = 'C:/Users/risan/PycharmProjects/scientificProject/data/Kaggle/titanic/test.csv'
    training_data = pd.read_csv(training_data_path, index_col=0, header='infer')
    test_data = pd.read_csv(test_data_path, index_col=0, header='infer')
    x_train: pd.DataFrame = training_data.drop(['Survived'], axis=1)
    y_train: pd.Series = training_data['Survived']
    x_train.Name = x_train['Name'].map(lambda i: i.lower())
    test_data.Name = test_data['Name'].map(lambda i: i.lower())
    # test_data.Name = test_data['Name'].map(lambda i: i.lower())

    x_train['salutation'] = None
    x_train['has_relatives'] = 0
    test_data['salutation'] = None
    test_data['has_relatives'] = 0
    si_ = SimpleImputer(strategy='most_frequent')
    origins = x_train.Embarked
    origins_arr = np.array(origins).reshape(-1, 1)
    filled_origins = si_.fit_transform(origins_arr)
    x_train.Embarked = pd.Series(filled_origins.reshape(1, -1)[0])
    x_train.Embarked = x_train.Embarked.fillna(value=x_train.Embarked.mode())
    x_train.at[891, 'Embarked'] = 'S'
    x_train.Embarked = x_train.Embarked.astype(str)


def transform_for_age_eval(x):
    ordinal_enc = OrdinalEncoder()
    std_scale = StandardScaler()
    x_t: pd.DataFrame = x.drop(['Name', 'Cabin', 'Ticket'], axis=1).copy()
    binner = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
    x_t.fillna(axis=0, inplace=True, method='ffill')
    c_transform = ColumnTransformer([('KBinsDiscretizer', binner, ['Fare']),
                                     ('cat_transform', ordinal_enc, ['Embarked', 'Sex', 'salutation'])],
                                    remainder='passthrough')
    c_transform.fit(x_t)
    x_t = c_transform.transform(x_t)
    return x_t


def train_for_age(x, y):
    kbd = get_kbd()
    x_t = transform_for_age_eval(x)
    y_t = kbd.fit_transform(np.array(y).reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x_t, y_t.ravel(), test_size=0.2, random_state=5, shuffle=False)

    # mlp__classifier = MLPClassifier(hidden_layer_sizes=(50, 100),
    #                                 activation='logistic',
    #                                 solver='lbfgs',
    #                                 verbose=False,
    #                                 warm_start=True,
    #                                 learning_rate_init=0.00001,
    #                                 learning_rate='constant',
    #                                 max_iter=10**10,
    #                                 early_stopping=True,
    #                                 )
    # mlp__classifier = RandomForestClassifier(n_estimators=200,
    #                                          random_state=0,
    #                                          class_weight='balanced_subsample',
    #                                          bootstrap=True,
    #                                          n_jobs=10,
    #                                          criterion='entropy', ccp_alpha=0.01)

    mlp__classifier = GradientBoostingClassifier(loss='deviance',
                                                 learning_rate=0.0001, n_estimators=50,
                                                 max_features=None, min_samples_split=2,
                                                 max_depth=10,
                                                 validation_fraction=0.3)

    mlp__classifier.fit(x_train, y_train)
    print(x_train.shape)
    val_dump(mlp__classifier, x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test


# return 0

def find_missing_age_category(x: pd.DataFrame):
    kbd = get_kbd()
    data_missing_age: pd.DataFrame = x[x.Age.isna() == True].copy()
    print("data_missing_age", data_missing_age.size)
    data_with_age: pd.DataFrame = x[x.Age.isna() == False].copy()
    print(data_with_age.shape)

    train_age_y = data_with_age['Age']
    # missing_ages = data_missing_age['Age']
    train_age_x, test_age_x = data_with_age.drop(['Age'], axis=1), data_missing_age.drop(['Age'], axis=1)
    x_train, x_test, y_train, y_test = train_for_age(train_age_x, train_age_y)
    clf = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/age__classifier')
    test_age_x_test = transform_for_age_eval(test_age_x)
    # test_age_x_pred = clf.predict(test_age_x_test)
    # discretize existing age values
    # kbd = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    train_age_y = kbd.fit_transform(np.array(train_age_y).reshape(-1, 1))
    missing_ages = clf.predict(test_age_x_test)
    # print(missing_ages)
    # y_train_pred = clf.predict(x_train)
    # print(age_model)

    data_with_age['Age_Bin'] = pd.Series(train_age_y.ravel(), index=data_with_age.index)
    print(train_age_y.ravel().shape)
    data_missing_age['Age_Bin'] = pd.Series(missing_ages, index=data_missing_age.index)
    print(missing_ages.shape)
    joblib.dump(data_with_age, 'C:/Users/risan/PycharmProjects/scientificProject/outputs'
                               '/data_with_age')
    joblib.dump(data_missing_age, 'C:/Users/risan/PycharmProjects/scientificProject/outputs'
                                  '/data_missing_age')
    x = pd.concat([data_missing_age, data_with_age], axis=0)
    # x.drop(['Age'],axis=1, inplace=True)
    joblib.dump(x, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/master_training_data')
    return "Successfully"


d = DataObject()
trainer: pd.DataFrame = d.x_train
survival_data: pd.Series = d.y_train
validator: pd.DataFrame = d.test_data
trainer = trainer.apply(d.find_salutation, axis=1)
validator = validator.apply(d.find_salutation, axis=1)
# noinspection PyTypeChecker
trainer = trainer.apply(d.has_relatives, axis=1)
# noinspection PyTypeChecker
validator = validator.apply(d.has_relatives, axis=1)
age_data = pd.concat([trainer, validator], axis=0)
joblib.dump(age_data, 'C:/Users/risan/PycharmProjects/scientificProject/outputs/age_data')
find_missing_age_category(age_data)


def age_prediction_performance():
    x_train = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/train_age')
    # print(x_train[0:10])
    x_test = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/x_test')
    y_train = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/y_train')
    clf = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/age__classifier')
    y_test = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/y_test')
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    # print(y_train_pred)
    print('precision_score from training: ',
          precision_score(y_train, y_train_pred, zero_division=1, average='weighted'))
    print('precision_score from test set', precision_score(y_test, y_test_pred, zero_division=1, average='weighted'))
    l1 = precision_score(y_train, y_train_pred, zero_division=1, average='weighted')
    l2 = precision_score(y_test, y_test_pred, zero_division=1, average='weighted')
    with open('C:/Users/risan/PycharmProjects/scientificProject/outputs/precision_log.csv', 'a') as logging:
        logging.write(str(l1) + ',' + str(l2) + '\n')


if __name__ == '__main__':
    age_prediction_performance()
