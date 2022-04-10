import numpy as np
import pandas as pd
import joblib
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, GridSearchCV, ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, add_dummy_feature
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer, KNNImputer
from dataclasses import dataclass
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, precision_score
import joblib


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
    # x_train.sort_values(by=['Name'], inplace=True)

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
    x_train.at[891, 'Embarked'] = x_train.Embarked.mode()
    x_train.Embarked = x_train.Embarked.astype(str)


# ('num_tranform', std_scale, ['Fare'])],
def train_for_age(x, y):
    ordinal_enc = OrdinalEncoder()
    std_scale = StandardScaler()
    x_t = x.drop(['Name', 'Cabin', 'Ticket'], axis=1).copy()
    kbd = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='kmeans')
    c_transform = ColumnTransformer([('KBinsDiscretizer', kbd, ['Fare']),
                                     ('cat_transform', ordinal_enc, ['Embarked', 'Sex', 'salutation'])],
                                    remainder='passthrough', )
    y_t = kbd.fit_transform(np.array(y).reshape(-1, 1))
    c_transform.fit(x_t)
    x_t = c_transform.transform(x_t)
    x_train, x_test, y_train, y_test = train_test_split(x_t, y_t.ravel(), test_size=0.3, random_state=5, shuffle=False)
    mlp__classifier = MLPClassifier(hidden_layer_sizes=(20, 45),
                                    activation='tanh',
                                    solver='lbfgs',
                                    verbose=False,
                                    warm_start=True,
                                    learning_rate='adaptive',
                                    max_iter=10 ** 5,
                                    early_stopping=True,
                                    )
    mlp__classifier.fit(x_train, y_train)
    val_dump(mlp__classifier, x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test


# return 0


def find_missing_age_category(x: pd.DataFrame):
    data_missing_age: pd.DataFrame = x[x.Age.isna() == True]
    data_with_age: pd.DataFrame = x[x.Age.isna() == False]
    train_age_y = data_with_age['Age']
    missing_ages = data_missing_age['Age']
    train_age_x, test_age_x = data_with_age.drop(['Age'], axis=1), data_missing_age.drop(['Age'], axis=1)
    x_train, x_test, y_train, y_test = train_for_age(train_age_x, train_age_y)
    clf = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/age__classifier')
    y_train_pred = clf.predict(x_train)
    # print(age_model)
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
find_missing_age_category(trainer)

x_train = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/train_age')
x_test = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/x_test')
y_train = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/y_train')
clf = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/age__classifier')
y_test = joblib.load('C:/Users/risan/PycharmProjects/scientificProject/outputs/y_test')
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
print('precision_score from training set', precision_score(y_train, y_train_pred, zero_division=1, average='weighted'))
print('precision_score from test set', precision_score(y_test, y_test_pred, zero_division=1, average='weighted'))
