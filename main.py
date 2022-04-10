import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
from dataclasses import dataclass

patch_sklearn()

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, GridSearchCV, ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, add_dummy_feature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer, KNNImputer
from dataclasses import dataclass


@dataclass
class DataObject(object):
	training_data_path = 'data/Kaggle/titanic/train.csv'
	test_data_path = 'data/Kaggle/titanic/test.csv'
	training_data = pd.read_csv(training_data_path, index_col=0, header='infer')
	test_data = pd.read_csv(test_data_path, index_col=0, header='infer')
	x_train: pd.DataFrame = training_data.drop(['Survived'], axis=1)
	y_train: pd.Series = training_data['Survived']
	x_train.Name = x_train['Name'].map(lambda x: x.lower())
	x_train.sort_values(by=['Name'], inplace=True)

	def clean_name(self):
		# self.x_train.Name =
		pass


d = DataObject()
# d.x_train

# print(d.x_train.Name)

# patch_sklearn()

x: pd.DataFrame = d.x_train
y: pd.Series = d.y_train

x_missing_age: pd.DataFrame = x[x.Age.isna()]


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


x_missing_age['salutation'] = 0
x_missing_age = x_missing_age.apply(find_salutation, axis=1)
x_missing_age['has_relatives'] = 0


def has_relatives(row):
	if row.SibSp + row.Parch > 1:
		row.has_relatives = 1
	else:
		row.has_relatives = 0
	return row


x_missing_age = x_missing_age.drop(['Age'], axis=1)
ordinal_enc = OrdinalEncoder()
std_scale = StandardScaler()
c_transform = ColumnTransformer(
	[('cat_transform', ordinal_enc, ['Embarked', 'Sex', 'salutation']), ('num_tranform', std_scale, ['Fare'])],
	remainder='passthrough')
x_missing_age_transformed = c_transform.fit_transform(x_missing_age)
print(x_missing_age_transformed)


def prepare_age_feature(X):
	x_missing_age_train = X[X.age.isna() == False]

	x_missing_age_test: pd.DataFrame = x[x.Age.isna()]
	t_age_missing = x_missing_age['Age']
	x_missing_age_test = x_missing_age_test.drop(labels=['Ticket', 'Cabin'], axis=1)
