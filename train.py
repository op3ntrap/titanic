from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    BaggingClassifier, StackingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, criterion='entropy', random_state=42, max_features=None)
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.00001, n_estimators=100, random_state=42,
                                 validation_fraction=0.05)
base_SVC = SVC(kernel='rbf', degree=3, probability=True, random_state=42, decision_function_shape='ovr',
               break_ties=True)
bc = BaggingClassifier(base_estimator=base_SVC, n_estimators=100, bootstrap_features=True, oob_score=True,
                       random_state=42, n_jobs=4)
ec = ExtraTreesClassifier(n_estimators=100, bootstrap=True, oob_score=True, max_features=None, random_state=42)
abc = AdaBoostClassifier(base_estimator=base_SVC, learning_rate=0.001, random_state=42)

mlpc = MLPClassifier(hidden_layer_sizes=(50, 100),
                     activation='logistic',
                     solver='lbfgs',
                     verbose=False,
                     warm_start=True,
                     learning_rate_init=0.00001,
                     learning_rate='constant',
                     max_iter=10 ** 10,
                     early_stopping=True,
                     random_state=42)
lg = LogisticRegression(random_state=42)
_estimators = [('rfc', rfc),
              ('gbc', gbc),
              ('bc', bc),
              ('ec', ec),
              ('abc', abc),
              ('mlpc', mlpc),
              ]
clf = StackingClassifier(estimators=_estimators, final_estimator=LogisticRegression())
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
