import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, minmax_scale
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

bc = load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators':[50, 100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1],
    'model__max_depth':[None, 5, 10],
    "model__subsample": [0.5, 0.8, 1.0],
    'model__min_samples_leaf':[1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid= param_grid,
    cv = cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
print(grid.best_params_)
best_model = grid.best_estimator_

y_pred_test = best_model.predict(X_test)
accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
report_test = classification_report(y_true=y_test, y_pred=y_pred_test)

print(f"accuracy_test_rf : {accuracy_test}")
print(f"report_test_rf : {report_test}")

pipe_3 = Pipeline([
    ('scale', StandardScaler()),
    ('model', DecisionTreeClassifier(random_state=42))
])

param_grid_3 = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [None, 5, 10, 15, 20],
    'model__min_samples_split': [2, 10, 20],
    'model__min_samples_leaf': [1, 5, 10],
    'model__max_features': [None, 'sqrt', 'log2']
}

cv_3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_3 = GridSearchCV(
    estimator=pipe_3,
    param_grid= param_grid_3,
    cv = cv_3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)