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