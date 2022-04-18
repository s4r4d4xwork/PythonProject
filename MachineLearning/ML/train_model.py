from joblib import dump, load
# impliment decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
# impliment other algorithms for comparision
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from varname import nameof

data_file = pd.read_csv('data.csv')

# 1 = liked, 0 = disliked
X = data_file.drop('liked', axis=1)
y = data_file['liked']


# 20% sample for test 80% sample for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# training our model using train sample
# n_estimators : int, default=100
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

# DecisionTree - 0
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
predictions_dt = model_dt.predict(X_test)
score_dt = accuracy_score(y_test, predictions_dt)


# RandomForest - 1
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
score_rf = accuracy_score(y_test, predictions_rf)


# ExtraTrees - 2
model_et = ExtraTreesClassifier()
model_et.fit(X_train, y_train)
predictions_et = model_et.predict(X_test)
score_et = accuracy_score(y_test, predictions_et)


# Bagging - 3
model_bg = BaggingClassifier()
model_bg.fit(X_train, y_train)
predictions_bg = model_bg.predict(X_test)
score_bg = accuracy_score(y_test, predictions_bg)


# AdaBoost - 4
model_ab = AdaBoostClassifier()
model_ab.fit(X_train, y_train)
predictions_ab = model_ab.predict(X_test)
score_ab = accuracy_score(y_test, predictions_ab)

# Comparision between models
temp_arr = []
temp_arr.extend([score_dt,score_rf,score_et,score_bg,score_ab])
temp = 0
i = 0
j = 0
while i < len(temp_arr):
    if(temp_arr[i] > temp):
        temp = temp_arr[i]
        j = i
    i = i + 1

# clean output folder
import os
os.chdir("../ML/output")
all_files = os.listdir()

for f in all_files:
    os.remove(f)

# choose the model has highest accuracy and put to output folder
model_name = 'D:\FPT\py-project\ML\output\music_rcm.joblib'
pre_text = 'Using model:'

#switch-case
match j:
    case 0:
        dump(model_dt, model_name)
        print(pre_text + 'Decision Tree')
    case 1:
        dump(model_rf, model_name)
        print(pre_text + 'RandomForest')
    case 2:
        dump(model_et, model_name)
        print(pre_text + 'ExtraTrees')
    case 3:
        dump(model_bg, model_name)
        print(pre_text + 'Bagging')
    case 4:
        dump(model_ab, model_name)
        print(pre_text + 'AdaBoost')


