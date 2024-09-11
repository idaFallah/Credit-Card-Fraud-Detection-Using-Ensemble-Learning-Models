#data precprocessing

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset with the help of api sincee dataset is large
! pip install -q kaggle

#uploading api key to colab notebook -> make a directory & copy kaggle.json here
! makedir ~/.kaggle
! cp kaggle.json ~/.kaggle/

# disable api key
! chmod 600 /content/kaggle.json

# importing the dataset
! kaggle datasets download -d mlg-ulb/creditcardfraud

#unzip the dataset
! unzip -q /content/creditcardfraud.zip

#data exploration
dataset = pd.read_csv('/content/creditcard.csv')

dataset.head()

dataset.shape

dataset.columns

dataset.info()

#statistical summary
dataset.describe()

#dealing with missing values
dataset.isnull().values.any()

dataset.isnull().values.sum()

#encoding categorical data

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

#countplot
sns.countplot(dataset['Class'])

#non fraud transactions
(dataset.Class == 0).sum()

#fraud detections
(dataset.Class == 1).sum()

#correlation matrix
dataset_2 = dataset.drop(columns='Class')

dataset_2.corrwith(dataset['Class']).plot.bar(
    figsize=(16,9), title='Correlated with Class', grid=True
)

corr = dataset.corr()

plt.figure(figsize=(25, 25))
ax = sns.heatmap(corr, annot=True, linewidths=2)

#splitting the dataset

dataset.head()

# x -> matrix of features/ independant variables
x = dataset.drop(columns='Class')

# y -> target variable/ dependant features
y = dataset['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

y_train.shape

x_test.shape

y_test.shape

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train

x_test

# building the model

# logistic regression
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state=0)
classifier_LR.fit(x_train, y_train)

y_pred = classifier_LR.predict(x_test)

# evaluating thr model
from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# random forest
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(random_state=0)
classifier_RF.fit(x_train, y_train)

y_pred = classifier_RF.predict(x_test)

# evaluating thr model
from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# XGBoost classifier
from xgboost import XGBClassifier
classifier_XGB = XGBClassifier(random_state=0)
classifier_XGB.fit(x_train, y_train)

y_pred = classifier_XGB.predict(x_test)

# evaluating thr model
from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# final model = XGB classifier

from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# evaluating thr model
from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

dataset.head()

dataset.shape

# predicting one single observation
single_obs = [[0.0, -1.359807,	-0.072781,	2.536347,	1.378155,	-0.338321,	0.462388,	0.239599,	0.098698,	0.363787,	0.090794,	-0.551600,	-0.617801,	-0.991390,	-0.311169,	1.468177,	-0.470401,	0.207971,	0.025791,	0.403993,	0.251412,	-0.018307,	0.277838,	-0.110474,	0.066928,	0.128539,	-0.189115,	0.133558,	-0.021053,	149.62
]]

classifier.predict(sc.transform(single_obs))













