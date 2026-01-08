# Generated from: model_v3.ipynb
# Converted at: 2026-01-04T15:17:21.164Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

"""
Referred to https://www.kaggle.com/code/alamebarham/spaceship-titanic-super-understandable-edition
https://www.kaggle.com/competitions/spaceship-titanic/discussion/585514 
Observations:
1. PassengerId -> group size
2. total spent == 0 -> CryptoSleep can be inferred to true

*** Only 1 is applied in this version for experimentation.

Version Info:

Logistic

Missing Value Imputation Strategies:
spending columns → 0
age -> median of age of VIP, non-VIP, and Unknown
categorical → 'Unknown'

New Features:
total spent > 0
age<14
Cabin -> Deck

Deleted Features:
Name
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import missingno as msno

## Load datasets

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

y_train = train["Transported"].astype(int)
X_train = train.drop(columns=["Transported"])
X_test = test.copy()

drop_cols = ['Name']
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

## Missing Value Imputation

# categorical → 'Unknown'
cat_cols = ['HomePlanet','CryoSleep','Destination','Cabin','VIP']
for c in cat_cols:
    X_train[c] = X_train[c].fillna('Unknown')
    X_test[c] = X_test[c].fillna('Unknown')

# spending columns -> 0
spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
X_train[spend_cols] = X_train[spend_cols].fillna(0)
X_test[spend_cols] = X_test[spend_cols].fillna(0)


# Age → median
med_age_all = X_train['Age'].median()
med_age_vip = X_train.loc[X_train['VIP'] == True, 'Age'].median()
med_age_non_vip = X_train.loc[X_train['VIP'] == False, 'Age'].median()
X_train.loc[X_train['VIP'] == True, 'Age'] = X_train.loc[X_train['VIP'] == True, 'Age'].fillna(med_age_vip)
X_train.loc[X_train['VIP'] == False, 'Age'] = X_train.loc[X_train['VIP'] == False, 'Age'].fillna(med_age_non_vip)
X_train.loc[X_train['VIP'] == 'Unknown', 'Age'] = X_train.loc[X_train['VIP'] == 'Unknown', 'Age'].fillna(med_age_all)

X_test.loc[X_test['VIP'] == True, 'Age'] = X_test.loc[X_test['VIP'] == True, 'Age'].fillna(med_age_vip)
X_test.loc[X_test['VIP'] == False, 'Age'] = X_test.loc[X_test['VIP'] == False, 'Age'].fillna(med_age_non_vip)
X_test.loc[X_test['VIP'] == 'Unknown', 'Age'] = X_test.loc[X_test['VIP'] == 'Unknown', 'Age'].fillna(med_age_all)

## New Features

X_train['Group'] = X_train['PassengerId'].str.split('_').str.get(0)
X_train['Number'] = X_train['PassengerId'].str.split('_').str.get(1)
X_train['Group_size'] = X_train.groupby('Group')['PassengerId'].transform('count')

X_test['Group'] = X_test['PassengerId'].str.split('_').str.get(0)
X_test['Number'] = X_test['PassengerId'].str.split('_').str.get(1)
X_test['Group_size'] = X_test.groupby('Group')['PassengerId'].transform('count')

pid_test = test["PassengerId"]
X_train = X_train.drop(columns=['Group','Number','PassengerId'])
X_test = X_test.drop(columns=['Group','Number','PassengerId'])

# total spent > 0
spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

total_spent = X_train[spend_cols].sum(axis=1)
X_train['has_spent'] = (total_spent > 0).astype(int)

total_spent = X_test[spend_cols].sum(axis=1)
X_test['has_spent'] = (total_spent > 0).astype(int)

X_train = X_train.drop('RoomService', axis=1)
X_train = X_train.drop('FoodCourt', axis=1)
X_train = X_train.drop('ShoppingMall', axis=1)
X_train = X_train.drop('Spa', axis=1)
X_train = X_train.drop('VRDeck', axis=1)

X_test = X_test.drop('RoomService', axis=1)
X_test = X_test.drop('FoodCourt', axis=1)
X_test = X_test.drop('ShoppingMall', axis=1)
X_test = X_test.drop('Spa', axis=1)
X_test = X_test.drop('VRDeck', axis=1)

# age < 14
X_train['is_kid'] = (X_train['Age'] < 14).astype(int)

X_test['is_kid'] = (X_test['Age'] < 14).astype(int)

# Cabin -> Deck
X_train['Deck'] = X_train['Cabin'].str.split('/').str.get(0)
X_test['Deck'] = X_test['Cabin'].str.split('/').str.get(0)

X_train = X_train.drop('Cabin', axis=1)
X_test = X_test.drop('Cabin', axis=1)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# train / test alignment
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

X_train.columns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_pred))


test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": pid_test,
    "Transported": test_pred.astype(bool)
})

submission.to_csv("./submission/v3.csv", index=False)