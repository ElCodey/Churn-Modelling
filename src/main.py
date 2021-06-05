import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import RandomOverSampler

def preprocessing(df):
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    
    df["Gender"] = df["Gender"].replace({"Female": 0, "Male":1})
    
    dummies = pd.get_dummies(df["Geography"])
    df = pd.concat((df, dummies), axis=1)
    df = df.drop("Geography", axis=1)
    
    return df

def split_and_scale(df):
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    
    grad_model = GradientBoostingClassifier()
    grad_model.fit(X_train, y_train)
    grad_pred = grad_model.predict(X_test)
    
    forest_model = RandomForestClassifier()
    forest_model.fit(X_train, y_train)
    forest_pred = forest_model.predict(X_test)
    
    mlp_model = MLPClassifier()
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    
    
    return log_pred, grad_pred, forest_pred, mlp_pred

def over_sampling(X_train, y_train):
    sampler = RandomOverSampler(random_state=1)
    X_train_oversample, y_train_oversample = sampler.fit_resample(X_train, y_train)
    return  X_train_oversample, y_train_oversample