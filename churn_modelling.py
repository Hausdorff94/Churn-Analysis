import pickle

import pandas as pd
from sklearn import ensemble, linear_model, metrics, neighbors, svm, tree
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold


class modelling(object):
    def __init__(self, df=None, classif=None, preproc_scaler=None, class_trained=None, target=True):
        self.df = df
        self.classif = classif
        self.preproc_scaler = preproc_scaler
        self.class_trained = class_trained
        self.target = target

    def var_dummies(self):
        df_model = self.df.iloc[:, 1:].copy()
        df_model['SeniorCitizen'] = df_model['SeniorCitizen'].apply(
            lambda i: "Yes" if i == 1 else "No")
        return pd.get_dummies(df_model)

    def transform_data(self, **kwargs):
        df_model = self.var_dummies()
        if self.target:
            X = df_model.drop(columns='Churn').values
            y = df_model['Churn']
            sc = self.preproc_scaler(**kwargs)
            sc.fit(X)
            X = sc.transform(X)
            X = pd.DataFrame(sc.transform(
                X), columns=df_model.drop(columns='Churn').columns)
            pickle.dump(sc, open('scaler.pkl', 'wb'))
            return X, y
        else:
            X = df_model.values
            sc = joblib.load('scaler.pkl')
            sc.fit(X)
            X = sc.transform(X)
            return pd.DataFrame(sc.transform(X), columns=df_model.columns)

    def strat_cross_val(self, shuffle=True, **kwargs):
        X, y = self.transform_data()
        strat_kfold = StratifiedKFold().split(X, y)
        y_hat = y.copy()

        for ind_train, ind_test in strat_kfold:
            X_train, X_test = X.iloc[ind_train], X.iloc[ind_test]
            y_train = y.iloc[ind_train]
            clf = self.classif(**kwargs)
            clf.fit(X_train, y_train)
            y_hat.iloc[ind_test] = clf.predict(X_test)
        return y_hat

    def predict_churn(self, X_pred):
        if not self.target:
            return self.class_trained.predict(X_pred)
