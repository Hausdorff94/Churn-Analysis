import pandas as pd
import joblib

from churn_modelling import *

data_test = pd.read_excel('.data\Test sample No Label.xlsx')
clf = joblib.load('gbc_model.pkl')
X_pred = modelling(df=data_test, target=False).transform_data()


print(clf.predict(X_pred))
