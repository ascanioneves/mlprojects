from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Reading diabetes dataset ...')
data = pd.read_csv('diabetes_dataset.csv')

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = data[feature_cols]
y = data.Outcome
#print(y)

#Splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Combining X_test and y_test with the purpose of dropping test set missing data
combined = pd.concat([X_test, y_test], axis=1)

#Dropping missing values of test set
dropped_test = combined.dropna(axis=0, how='any')
X_test = dropped_test[feature_cols]
y_test = dropped_test.Outcome

#Combining X_train and y_train with the pyrpose of dropping train set missing data
combined = pd.concat([X_train, y_train], axis=1)
dropped_train = combined.dropna(axis=0, how='any')
X_train = dropped_train[feature_cols]
y_train = dropped_train.Outcome
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#Make predictions
y_pred = neigh.predict(X_test)

#Accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

