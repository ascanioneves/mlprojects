import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print('\nReading diabetes dataset ...')
data = pd.read_csv('diabetes_dataset.csv')

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = data[feature_cols]
y = data.Outcome
acc = 0

print("Generating 5 different results ...")
for i in range(5):

    #Splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #Combining X_test and y_test with the purpose of dropping test set missing data
    combined = pd.concat([X_test, y_test], axis=1)

    #Dropping missing values of test set
    dropped_test = combined.dropna(axis=0, how='any')
    X_test = dropped_test[feature_cols]
    y_test = dropped_test.Outcome

    #Combining X_train and y_train with the pyrpose of dropping train set missing data
    #combined = pd.concat([X_train, y_train], axis=1)
    #dropped_train = combined.dropna(axis=0, how='any')
    #X_train = dropped_train[feature_cols]
    #y_train = dropped_train.Outcome
    imputer = SimpleImputer(strategy='mean')
    df = pd.concat([X_train, y_train], axis=1)
    imputer.fit(df)
    imputed_data = imputer.transform(df)
    df = pd.DataFrame(imputed_data, columns=df.columns)
    X_train = df[feature_cols]
    y_train = df.Outcome
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    #Make predictions
    y_pred = neigh.predict(X_test)

    #Accuracy
    a = accuracy_score(y_test, y_pred)
    print(f'Accuracy [{i}]: {a}')
    acc += a

log = './log.txt'
with open(log, "r") as file:
    content = file.read()

new_content = content + '\n' + str(acc/5)
obj = ' (using mean strategy)'

with open(log, "w") as file:
    file.write(new_content + obj)

print(f'Mean of accuracy: {acc/5}')
