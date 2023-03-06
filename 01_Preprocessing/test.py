import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

print('\nReading diabetes dataset ...')
data = pd.read_csv('diabetes_dataset.csv')

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#missing_features = ['Glucose', 'BloodPressure', 'SkinThickness']
#correlated_features = ['BMI', 'Age', 'Pregnancies']

X = data[feature_cols]
y = data.Outcome
acc = 0

print("Generating 5 different results ...")
for i in range(5):

    #Splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    #complete_data = df.dropna(subset=missing_features + correlated_features)
    #missing_data = df.loc[df.index.difference(complete_data.index), :]

    #X_t = complete_data[correlated_features]
    #y_t = complete_data[missing_features]
    #regressor = LinearRegression().fit(X_train, y_train)

    #X_te = missing_data[correlated_features]
    #y_p = regressor.predict(X_te)

    df = pd.concat([X_train, y_train], axis=1)

    
    #missing_data[missing_features] = y_p
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
    imputer = SimpleImputer(strategy='median')
    
    imputer.fit(df)
    imputed_data = imputer.transform(df)
    df = pd.DataFrame(imputed_data, columns=df.columns)
    X_train = df[feature_cols]
    y_train = df.Outcome

    column = X_train['Insulin']

    scaler = MinMaxScaler()
    column_norm = scaler.fit_transform(column.values.reshape(-1, 1))
    X_train['Insulin'] = column_norm

    #print(X_train['Insulin'])

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
obj = ' (using median strategy)'

with open(log, "w") as file:
    file.write(new_content + obj)

print(f'Mean of accuracy: {acc/5}')
