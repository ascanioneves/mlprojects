"""
Atividade para trabalhar o pré-processamento dos dados.
Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.
@author: Aydano Machado <aydano.machado@gmail.com>
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

print('\n - Lendo o arquivo com o dataset sobre diabetes')

print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')


data = pd.read_csv('abalone_dataset.csv')
#high correlation with diameter
data = data.drop('length', axis=1)
# Remove all rows that have missing data (initial test, just to see the model behavior)
feature_cols = ['sex','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']

#char -> numbers
data['sex'] = data['sex'].map({'M': 0, 'F': 1, 'I': 2})
X = data[feature_cols]
y = data.type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(' - Criando modelo preditivo')

#mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', max_iter=1500)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)


y_pred = clf.predict(X_test)

a = accuracy_score(y_test, y_pred)
print(f'Accuracy {a}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix: {conf_matrix}')

precision_micro = precision_score(y_test, y_pred, average='micro')
print(precision_micro)


recall_micro = recall_score(y_test, y_pred, average='micro')
print(recall_micro)

f1_micro = f1_score(y_test, y_pred, average='micro')
print(f1_micro)

