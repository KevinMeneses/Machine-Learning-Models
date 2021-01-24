import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from utiles import ParametrosEfectividad

data = pd.read_csv("D:/Proyecto/DatasetsFinales/DatosEval.csv", index_col=0)

X = data.values
# set the dependent variable
y = data.index.values

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)

# create the model
model = SVC(kernel='linear', gamma='auto')

y_pred = []
y_test = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    print('\nConstruyendo Fold ')
    # fit model
    model.fit(X_train, y_train)
    # test the model
    y_pred = model.predict(X_test)
    # Calculate the absolute errors
    print(f'\nPrecisión del Modelo: {model.score(X_test, y_test)}')
    mc = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("\nParámetros de Efectividad:\n")
    ParametrosEfectividad.print_stats(mc)