import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import ParametrosEfectividad

# load dataset from csv file
data = pd.read_csv("D:/Proyecto/datosfinal.csv")

# One-hot encode the data using pandas get_dummies
data = pd.get_dummies(data)
# set the dependent variable
labels = np.array(data['Actividad'])
# Remove the labels from the features
data = data.drop('Actividad', axis=1)
# Saving feature names for later use
data_list = list(data.columns)
# Convert to numpy array
data = np.array(data)

# split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
# create the model
model = SVC(kernel='linear', gamma='auto')

# set the number of folds
folds = range(1, 10)

# train the model with folds
for j in folds:
    print('\nConstruyendo Fold ', j)
    print('\nDatos de prueba', x_test, y_test)
    # fit model
    model.fit(x_train, y_train)
    # test the model
    y_pred = model.predict(x_test)
    # Calculate the absolute errors
    print(f'\nPrecisión del Modelo: {model.score(x_test, y_test)}')
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

print("\n\nConstruyendo el modelo final...")
y_pred = model.predict(x_test)
# Calculate the absolute errors
print(f'Precisión del Modelo: {model.score(x_test, y_test)}')
# create the confusion matrix
print("\n\nMatriz de Confusión:\n")
mc = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(mc)

print("\n\nParámetros de Efectividad:\n")
ParametrosEfectividad.print_stats(mc)