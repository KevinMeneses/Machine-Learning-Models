import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

model = RandomForestClassifier(
    random_state=1,  # semilla inicial de aleatoriedad del algoritmo
    n_estimators=1000,  # cantidad de arboles a crear
    min_samples_split=2,  # cantidad minima de observaciones para dividir un nodo
    min_samples_leaf=1,  # observaciones minimas que puede tener una hoja del arbol
    n_jobs=-1  # tareas en paralelo. para todos los cores disponibles usar -1
)

# set the number of folds
folds = range(1, 10)

# train the model with folds
for j in folds:
    print('\nFold ', j)
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
    # fit model
    model.fit(x_train, y_train)
    # test the model
    y_pred = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    print('Error:', round(100 - accuracy, 2), '%.')

y_pred = model.predict(x_test)
# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print('Error:', round(100 - accuracy, 2), '%.')
# create the confusion matrix
print(confusion_matrix(y_true=y_test, y_pred=y_pred))