import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from redes_neuronales.NeuralNetwork import create_model
from utiles import ParametrosEfectividad

data = pd.read_csv("D:/Proyecto/DatasetsFinales/DatosEval.csv", index_col=0)

X = data.values
# set the dependent variable
y = data.index.values

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)

# create the model
model = create_model(20)

X_test = []
y_test = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = tf.keras.utils.to_categorical(y[train_index]), tf.keras.utils.to_categorical(y[test_index])

    print('\nConstruyendo Fold ')
    # fit model
    model.fit(x=X_train, y=y_train, epochs=5)
    # test the model
    print("\nConstruyendo el modelo final...")
    y_pred = model.predict(X_test)
    # test the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    # Calculate the absolute errors
    print(f'\nPrecisión del Modelo:')
    print(val_acc)
    prediction = np.argmax(y_pred, axis=1)
    true = np.argmax(y_test, axis=1)
    mc = confusion_matrix(true, prediction)
    print("\nParámetros de Efectividad:\n")
    ParametrosEfectividad.print_stats(mc)