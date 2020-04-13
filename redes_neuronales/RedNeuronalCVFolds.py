import pandas as pd
import tensorflow as tf
import numpy as np
from redes_neuronales.RedesNeuronales import create_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# load dataset from csv file
data = pd.read_csv("D:/Proyecto/datosfinal.csv", index_col=0)
# set the dependent variable
y = tf.keras.utils.to_categorical(data.index)

# create the model
model = create_model(20)
# split the data
x_train, x_test, y_train, y_test = train_test_split(data.values, y, test_size=0.25)

# set the number of folds
folds = range(1, 10)

# train the model with folds
for j in folds:
    print('\nFold ', j)
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, test_size=0.25)
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    print("\nTraining model...")
    model.fit(x=x_train, y=y_train, epochs=4)
    print("\nEvaluating model...")
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

# test the model
y_pred = model.predict(x_test)
# create the confusion matrix
prediction = np.argmax(y_pred, axis=1)
true = np.argmax(y_test, axis=1)
print(confusion_matrix(y_true=true, y_pred=prediction))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")