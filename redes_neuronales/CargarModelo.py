from keras.models import model_from_json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#load dataset from csv file
data = pd.read_csv("D:/Proyecto/datosfinal.csv", index_col=0)
y = tf.keras.utils.to_categorical(data.index)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(data.values, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))