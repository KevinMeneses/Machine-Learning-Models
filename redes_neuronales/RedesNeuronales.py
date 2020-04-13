import tensorflow as tf


def create_model(output):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(71, activation=tf.nn.relu, input_dim=(9120, 74)))
    model.add(tf.keras.layers.Dense(71, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(70, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(output, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def show_gpu_list():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
