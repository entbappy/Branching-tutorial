import tensorflow as tf
import numpy as np
import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import mlflow.keras


#Lets now load the data
f_MNIST = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = f_MNIST.load_data()

# Creating a validatiion set from X_train_full and Scaling the data by 255. Cz, it's an uint8 data
X_valid, X_train = X_train_full[:5000] / 255. , X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# Scaling the test set as well
X_test = X_test / 255.

# Lets create the model
LAYERS = [
          tf.keras.layers.Flatten(input_shape = [28,28], name="InputLayer" ),
          tf.keras.layers.Dense(300, activation="relu", name="HiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name = "Hiddenlayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name ="OutputLater")
]

model = tf.keras.models.Sequential(LAYERS)
model.summary()

# Compiling model
LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "Adam"
METRICS = ["accuracy"]

model.compile(loss=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS)

EPOCHS = 10
VALIDATION_SET = (X_valid, y_valid)

mlflow.tensorflow.autolog()
with mlflow.start_run():
    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_SET, batch_size=256)
    preds = model.predict(X_test)
    preds = np.round(preds)

    eval_acc = model.evaluate(X_test, y_test)[1]
    print("eval_auc",eval_acc)

