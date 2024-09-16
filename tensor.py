import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
x_train = x_train_raw / 255.0
x_test = x_test_raw / 255.0

model_lr = tf.keras.models.Sequential([
    layers.Input(shape=x_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
model_lr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_lr = model_lr.fit(
    x_train, y_train_raw,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test_raw),
    verbose=1
)

test_loss, test_acc = model_lr.evaluate(x_test, y_test_raw)
print(f"Logistic Regression Test accuracy: {test_acc:.4f}")

probs = model_lr.predict(x_test[:5])
predictions = np.argmax(probs, axis=1)
for i in range(5):
    plt.imshow(x_test[i], cmap="Greys")
    plt.title(f"Predicted: {predictions[i]}")
    plt.show()

model_mlp = tf.keras.models.Sequential([
    layers.Input(shape=x_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(10, activation='softmax')
])
model_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_mlp = model_mlp.fit(
    x_train, y_train_raw,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test_raw),
    verbose=1
)

test_loss_mlp, test_acc_mlp = model_mlp.evaluate(x_test, y_test_raw)
print(f"MLP Test accuracy: {test_acc_mlp:.4f}")
