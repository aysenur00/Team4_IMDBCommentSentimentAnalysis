#https://blog.tensorflow.org/2020/12/making-bert-easier-with-preprocessing-models-from-tensorflow-hub.html
#imported bert encoder and preprocessor from link
import tensorflow_hub as hub
import tensorflow as tf
from initDataset import init_dataset

# Path to the saved model
path = "my_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(
       (path),
       custom_objects={'KerasLayer':hub.KerasLayer},
       compile=False
)

# Define evaluation metrics
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

# Initialize dataset
X_train, X_test, Y_train, Y_test = init_dataset()

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=METRICS)

# Evaluate the model on the test dataset
model.evaluate(X_test, Y_test, verbose=2)


