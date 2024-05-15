#https://blog.tensorflow.org/2020/12/making-bert-easier-with-preprocessing-models-from-tensorflow-hub.html
#imported bert encoder and preprocessor from link
import tensorflow_hub as hub
import tensorflow_text as tf_text
import tensorflow as tf
from initDataset import init_dataset
preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')

# Cümleler
sentences = [
    "it is a good and excellent movie",
    "this was a terrible movie"
]

# TensorFlow Keras modeli tanımı
input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
preprocess_layer = preprocessor(input_layer)
encoded_output = encoder(preprocess_layer)
output_layer = tf.keras.layers.Dropout(0.1, name='dropout')(encoded_output['pooled_output'])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(output_layer)

# Model oluşturma
model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

# Model özetini yazdırma
print(model.summary())

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
X_train, X_test, Y_train, Y_test = init_dataset()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

model.fit(X_train, Y_train, epochs=10)