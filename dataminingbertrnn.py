import tensorflow_hub as hub
import tensorflow_text as tf_text
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import LSTM,Dense,Bidirectional,Input
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tokenizers import BertWordPieceTokenizer
import numpy as np
from initDataset import init_dataset

nltk.download("stopwords")
nltk.download('punkt')
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased' , lower = True)
tokenizer.save_pretrained('.')
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)

def encode(texts, tokenizer, chunk_size=256, max_length=512):
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length)

    all_ids = []
    all_masks = []
    all_types = []

    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
        all_masks.extend([enc.attention_mask for enc in encs])
        all_types.extend([enc.type_ids for enc in encs])

    return np.array(all_ids), np.array(all_masks), np.array(all_types)

X_train, X_test, Y_train, Y_test = init_dataset()

X_train_enc, X_train_mask, X_train_type = encode(X_train, fast_tokenizer)
X_test_enc, X_test_mask, X_test_type = encode(X_test, fast_tokenizer)

preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')

input_word_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_mask")
input_type_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_type_ids")

bert_inputs = {
    'input_word_ids': input_word_ids,
    'input_mask': input_mask,
    'input_type_ids': input_type_ids
}

encoder_output = encoder(bert_inputs)
pooled_output = encoder_output['pooled_output']

# RNN (LSTM) katmanı
lstm_output = tf.keras.layers.LSTM(128)(tf.expand_dims(pooled_output, axis=1))
dropout_layer = tf.keras.layers.Dropout(0.1, name='dropout')(lstm_output)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dropout_layer)

model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[output_layer])

# Model özeti
model.summary()

# Değerlendirme metrikleri
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

# Modeli derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=METRICS)

model.fit(
    [X_train_enc, X_train_mask, X_train_type],
    Y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_test_enc, X_test_mask, X_test_type], Y_test)
)