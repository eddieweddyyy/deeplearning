import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFAutoModel
import re
import itertools
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_model = TFAutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True, from_pt=True)

raw = pd.read_csv('/Users/edwardjang/Desktop/deeplearning/labeled_data.csv')
raw.drop_duplicates(subset=['tweet'], inplace=True)
raw['tweet'] = raw['tweet'].apply(lambda x: ''.join(ch for ch, _ in itertools.groupby(x)))

print(raw.shape)

def remove_unnecessaries(data):
    text = re.sub(r"[^\w\s]", '', data)
    text = text.replace("RT", '')
    return text

def remove_url(data):
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", '', data)
    return text

def remove_unicode(data):
    text = re.sub(r"/(&.+;)/ig", '', data)
    text = re.sub(r"@([^ ]+)", '', text)
    text = re.sub(r"/(\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])/g", '', text)
    return text

def clean_data(data):
    text = remove_url(data)
    text = remove_unnecessaries(text)
    text = remove_unicode(text)
    return text

raw['tweet'] = raw['tweet'].apply(clean_data)

def encode_tweets(data, max_len=200):
    encoded = tokenizer.batch_encode_plus(
        data,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    return encoded


encoded_tweets = encode_tweets(raw['tweet'].tolist(), max_len=200)
print(encoded_tweets)

input_ids = encoded_tweets['input_ids'].numpy()
attention_masks = encoded_tweets['attention_mask'].numpy()

X_train, X_val, y_train, y_val = train_test_split(
    input_ids,
    raw['class'].values,
    test_size=0.2,
    random_state=42
)
train_masks, val_masks = train_test_split(
    attention_masks,
    test_size=0.2,
    random_state=42
)

input_ids_layer = tf.keras.Input(shape=(200,), dtype=tf.int32, name="input_ids")
attention_mask_layer = tf.keras.Input(shape=(200,), dtype=tf.int32, name="attention_mask")

def get_bert_embeddings(inputs):
    input_ids, attention_mask = inputs
    output = bert_model(input_ids, attention_mask=attention_mask)
    return output.last_hidden_state

emb_output = tf.keras.layers.Lambda(get_bert_embeddings, output_shape=(200, 768))([input_ids_layer, attention_mask_layer])

x = tf.keras.layers.LSTM(64, return_sequences=False)(emb_output)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

new_output = tf.keras.layers.Dense(3, activation='softmax')(x)

classifier_model = tf.keras.Model(inputs=[input_ids_layer, attention_mask_layer], outputs=new_output)

classifier_model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = classifier_model.fit(
    [X_train, train_masks],
    y_train,
    validation_data=([X_val, val_masks], y_val),
    epochs=3,
    batch_size=16
)

loss, accuracy = classifier_model.evaluate([X_val, val_masks], y_val)
print(f"Validation Accuracy: {accuracy}")
print(raw['tweet'])
