import io
import re
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
def splitsentence(file_path):
    input = []
    csv_reader = io.open(file_path, encoding='utf-8').read().strip().split('\n')
    for row in csv_reader:
        row = re.sub(r"=[\s\S]*=", "", row)
        row = re.sub(r"[^a-zA-Z?.!,¿]+", " ", row)
        row = re.sub(r"[ ]+", " ", row)
        row=re.split(r"[?.!¿]",row)
        for line in row:
            if line != " " and line != "":
                print(line)
                input.append(line)
    return input
def tokenize(input):
    tokenizer=Tokenizer(filters='')
    tokenizer.fit_on_texts(input)
    tensor=tokenizer.texts_to_sequences(input)
    tensor=pad_sequences(tensor,padding='post')
    return tensor,tokenizer
path="test.csv"
test=splitsentence(path)
tensor,tokenizer=tokenize(test)
text=np.array(tensor)
word=tf.data.Dataset.from_tensor_slices(text)
print(word)
def split_input_target(chunk):
    input_text=chunk[:-1]
    target_text=chunk[1:]
    return input_text,target_text
dataset=word.map(split_input_target)
print(dataset)
BATCH_SIZE=16
BUFFER_SIZE=10000
dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
print(dataset)
total_words=len(tokenizer.word_index)+1
embedding_dim=256
rnn_units=1024
model=tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words,embedding_dim,batch_input_shape=[BATCH_SIZE,None]),
    tf.keras.layers.GRU(rnn_units,
                 return_sequences=True),
    tf.keras.layers.Dense(total_words)
])
for input_example_batch,target_example_batch in dataset.take(1):
    example_batch_prediction=model(input_example_batch)
    print(example_batch_prediction.shape,"#(batch_size,sequence_length,vocab_size)")

example_batch_loss=tf.keras.losses.sparse_categorical_crossentropy(target_example_batch,example_batch_prediction,from_logits=True)
print("Prediction shape: ", example_batch_prediction.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam',loss=tf.keras.losses.sparse_categorical_crossentropy)
checkpoint_dir='./traning_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,"ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)
Epoch=10
history=model.fit(dataset,epochs=Epoch,callbacks=[checkpoint_callback])
