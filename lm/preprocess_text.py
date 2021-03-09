import io
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
def splitsentence(file_path):
    input = []
    csv_reader = io.open(file_path, encoding='utf-8').read().strip().split('\n')
    for row in csv_reader:
        row = re.sub(r"=[\s\S]*=", "", row)
        row = re.sub(r"[^a-zA-Z?.!,¿]+", " ", row)
        row = re.sub(r"[ ]+", " ", row)
        generate = row.split('.')
        for line in generate:
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
print(tokenizer.word_index)
