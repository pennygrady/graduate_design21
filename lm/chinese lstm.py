import tensorflow as tf
import jieba
def get_chinese_indexlist(text):
    seg_list=jieba.cut(text,cut_all=True)
    word_index={}
    index_word={}
    temp=1
    for i in seg_list:
        if word_index.get(i,"NONE")!="NONE":
            continue
        word_index[i]=temp
        index_word[temp]=i
        temp=temp+1
    return word_index,index_word

def text_to_index(text,word_index):
    seg_list=jieba.cut(text,cut_all=True)
    temp=[]
    for i in seg_list:
        temp.append(word_index[i])
    return temp

def pad_sequence(sequence):
    lens=[len(seq) for seq in sequence]
    maxlen=max(lens)
    pad_seq=(seq+[0]*(maxlen-len(seq)) for seq in sequence)
    return pad_seq

word =["十五的月亮，月亮的十五"]
word_index,index_word=get_chinese_indexlist(word[0])
print(word_index)
for change in word:
    change=text_to_index(change,word_index)
    print(change)