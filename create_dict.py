from Preprocess import Preprocess
import tensorflow as tf
from tensorflow.keras import preprocessing
import pickle
import pandas as pd

movie_review = pd.read_csv('C:/tokenizer/영화리뷰.csv')
purpose = pd.read_csv('C:/tokenizer/용도별목적대화데이터.csv')
topic =  pd.read_csv('C:/tokenizer/주제별일상대화데이터.csv')
common_sence =  pd.read_csv('C:/tokenizer/일반상식.csv')

movie_review.dropna(inplace=True)
purpose.dropna(inplace=True)
topic.dropna(inplace=True)
common_sence.dropna(inplace=True)

text1 = list(movie_review['document'])
text2 = list(purpose['text'])
text3 = list(topic['text'])
text4 = list(common_sence['query']) + list(common_sence['answer'])

corpus_data = text1 + text2 + text3 + text4

p = Preprocess()
dict = []
for c in corpus_data:
    pos = p.pos(c)
    for k in pos:
        dict.append(k[0])

tokenizer = preprocessing.text.Tokenizer(oov_token='OOV', num_words=100000)
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index
print(len(word_index))

f = open("C:/tokenizer/chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()