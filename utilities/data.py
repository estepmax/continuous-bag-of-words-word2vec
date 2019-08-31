from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np

import string 

tokenizer = Tokenizer(lower=True)

def load_corpus(filename):
    
    file = open("./utilities/{}".format(filename),"r")
    corpus = file.read()
    file.close()
    
    return corpus

def load_dataset():
    return np.load('./utilities/data.npz')

def create_dataset(corpus,sep,window_size,save=True):
    
    sentences = corpus.split(sep)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences) 
    vocab_size = len(tokenizer.word_index) + 1

    context = list()
    target = list()

    for j in range(len(sequences)):
    
        words = sequences[j] 
        k = len(words)
    
        for i in range(k):
        
            m = max(0,i-window_size)
            n = min(window_size+i,k-1)
        
            content = words[m:n+1]
            content.remove(words[i])

            context.append(content)
            target.append(words[i])
    
    context = pad_sequences(context,maxlen=2*window_size,padding='post')
    target = to_categorical(target,num_classes=vocab_size)

    if save:
        print('Saving generated context/target vectors to disk...')
        np.savez('./utilities/data',context=context,target=target)

    return target,context,vocab_size

