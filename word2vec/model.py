from numpy import unique

from keras import backend as k
from keras.layers import Dense,Embedding,Lambda
from keras.models import Model,Sequential

from keras.preprocessing.sequence import pad_sequences

from utilities import model


class Word2Vec(object):
       
    NAME = "Word2Vec"

    def __init__(self):  
                
        self.model = None
        self.history = None
        self.init_settings = model.initialize()
        self.compile_settings = model.compile()
        self.fit_settings = model.fit()

    def initialize(self,vocab_size,window_size):

        model = Sequential()

        model.add(Embedding(input_dim=vocab_size,output_dim=self.init_settings["embedding_size"],input_length=2*window_size))
        model.add(Lambda(lambda X: k.mean(X,axis=1),output_shape=[self.init_settings["embedding_size"]]))
        model.add(Dense(vocab_size,activation=self.init_settings["activation"]))
        
        model.compile(**self.compile_settings)

        print(model.summary())

        self.model = model
    
    def fit(self,context,target):

        self.history = self.model.fit(context,target,batch_size=self.fit_settings["batch_size"],epochs=self.fit_settings["epochs"],verbose=self.fit_settings["verbose"]) 
   
    '''
    def predict(self,context,target,tokenizer):
        pass
    '''
    
    def save_weights(self):
        print('saving model weights to disk....')
        self.model.save_weights("{}/trained/model.hd5".format(Word2Vec.NAME))
        print('completed')

    def load_weights(self):
        print('loading model weights from disk...')
        self.model.load_weights("{}/trained/model.hd5".format(Word2Vec.NAME))
        print('completed')
