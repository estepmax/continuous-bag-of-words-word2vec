from word2vec.model import Word2Vec
from utilities import data,plot

def main():
   
   window_size = 2
   corpus = data.load_corpus('stay_postmalone.txt')
   target,context,vocab_size = data.create_dataset(corpus=corpus,sep='\n',window_size=window_size,save=False)

   model = Word2Vec()
   model.initialize(vocab_size,window_size)
   model.fit(context,target)

   plot.word_embeddings(model)
   
if __name__=='__main__':
    main()
    
