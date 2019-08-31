import matplotlib.pyplot as plt 

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

import numpy as np

from utilities import data

def word_embeddings(cbow):

    embeddings = cbow.model.get_weights()[0]
    embeddings = embeddings[2:]

    labels = data.tokenizer.word_index.keys() 

    distance_matrix = euclidean_distances(embeddings)

    tnse = TSNE(n_components=2,random_state=0,n_iter=10000,perplexity=50)
    
    np.set_printoptions(suppress=True)
    
    T = tnse.fit_transform(distance_matrix)
    
    plt.figure(figsize=(20,10))
    plt.scatter(T[:,0],T[:,1])
    
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        
        plt.annotate(label, xy=(x, y))
    
    plt.show()