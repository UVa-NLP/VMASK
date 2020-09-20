import numpy as np
from gensim.models import KeyedVectors


def getVectors(args, wordvocab):
    vectors = []
    if args.mode != 'rand':
        word2vec = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(wordvocab)):
            word = wordvocab[i]
            if word in word2vec.vocab:
                vectors.append(word2vec[word])
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, args.embed_dim))
    else:
        for i in range(len(wordvocab)):
            vectors.append(np.random.uniform(-0.01, 0.01, args.embed_dim))
    return np.array(vectors)
