import os
import sys
import logging
import time
import gensim
import numpy as np



def get_embedding_vector(sentences):
    print("loading word2vec model now...........")
    model=gensim.models.KeyedVectors.load_word2vec_format("D:\\Emotion_on_Comments\\data\\embedding_64.bin",binary=True)
    print("loading word2vec finished")
    all_sample_vector_lists=[]
    padding_embedding=np.array([0] * 64,dtype=np.float32)
    print("transform word to vector now.......")
    for sentence in sentences:
        sentence_vector = []
        for word in sentence:
            if word in model.vocab:
                sentence_vector.append(model[word])
            else:
                sentence_vector.append(padding_embedding)
        all_sample_vector_lists.append(sentence_vector)
    print("transform word to vector finished")
    return all_sample_vector_lists