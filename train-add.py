from __future__ import print_function, division
from builtins import range

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

from collections import Counter
import random

# read the dataset
data = pd.read_csv('./data/dataset.csv', header=0)
data.columns = ['text', 'dataset', 'fyhao', 'twinword', 'synmato']
# print(data) # test purpose
manual = np.array(data.dataset)
api_1 = np.array(data.fyhao)
api_2 = np.array(data.twinword)
api_3 = np.array(data.synmato)


weight = 0.6
scalar = 0.4

class GloveVectorizer:
  def __init__(self):
    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    embedding = []
    idx2word = []
    with open('./glove.6B/glove.6B.50d.txt') as f:
      # open space-separated text vector dictionary:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        values.extend([0.1, 0.1, 0.1])
        word = values[0]
        vec = np.array(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))

    # initialization
    self.word2vec = word2vec
    self.embedding = np.array(embedding)
    self.word2idx = {v:k for k,v in enumerate(idx2word)}
    self.V, self.D = self.embedding.shape

  def fit(self, data):
    pass

  def transform(self, data):
    # declare a new array with the same length and dimensions of the dataset
    Xmean = np.zeros((len(data), self.D))
    Xsum = np.zeros((len(data), self.D))
    print(len(data), " ", self.D)
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.lower().split()
      vecs = []
      for word in tokens:
        if word in self.word2vec:
          vec = self.word2vec[word]
          vecs.append(vec)
      # add 3 more vectors based on API' results
      # print(vecs)
      if len(vecs) > 0:
        vecs = np.array(vecs)
        # print(vecs) # Test vecs
        Xmean[n] = vecs.mean(axis=0) # get mean along the column to transform the data
        Xsum[n] = vecs.sum(axis=0)
        Xsum[n] *= scalar
        # print(Xsum[n])
        if api_1[n] == "negative\n":
          Xsum[n][50]=1*weight
        elif api_1[n] == "neutral\n":
          Xsum[n][50]=2*weight
        else:
          Xsum[n][50]=3*weight
        if api_2[n] == "negative\n":
          Xsum[n][51]=1*weight
        elif api_2[n] == "neutral\n":
          Xsum[n][51]=2*weight
        else:
          Xsum[n][51]=3*weight
        if api_3[n] == "negative\n":
          Xsum[n][52]=1*weight
        elif api_3[n] == "neutral\n":
          Xsum[n][52]=2*weight
        else:
          Xsum[n][52]=3*weight
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return Xsum

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


# vectorize the dataset
vectorizer = GloveVectorizer()
Xdata = vectorizer.fit_transform(data.text)


# define the number of the clusters
k_list = [3, 5, 7, 9, 11, 13]
sample_list = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 85, 100]

for k in k_list:
  for sample_size in sample_list:
    ratio = float(sample_size/2000)
    # Create a k-means model and fit it to the data
    km = KMeans(n_clusters=k)
    km.fit(Xdata)

    # Predict the clusters for each document
    y_pred = km.predict(Xdata)

    # Print the cluster assignments
    print(y_pred)

    def CompareAccuracy(api):
      sample = []
      arr = []
      # generate random numbers represent index to the data
      for i in range(20):
        randlist_n = random.sample(range(0, elem[i]), round(elem[i]*ratio))
        for num in randlist_n:
          key = elem_n[i][num]
          sample.append(api[key])
          arr.append(manual[key])
      accurate = accuracy(sample, arr)
      return accurate

    # compare two arrays to get accuracy
    def accuracy(sample, arr):
      sample = np.array(sample)
      arr = np.array(arr)
      comm = np.where(sample == arr)[0]
      accurate = len(comm)/len(arr)
      return accurate

    # calculate the number of elements in each cluster
    elem = Counter(y_pred)
    print("\nk-clusters count:\t", elem)
    elem_n = []
    for i in range(20):
      elem_n.append(np.where(y_pred == i)[0])

    def ClusterAccuracy():
      # print("\nk-Clustering Accuracy:\nAPI 1\tAPI 2\tAPI 3")
      # print("{0:.2%}".format(CompareAccuracy(api_1)), "\t", "{0:.2%}".format(CompareAccuracy(api_2)), " ", "{0:.2%}".format(CompareAccuracy(api_3)))
      return [CompareAccuracy(api_1), CompareAccuracy(api_2), CompareAccuracy(api_3)]


    # calculate total accuracy of 3 APIs
    def PopulationAccuracy():
      accuracy1 = accuracy(api_1, manual)
      accuracy2 = accuracy(api_2, manual)
      accuracy3 = accuracy(api_3, manual)
      # print("\nTotal Population Accuracy: \nAPI 1\tAPI 2\tAPI 3")
      # print("{0:.2%}".format(accuracy1), "\t", "{0:.2%}".format(accuracy2), " ", "{0:.2%}".format(accuracy3))
      return [accuracy1, accuracy2, accuracy3]


    # random generate samples and calculate the accuracy
    def RandomSampleAccuracy():
      randomIndex = random.sample(range(0,2000), sample_size)
      randomIndex.sort()
      sample = []
      sample_api_1 = []
      sample_api_2 = []
      sample_api_3 = []
      for num in randomIndex:
        sample.append(manual[num])
        sample_api_1.append(api_1[num])
        sample_api_2.append(api_2[num])
        sample_api_3.append(api_3[num])
      accuracy1 = accuracy(sample_api_1, sample)
      accuracy2 = accuracy(sample_api_2, sample)
      accuracy3 = accuracy(sample_api_3, sample)
      # print("\nRandom Pick Sample Accuracy: \nAPI 1\tAPI 2\tAPI 3")
      # print("{0:.2%}".format(accuracy1), "\t", "{0:.2%}".format(accuracy2), " ", "{0:.2%}".format(accuracy3))
      return [accuracy1, accuracy2, accuracy3]

    RandomSampleAccuracy()

    ClusterAccuracy()

    Xbar = np.array(PopulationAccuracy())
    randomPick = []
    clusterPick = []
    # Standard Deviation List
    randomPickStd = []
    clusterPickStd = []
    # run 1000 times then compare the difference
    for i in range(0, 10000):
      r = RandomSampleAccuracy()
      c = ClusterAccuracy()
      rList = []
      cList = []
      rSqList = []
      cSqList = []
      for x, y in zip(r, Xbar):
        rList.append(abs(x-y))
        rSqList.append(pow(x-y, 2)/1000)
      randomPick.append(rList)
      randomPickStd.append(rSqList)
      for x, y in zip(c, Xbar):
        cList.append(abs(x-y))
        cSqList.append(pow(x-y, 2)/1000)
      clusterPick.append(cList)
      clusterPickStd.append(cSqList)

    randomPick = np.array(randomPick)
    clusterPick = np.array(clusterPick)
    Xrandom = randomPick.mean(axis=0)
    Xcluster = clusterPick.mean(axis=0)
    with open('text-4.txt', 'a') as f:
      f.write('\nSample size: ')
      f.write(str(sample_size))
      f.write('\nk = ')
      f.write(str(k))
      f.write('\nRandom Pick Deviation with: ')
      f.write(str(Xrandom))
      f.write('\nk-Clustering Pick Deviation with: ')
      f.write(str(Xcluster))

    ############ standard deviation###################
    randomPickStd = np.array(randomPickStd)
    clusterPickStd = np.array(clusterPickStd)
    Drandom = np.sqrt(randomPickStd.sum(axis=0))
    Dcluster = np.sqrt(clusterPickStd.sum(axis=0))
    print("\nRandom Pick Standard Deviation with 53 vectors: ", Drandom)
    print("k-Clustering Pick Standard Deviation with 53 vectors: ", Dcluster)