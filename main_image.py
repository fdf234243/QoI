import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

from collections import Counter
import random

import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby

# to compare against other 2D images. Using Resnet to extract the compressed representation.
model = models.resnet18(pretrained='imagenet')
content_pd = pd.read_csv('./all_responses.csv', header = 0)
content_pd.columns = ['image1', 'image2', 'groundTruth', 'aws', 'comp', 'face']
gTruth = np.array(content_pd.groundTruth)
aws = np.array(content_pd.aws)
comp = np.array(content_pd.comp)
face = np.array(content_pd.face)


#Resize the image to 224x224 px
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

def extract_feature_vector(img):
    img = Image.open(img)
    # 2. Create a PyTorch Variable with the transformed image
    #Unsqueeze actually converts to a tensor by adding the number of images as another dimension.
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1, 512, 1, 1)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding.squeeze().numpy()

def get_cosine_distance(im_url1, im_url2):
    im1, im2 = str(im_url1), str(im_url2)
    image1 = extract_feature_vector(im1).reshape(1, -1)
    image2 = extract_feature_vector(im2).reshape(1, -1)
    return cosine_similarity(image1, image2)


ids1 =  content_pd.image1
ids2 =  content_pd.image2
# print(ids1)
folder1 = ids1.str[:5]
folder2 = ids2.str[:5]

# create images' reletive path
urls1 = "./images/" + folder1 + "/" + ids1
urls2 = "./images/" + folder2 + "/" + ids2

Xdata1 = np.empty([len(ids1), 512])
Xdata2 = np.empty([len(ids2), 512])
n = 0
for (img1, img2) in zip(urls1, urls2):
#    print('Distance between 2 images :', get_cosine_distance(img1, img2))
    Xdata1[n] = extract_feature_vector(img1).reshape(1, -1)
    Xdata2[n] = extract_feature_vector(img2).reshape(1, -1)
    n += 1
    # print(image1, image2)

Xdata = np.column_stack((Xdata1, Xdata2))
# print(Xdata.ndim)
# # print(Xdata)
# # Create a k-means model and fit it to the data
# km = KMeans(n_clusters=5)
# km.fit(Xdata)

# # Predict the clusters for each document
# y_pred = km.predict(Xdata)

# # Print the cluster assignments
# print(y_pred)


# define the number of the clusters
k_list = [3, 5, 7, 9, 11, 13]
sample_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 85, 100]


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
          arr.append(gTruth[key])
      accurate = accuracy(sample, arr)
      return accurate

    # compare two arrays to get accuracy
    def accuracy(sample, arr):
      sample = np.array(sample)
      arr = np.array(arr)
      # print(sample)
      # print(arr)
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
      return [CompareAccuracy(aws), CompareAccuracy(comp), CompareAccuracy(face)]


    # calculate total accuracy of 3 APIs
    def PopulationAccuracy():
      accuracy1 = accuracy(aws, gTruth)
      accuracy2 = accuracy(comp, gTruth)
      accuracy3 = accuracy(face, gTruth)
      # print("\nTotal Population Accuracy: \nAPI 1\tAPI 2\tAPI 3")
      # print("{0:.2%}".format(accuracy1), "\t", "{0:.2%}".format(accuracy2), " ", "{0:.2%}".format(accuracy3))
      return [accuracy1, accuracy2, accuracy3]


    # random generate samples and calculate the accuracy
    def RandomSampleAccuracy():
      randomIndex = random.sample(range(0,2000), sample_size)
      randomIndex.sort()
      sample = []
      sample_aws = []
      sample_comp = []
      sample_face = []
      for num in randomIndex:
        sample.append(gTruth[num])
        sample_aws.append(aws[num])
        sample_comp.append(comp[num])
        sample_face.append(face[num])
      accuracy1 = accuracy(sample_aws, sample)
      accuracy2 = accuracy(sample_comp, sample)
      accuracy3 = accuracy(sample_face, sample)
      # print("\nRandom Pick Sample Accuracy: \nAPI 1\tAPI 2\tAPI 3")
      # print("{0:.2%}".format(accuracy1), "\t", "{0:.2%}".format(accuracy2), " ", "{0:.2%}".format(accuracy3))
      return [accuracy1, accuracy2, accuracy3]

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
    randomPickT = randomPick.T
    randPT_1 = np.array(randomPickT[0, 0:])
    randPT_2 = np.array(randomPickT[1, 0:])
    randPT_3 = np.array(randomPickT[2, 0:])
    clusterPick = np.array(clusterPick)
    clusterPickT = clusterPick.T
    clustPT_1 = np.array(clusterPickT[0, 0:])
    clustPT_2 = np.array(clusterPickT[1, 0:])
    clustPT_3 = np.array(clusterPickT[2, 0:])

    RP_freq_1 = {key:len(list(group)) for key, group in groupby(np.sort(randPT_1))}
    RP_freq_2 = {key:len(list(group)) for key, group in groupby(np.sort(randPT_2))}
    RP_freq_3 = {key:len(list(group)) for key, group in groupby(np.sort(randPT_3))}
    CP_freq_1 = {key:len(list(group)) for key, group in groupby(np.sort(clustPT_1))}
    CP_freq_2 = {key:len(list(group)) for key, group in groupby(np.sort(clustPT_2))}
    CP_freq_3 = {key:len(list(group)) for key, group in groupby(np.sort(clustPT_3))}
    print("\n##########################################################################")
    print("Random Pick for AWS, sample size = ", sample_size)
    mylist_1 = [key for key, val in RP_freq_1.items() for _ in range(val)]
    title_1 = "AWS_Rand_S" + str(sample_size)
    plot_1 = title_1 + ".png"
    plt.hist(mylist_1, bins=20, color='red')
    # print(len(mylist_1))
    plt.title(title_1)
    plt.savefig(plot_1)
    # plt.show()
    print("\n##########################################################################")
    print("Random Pick for Compreface, sample size = ", sample_size)
    mylist_2 = [key for key, val in RP_freq_2.items() for _ in range(val)]
    plt.hist(mylist_2, bins=20, color='red')
    title_2 = "Comp_Rand_S" + str(sample_size)
    plot_2 = title_2 + ".png"
    plt.title(title_2)
    plt.savefig(plot_2)
    # plt.show()
    print("\n##########################################################################")
    print("Random Pick for PresentID, sample size = ", sample_size)
    mylist_3 = [key for key, val in RP_freq_3.items() for _ in range(val)]
    plt.hist(mylist_3, bins=20, color='red')
    print(len(mylist_3))
    title_3 = "Present_Rand_S" + str(sample_size)
    plot_3 = title_3 + ".png"
    plt.title(title_3)
    plt.savefig(plot_3)
    # plt.show()
    print("\n##########################################################################")
    print("Clustering Pick for AWS, k = ", k, ", sample size = ", sample_size)
    mylist_4 = [key for key, val in CP_freq_1.items() for _ in range(val)]
    plt.hist(mylist_4, bins=20, color='red')
    title_4 = "AWS_Cluster_S" + str(sample_size) + "_k" + str(k)
    plot_4 = title_4 + ".png"
    print(len(mylist_4))
    plt.title(title_4)
    plt.savefig(plot_4)
    # plt.show()
    print("\n##########################################################################")
    print("Clustering Pick for Compreface, k = ", k, ", sample size = ", sample_size)
    mylist_5 = [key for key, val in CP_freq_2.items() for _ in range(val)]
    plt.hist(mylist_5, bins=20, color='red')
    title_5 = "Comp_Cluster_S" + str(sample_size) + "_k" + str(k)
    plot_5 = title_5 + ".png"
    plt.title(title_5)
    plt.savefig(plot_5)
    # plt.show()
    print("\n##########################################################################")
    print("Clustering Pick for PresentID, k = ", k, ", sample size = ", sample_size)
    mylist_6 = [key for key, val in CP_freq_3.items() for _ in range(val)]
    plt.hist(mylist_6, bins=20, color='red')
    title_6 = "Present_Cluster_S" + str(sample_size) + "_k" + str(k)
    plot_6 = title_6 + ".png"
    plt.title(title_6)
    plt.savefig(plot_6)
    # plt.show()  

    # Xrandom = randomPick.mean(axis=0)
    # Xcluster = clusterPick.mean(axis=0)
    # with open('text.txt', 'a') as f:
    #   f.write('\nSample size: ')
    #   f.write(str(sample_size))
    #   f.write('\nk = ')
    #   f.write(str(k))
    #   f.write('\nRandom Pick Deviation with: ')
    #   f.write(str(Xrandom))
    #   f.write('\nk-Clustering Pick Deviation with: ')
    #   f.write(str(Xcluster))

    ############ standard deviation###################
    # randomPickStd = np.array(randomPickStd)
    # clusterPickStd = np.array(clusterPickStd)
    # Drandom = np.sqrt(randomPickStd.sum(axis=0))
    # Dcluster = np.sqrt(clusterPickStd.sum(axis=0))
    # print("\nRandom Pick Standard Deviation: ", Drandom)
    # print("k-Clustering Pick Standard Deviation", Dcluster)      


