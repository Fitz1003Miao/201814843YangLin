import os
import utils, vsm
import numpy as np
import numexpr as ne
import collections
import time
import sklearn.neighbors
from sklearn import preprocessing

import sys
class Logger(object):
    def __init__(self, filename = "logout.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(filename = "logout-3.log")

def calcEuDistance(v1, v2):
    d = np.sum(np.square(v1 - v2), axis = 1)
    return d

def calcCosDistance(v1, v2):
    d = -1 * v1.dot(np.transpose(v2)) / (np.sqrt(v1.dot(np.transpose(v1))) * np.sqrt(np.sum(v2.dot(np.transpose(v2)), axis = 0)))
    return d.reshape(-1, )

if __name__ == "__main__":

    word_frequency = 15
    document_frequency = 5

    tf_path = "../data/tf_%d_%d.txt" % (word_frequency, document_frequency)

    train_path = "../data/train"
    cv_path = "../data/cv"
    test_path = "../data/test"
    raw_path = "../20news-18828"

    if not os.path.exists(tf_path):
        start = time.time()
        print("--------------------------------------------")
        print("TF file not exist, Start Read Train Data Get TF")
        result = utils.ReadDirsToStem(raw_path)
        stop = time.time()
        print("Read Train Data has cost time %fs" % (stop - start))
    

        start = time.time()
        print("--------------------------------------------")
        print("Start Write TF")
        with open(file = tf_path, mode = "w", encoding = "ISO-8859-1") as f:
            f.write(str(result))

        stop = time.time()
        print("Write TF has cost time %fs" % (stop - start))

    start = time.time()
    print("--------------------------------------------")
    print("Start Read TF")
    with open(file = tf_path, mode = "r", encoding = "ISO-8859-1") as f:
        result = eval(f.read())

    stop = time.time()
    print("Read TF has cost time %fs" % (stop - start))
    
    data_train = {}

    filelist = utils.GetFileLists(train_path)
    num_document_train = len(filelist)
    
    for file in filelist:
        data_train[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]

    start = time.time()
    print("--------------------------------------------")
    print("Start Build Dict")
    dic = utils.build_dict(data_train, word_frequency= word_frequency, document_frequency = document_frequency)
    stop = time.time()
    print("Build Dict has cost time %fs" % (stop - start))

    dic_names = []
    for key in dic:
        dic_names.append(key)

    start = time.time()
    print("--------------------------------------------")
    print("Start Build Train VSM")
    vector_space_train, vector_space_train_names = vsm.buildVSM(data_train, dic, dic_names, num_document_train)
    stop = time.time()
    print("Build Train VSM has cost time %fs" % (stop - start))

    filelist = utils.GetFileLists(cv_path)
    num_document_cv = len(filelist)

    data_cv = {}
    for file in filelist:
        data_cv[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]

    start = time.time()
    print("--------------------------------------------")
    print("Start Load CV Data and Build Vector")
    vector_space_cv, vector_space_cv_names = vsm.buildVSM(data_cv, dic, dic_names, num_document_cv)
    stop = time.time()
    print("Load CV Data and build vector has cost time %fs" % (stop - start))

    filelist = utils.GetFileLists(test_path)
    num_document_cv = len(filelist)
    data_test = {}
    for file in filelist:
        data_test[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]

    start = time.time()
    print("--------------------------------------------")
    print("Start Load Test Data and Build Vector")
    vector_space_test, vector_space_test_names = vsm.buildVSM(data_test, dic, dic_names, num_document_cv)
    stop = time.time()
    print("Load Test Data and build vector has cost time %fs" % (stop - start))

    start = time.time()
    print("--------------------------------------------")
    print("Start Convert Data to Matrix")

    vector_space_cv = np.array(vector_space_cv, dtype = np.float)
    vector_space_train = np.array(vector_space_train, dtype = np.float)
    vector_space_test = np.array(vector_space_test, dtype = np.float)

    vector_space_train_names = np.array(vector_space_train_names, dtype = np.str)
    vector_space_cv_names = np.array(vector_space_cv_names, dtype = np.str).reshape(-1, 1)
    vector_space_test_names = np.array(vector_space_test_names, dtype = np.str).reshape(-1, 1)

    stop = time.time()
    print("Convert data to matrix has cost time %fs" % (stop - start))

    start = time.time()
    print("--------------------------------------------")
    print("Start Build Tree")
    vector_space_cv = preprocessing.scale(vector_space_cv, axis = 1)
    vector_space_train = preprocessing.scale(vector_space_train, axis = 1)
    vector_space_test = preprocessing.scale(vector_space_test, axis = 1)

    tree = sklearn.neighbors.BallTree(vector_space_train, metric = "euclidean")
    stop = time.time()
    print("Build Tree has cost time %fs" % (stop - start))

    start = time.time()
    print("--------------------------------------------")
    print("Start Choose K value")

    startPos = 0
    stopPos = vector_space_cv.shape[0]
    # stopPos = 100

    k_min = 2
    k_max = 10

    k_list = np.arange(k_min, k_max)
    predict_accuracy = np.zeros([len(k_list)])
    
    for index in range(len(k_list)):
        predict_labels = []
        print("--------------------------------------------")
        print("Start evaluate k value is %d" % (k_list[index]))
        for i in range(startPos, stopPos):
            # distance = calcEuDistance(vector_space_cv[i,:].reshape(1, -1), vector_space_train)
            # distance = calcCosDistance(vector_space_cv[i,:].reshape(1, -1), vector_space_train)
            # predict_labels.append(collections.Counter(vector_space_train_names[np.argpartition(distance,k)[0:k]]).most_common(1)[0][0])
            # print("%d: " % (i),predict_labels[i], vector_space_cv_names[i][0])
        
            predict_labels.append(collections.Counter(vector_space_train_names[tree.query(vector_space_cv[i,:].reshape(1, -1), k = k_list[index])[1]][0]).most_common(1)[0][0])
            # print("%d: " % (i),predict_labels[i], vector_space_cv_names[i][0])

        predict_labels = np.array(predict_labels, dtype = np.str).reshape(-1, 1)

        accuracy = np.sum(predict_labels == vector_space_cv_names[startPos:stopPos]) / (stopPos - startPos) * 100
        print("k value is %d,Predict Accuracy is %f%%" % (k_list[index], accuracy))
        predict_accuracy[index] = accuracy

    stop = time.time()
    print("Choose K value has cost time %fs" % (stop - start))

    max_index = np.argmax(predict_accuracy)
    k = k_list[max_index]
    print("--------------------------------------------")
    print("max accuracy is %f%%, k value is %d" % (predict_accuracy[max_index], k))

    print("--------------------------------------------")
    print("Start Predict")
    start = time.time()

    predict_labels = []
    
    startPos = 0
    stopPos = vector_space_test.shape[0]
    for i in range(startPos, stopPos):
        predict_labels.append(collections.Counter(vector_space_train_names[tree.query(vector_space_test[i,:].reshape(1, -1), k = k)[1]][0]).most_common(1)[0][0])

    predict_labels = np.array(predict_labels, dtype = np.str).reshape(-1, 1)
    accuracy = np.sum(predict_labels == vector_space_test_names[startPos:stopPos]) / (stopPos - startPos) * 100
    stop = time.time()
    
    print("Predict has cost time %fs" %(stop - start))
    print("Predict Accuracy is %f%%" % (accuracy))
    

