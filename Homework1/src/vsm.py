# -*- coding: utf-8 -*-
from __future__ import print_function
from textblob import TextBlob
import utils
import os
import math

def buildVSM(datas, dic, dic_names, num_document):
    vector_space = []
    vector_space_names = []
    for name, data in datas.items():
        vector = []
        vector.extend([0] * len(dic_names))
        vector_space_names.append(name.split("/")[-2])
        
        for i in range(len(dic_names)):
            if data.get(dic_names[i]) is not None:
                tf = 1 + math.log(data[dic_names[i]])
                idf = math.log(num_document / dic[dic_names[i]])
                vector[i] = tf * idf

        vector_space.append(vector)

    return vector_space, vector_space_names


if __name__ == "__main__":

    tf_path = "../data/tf.txt"
    word_frequency = 50
    document_frequency = 25
    
    train_path = "../data/train"
    test_path = "../data/test"
    raw_path = "../20news-18828"

    filelist = utils.GetFileLists(train_path)

    if not os.path.exists(tf_path):

        result = utils.ReadDirsToStem(raw_path)
    
        with open(file = tf_path, mode = "w", encoding = "ISO-8859-1") as f:
            f.write(str(result))

    with open(file = tf_path, mode = "r", encoding = "ISO-8859-1") as f:
        result = eval(f.read())
        result_new = {}
        num_document = len(filelist)
        print(num_document)
        for file in filelist:
            result_new[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]

    dic = utils.build_dict(result_new, word_frequency= word_frequency, document_frequency = document_frequency)

    dic_names = []
    for key in dic:
        dic_names.append(key)

    vector_space = buildVSM(result_new, dic, dic_names, num_document)



    
    