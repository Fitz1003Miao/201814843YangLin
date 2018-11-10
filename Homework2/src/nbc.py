#-*- coding: utf-8 -*-

import utils, Model
import sys, os, time

class Logger(object):
    def __init__(self, filename = "../log/logout.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(filename = "../log/logout.log")

def main():
    # 1. 统计出现词频
    tf_path = "../data/tf.txt"
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

    # 2. 获取 训练数据 以及 训练label 

    start = time.time()
    print("--------------------------------------------")
    print("Start Train")
    
    print("--------------------------------------------")
    print("Start Load Train Data")

    filelist = utils.GetFileLists(train_path)
    data_train = {}

    for file in filelist:
        data_train[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]
    
    stop = time.time()
    print("Load Train Data has cost %fs" % (stop - start))


    print("--------------------------------------------")
    print("Start Load Test Data")
    start = time.time()

    filelist = utils.GetFileLists(cv_path)
    data_test = {}

    for file in filelist:
        data_test[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]

    filelist = utils.GetFileLists(test_path)
    for file in filelist:
        data_test[file] = result[os.path.join(raw_path, os.path.join(file.split("/")[-2], file.split("/")[-1]))]

    stop = time.time()
    print("--------------------------------------------")
    print("Load Test Data has cost %fs" % (stop - start))

    doc_types = list(utils.GetDirLists(train_path))

    model = Model.PolynomialModel()
    model.train(dataset_train = data_train, labels = doc_types)
    model.test(dataset_test = data_test)

    model = Model.BernoulliModel()
    model.train(dataset_train = data_train, labels = doc_types)
    model.test(dataset_test = data_test)

    model = Model.MixModel()
    model.train(dataset_train = data_train, labels = doc_types)
    model.test(dataset_test = data_test)

if __name__ == "__main__":
    main()