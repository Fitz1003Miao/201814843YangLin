import utils
import os, math
import numpy as np
import time

train_path = "../data/train"

class Model(object):
    def __init__(self):
        self.labels_p = {}
        self.labels_word_p = {}
        self.labels = None
        self.name = None

class PolynomialModel(Model):

    def __init__(self):
        Model.__init__(self)
        self.name = "PolynomialModel"

    def GetWordP(self, label, word, count):
        if self.labels_word_p[label].get(word) is None:
            return count * math.log(1.0 / (self.labels_word_total[label] + self.num))
        else:
            return count * math.log(self.labels_word_p[label][word])

    def train(self, dataset_train, labels, word_frequency = 15, document_frequency = 5):
        start = time.time()
        print("--------------------------------------------")
        print("%s Train Start" % self.name)

        self.labels = labels

        self.global_dict = utils.build_dict(dataset_train, word_frequency = word_frequency, document_frequency = document_frequency)
        self.num = len(self.global_dict)

        train_count = len(dataset_train)
        
        self.labels_word_total = {}
        self.labels_word_freq = {}
        self.labels_word_num = {}

        # 计算 每类文档 的概率
        for label in labels:
            self.labels_p[label] = len(utils.GetFileLists(os.path.join(train_path, label))) * 1.0 / train_count

        # 统计 每类文档 中的词频
        for name, data in dataset_train.items():
            label = name.split("/")[-2]

            if self.labels_word_freq.get(label) is None:
                self.labels_word_freq[label] = {}

            if self.labels_word_total.get(label) is None:
                self.labels_word_total[label] = 0
                self.labels_word_num[label] = 0

            for word, count in data.items():

                if self.global_dict.get(word) is None:
                    continue
                
                if self.labels_word_freq[label].get(word) is None:
                    self.labels_word_freq[label][word] = count
                else:
                    self.labels_word_freq[label][word] += count
                
                self.labels_word_total[label] += count
        # 计算每个词的概率
        for label, data in self.labels_word_freq.items():
            if self.labels_word_p.get(label) is None:
                self.labels_word_p[label] = {}
            
            for word, count in data.items():
                if self.global_dict.get(word) is not None:
                    self.labels_word_p[label][word] = (count * 1.0 + 1) / (self.labels_word_total[label] + self.num)
        stop = time.time()
        print("%s Train Finished has cost %fs" % (self.name, stop - start))

    def test(self, dataset_test):
        start = time.time()
        print("--------------------------------------------")
        print("%s Test Start" % (self.name))

        print("labels is {}".format(self.labels))

        num = 0
        for name, data in dataset_test.items():
            label = name.split("/")[-2]

            p = [0] * len(self.labels)

            for index, label_t in enumerate(self.labels):
                for word, count in data.items():
                    if self.global_dict.get(word) is None:
                        continue
                    p[index] += self.GetWordP(label_t, word, count)
                p[index] += math.log10(self.labels_p[label_t])

            predict = self.labels[p.index(max(p))]

            print("%s: label is %s, predict is %s" % (name.split("/")[-1], label, predict), end = ",")
            print("predict log p is {}".format(p))

            if predict == label:
                num += 1
        stop = time.time()
        print("{} Test Finished cost time {}s, Predict accuary is {} / {}, {}%%".format(self.name, (stop - start), num, len(dataset_test), num / len(dataset_test) * 100))

class BernoulliModel(Model):

    def __init__(self):
        Model.__init__(self)
        self.name = "BernoulliModel"

    def GetWordP(self, label, word, count):
        if self.labels_word_p[label].get(word) is None:
            return count * math.log(1.0 / (self.labels_word_total[label] + self.num))
        else:
            return count * math.log(self.labels_word_p[label][word])

    def train(self, dataset_train, labels, word_frequency = 15, document_frequency = 5):
        start = time.time()
        print("--------------------------------------------")
        print("%s Train Start" % self.name)

        self.labels = labels

        self.global_dict = utils.build_dict(dataset_train, word_frequency = word_frequency, document_frequency = document_frequency)
        self.num = len(self.global_dict)

        train_count = len(dataset_train)
        
        self.labels_word_total = {}
        self.labels_word_freq = {}
        self.labels_word_num = {}

        # 计算 每类文档 的概率
        for label in labels:
            self.labels_p[label] = len(utils.GetFileLists(os.path.join(train_path, label))) * 1.0 / train_count

        # 统计 每类文档 中的词频
        for name, data in dataset_train.items():
            label = name.split("/")[-2]

            if self.labels_word_freq.get(label) is None:
                self.labels_word_freq[label] = {}

            if self.labels_word_total.get(label) is None:
                self.labels_word_total[label] = 0
                self.labels_word_num[label] = 0

            for word, count in data.items():

                if self.global_dict.get(word) is None:
                    continue
                
                if self.labels_word_freq[label].get(word) is None:
                    self.labels_word_freq[label][word] = 1
                else:
                    self.labels_word_freq[label][word] += 1
                
                self.labels_word_total[label] += 1

        # 计算每个词的概率
        for label, data in self.labels_word_freq.items():
            if self.labels_word_p.get(label) is None:
                self.labels_word_p[label] = {}
            
            for word, count in data.items():
                if self.global_dict.get(word) is not None:
                    self.labels_word_p[label][word] = (count * 1.0 + 1) / (self.labels_word_total[label] + self.num)
        
        stop = time.time()
        print("%s Train Finished has cost %fs" % (self.name, stop - start))

    def test(self, dataset_test):
        start = time.time()
        print("--------------------------------------------")
        print("%s Test Start" % (self.name))

        print("labels is {}".format(self.labels))

        num = 0
        for name, data in dataset_test.items():
            label = name.split("/")[-2]

            p = [0] * len(self.labels)

            for index, label_t in enumerate(self.labels):
                for word, count in data.items():
                    if self.global_dict.get(word) is None:
                        continue
                    p[index] += self.GetWordP(label_t, word, 1)
                p[index] += math.log10(self.labels_p[label_t])

            predict = self.labels[p.index(max(p))]

            print("%s: label is %s, predict is %s" % (name.split("/")[-1], label, predict), end = ",")
            print("predict log p is {}".format(p))

            if predict == label:
                num += 1
        stop = time.time()
        print("{} Test Finished cost time {}s, Predict accuary is {} / {}, {}%%".format(self.name, (stop - start), num, len(dataset_test), num / len(dataset_test) * 100))


class MixModel(Model):

    def __init__(self):
        Model.__init__(self)
        self.name = "MixModel"

    def GetWordP(self, label, word, count):
        if self.labels_word_p[label].get(word) is None:
            return count * math.log(1.0 / (self.labels_word_total[label] + self.num))
        else:
            return count * math.log(self.labels_word_p[label][word])

    def train(self, dataset_train, labels, word_frequency = 15, document_frequency = 5):
        start = time.time()
        print("--------------------------------------------")
        print("%s Train Start" % self.name)

        self.labels = labels

        self.global_dict = utils.build_dict(dataset_train, word_frequency = word_frequency, document_frequency = document_frequency)
        self.num = len(self.global_dict)

        train_count = len(dataset_train)
        
        self.labels_word_total = {}
        self.labels_word_freq = {}
        self.labels_word_num = {}

        # 计算 每类文档 的概率
        for label in labels:
            self.labels_p[label] = len(utils.GetFileLists(os.path.join(train_path, label))) * 1.0 / train_count

        # 统计 每类文档 中的词频
        for name, data in dataset_train.items():
            label = name.split("/")[-2]

            if self.labels_word_freq.get(label) is None:
                self.labels_word_freq[label] = {}

            if self.labels_word_total.get(label) is None:
                self.labels_word_total[label] = 0
                self.labels_word_num[label] = 0

            for word, count in data.items():

                if self.global_dict.get(word) is None:
                    continue
                
                if self.labels_word_freq[label].get(word) is None:
                    self.labels_word_freq[label][word] = count
                else:
                    self.labels_word_freq[label][word] += count
                
                self.labels_word_total[label] += count
        # 计算每个词的概率
        for label, data in self.labels_word_freq.items():
            if self.labels_word_p.get(label) is None:
                self.labels_word_p[label] = {}
            
            for word, count in data.items():
                if self.global_dict.get(word) is not None:
                    self.labels_word_p[label][word] = (count * 1.0 + 1) / (self.labels_word_total[label] + self.num)
        stop = time.time()
        print("%s Train Finished has cost %fs" % (self.name, stop - start))

    def test(self, dataset_test):
        start = time.time()
        print("--------------------------------------------")
        print("%s Test Start" % (self.name))

        print("labels is {}".format(self.labels))

        num = 0
        for name, data in dataset_test.items():
            label = name.split("/")[-2]

            p = [0] * len(self.labels)

            for index, label_t in enumerate(self.labels):
                for word, count in data.items():
                    if self.global_dict.get(word) is None:
                        continue
                    p[index] += self.GetWordP(label_t, word, 1)
                p[index] += math.log10(self.labels_p[label_t])

            predict = self.labels[p.index(max(p))]

            print("%s: label is %s, predict is %s" % (name.split("/")[-1], label, predict), end = ",")
            print("predict log p is {}".format(p))

            if predict == label:
                num += 1
        stop = time.time()
        print("{} Test Finished cost time {}s, Predict accuary is {} / {}, {}%%".format(self.name, (stop - start), num, len(dataset_test), num / len(dataset_test) * 100))


            

                
                    