#-*- coding: utf-8 -*-
import os
from textblob import TextBlob
import collections

from nltk.corpus import stopwords

def ReadFile(filePath = None):
    if filePath is None or not os.path.exists(filePath):
        return None

    with open(file = filePath, mode = "r", encoding = "ISO-8859-1") as f:
        return f.read()

def GetFileLists(dirPath = None):
    if dirPath is None or not os.path.isdir(dirPath):
        return None
    result = set()

    for maindir, subdirs, files in os.walk(dirPath):
        for subdir in subdirs:
            result.update(GetFileLists(os.path.join(maindir, subdir)))
        for file in files:
            result.add(os.path.join(maindir, file))

    return result

def GetDirLists(dirPath = None):
    if dirPath is None or not os.path.isdir(dirPath):
        return None
    result = set()
    
    for maindir, subdirs, files in os.walk(dirPath):
        result.update(subdirs)
        for subdir in subdirs:
            result.update(GetDirLists(os.path.join(maindir, subdir)))

    return result



def ReadDirsToStem(dirPath = None):

    if dirPath is None or not os.path.isdir(dirPath):
        return None
    print(dirPath)
    result = {}

    for maindir, subdirs, files in os.walk(dirPath):
        subdirs.sort()
        for subdir in subdirs:
            result.update(ReadDirsToStem(os.path.join(maindir, subdir)))

        for file in files:
            words = TextBlob(ReadFile(filePath = os.path.join(maindir, file))).words.lower().stem()
            result[os.path.join(maindir, file)] = dict(collections.Counter(words))
    
    return result

def build_dict(datas, word_frequency = 0, document_frequency = 0):
    s = set()
    dic = {}
    dic_num = {}

    stop_words = set(stopwords.words('english'))
    for name, data in datas.items():
        for word, count in data.items():
            if word in stop_words:
                continue

            if dic.get(word) is None:
                dic[word] = count
                dic_num[word] = 1
            else:
                dic[word] += count
                dic_num[word] += 1
            
            if dic[word] >= word_frequency and word not in s:
                s.add(word)

    result = {}
    for word in s:
        if dic_num[word] >= document_frequency:
            result[word] = dic_num[word]
    return result

def load_dict(filePath = None):

    if filePath is None or not os.path.exists(filePath):
        return None
    s = {}
    with open(file = filePath, mode = "r", encoding = "ISO-8859-1") as f:
        s = eval(f.read())
    return s
    