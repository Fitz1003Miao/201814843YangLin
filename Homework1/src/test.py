from __future__ import print_function
from textblob import TextBlob
import utils

if __name__ == "__main__":
    result = utils.ReadDirsToStem("../20news-18828/alt.atheism")
    
    with open(file = "../data/tf.txt", mode = "w", encoding = "ISO-8859-1") as f:
        f.write(str(result))
    
    with open(file = "../data/tf.txt", mode = "r", encoding = "ISO-8859-1") as f:
        result_new = eval(f.read())
        utils.build_dict(result_new, frequency = 50)
        print(len(utils.load_dict(filePath = "../data/dict_50.txt")))
