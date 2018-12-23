import argparse
import json
import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
# from sklearn.cluster import Ward
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import sys
class Logger(object):
    def __init__(self, filename = "../log/logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def loadData(filepath):
    with open(filepath, 'r') as f:
        datas = []
        labels = []
        for line in f.readlines():
            line = line.strip()
            data = json.loads(line)
            datas.append(data['text'])
            labels.append(data['cluster'])

        return datas, labels

def TF_IDF(datas):
    tfidf = TfidfVectorizer(stop_words = 'english').fit_transform(datas)
    return tfidf

def k_means(datas, labels):
    n_clusters = len(set(labels))
    labels_pred = KMeans(n_clusters = n_clusters).fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def affinity_propagation(datas, labels):
    labels_pred = AffinityPropagation(damping = 0.5).fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def mean_shift(datas, labels):
    labels_pred = MeanShift(bandwidth = 0.5, bin_seeding=True).fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def spectral_clustering(datas, labels):
    n_clusters = len(set(labels))
    labels_pred = SpectralClustering(n_clusters = n_clusters, eigen_solver='arpack', affinity="nearest_neighbors").fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def ward_hierarchical_clustering(datas, labels):
    n_clusters = len(set(labels))
    labels_pred = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def agglomerative_clustering(datas, labels):
    n_clusters = len(set(labels))
    labels_pred = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def dbscan(datas, labels):
    labels_pred = DBSCAN(eps=1.13).fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def gaussian_mixtures(datas, labels):
    n_clusters = len(set(labels))
    gmm = GaussianMixture(n_components = n_clusters, covariance_type='diag')
    labels_pred = gmm.fit_predict(datas)
    # score = adjusted_mutual_info_score(labels, labels_pred)
    score = normalized_mutual_info_score(labels, labels_pred)
    return score

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--filepath", "-f", help = "Path to file")
    parse.add_argument("--log", "-l", help = "Path to log")
    args = parse.parse_args()
    print(args)
    
    sys.stdout = Logger(filename = args.log)
    datas, labels = loadData(args.filepath)
    datas = TF_IDF(datas).toarray()

    methods = { "KMeans" : k_means, "AffinityPropagation" : affinity_propagation, 
                "MeanShift" : mean_shift, "SpectralClustering" : spectral_clustering, 
                "Ward Hierarchical Clustering" : ward_hierarchical_clustering, "AgglomerativeClustering" : agglomerative_clustering, 
                "DBSCAN" : dbscan, "GaussianMixture" : gaussian_mixtures }

    for name, f in methods.items():
        print("---------- {} ----------".format(name))

        start = time.time()
        score = f(datas, labels)
        stop = time.time()

        print("{} cost time {}s, score is {} \n\n".format(name, (stop - start), score))
        sys.stdout.flush()

if __name__ == "__main__":
    main()