# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/10/15 15:02 
from utils import load_any_obj_pkl
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score
from sklearn.metrics.pairwise import pairwise_distances
from kmedoids import kmedoids
import matplotlib.pyplot as plt
from jqmcvi.base import dunn
from utils import save_dict
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import save_any_obj_pkl
import warnings

warnings.filterwarnings("ignore")


def Dunn_Validity_Index(labels, data, n_clusters):
    pred = pd.DataFrame(labels)
    pred.columns = ["Type"]
    pred.index = data.index.tolist()
    prediction = pd.concat([data, pred], axis=1)
    cluster_list = []
    for j in range(n_clusters):
        temp = prediction.loc[prediction.Type == j]
        cluster_list.append(temp.values)
    return dunn(cluster_list)


def load_DynWalks_Embedding(method):
    data = load_any_obj_pkl("DynWalks/output/hangzhou_20140301_MCC_" + method + "_embs.pkl")[-1]
    X = None
    car_ids = []
    for key, value in data.items():
        car_ids.append(key)
        if X is None:
            X = np.array(value).reshape(1, -1)
        else:
            X = np.vstack((X, value.reshape(1, -1)))
    X = 1.0 * (X - X.mean()) / X.std()
    return pd.DataFrame(X, index=car_ids)


def load_OpenNE_Embedding(method):
    X = None
    car_ids = []
    with open(r"OpenNE\output\hangzhou_20140301_MCC_" + method + "_embs.txt", "r") as embeddings:
        embeddings.readline()
        for embedding in embeddings:
            l = embedding.split()
            car_ids.append(l[0])
            if X is None:
                X = np.array([float(n) for n in l[1:]]).reshape(1, -1)
            else:
                X = np.vstack((X, np.array([float(n) for n in l[1:]]).reshape(1, -1)))
    embeddings.close()
    X = 1.0 * (X - X.mean()) / X.std()
    return pd.DataFrame(X, index=car_ids)


if __name__ == '__main__':
    clustering_method = "SpectralClustering"  # ["KMeans", "HierarchicalClustering", "KMedoids", "SpectralClustering", GaussianMixture]
    embedding_method = "DynWalks"  # ["DynWalks", "DeepWalk", "LINE", "GraRep", "Node2Vec"]
    print("clustering_method:", clustering_method)
    print("embedding_method:", embedding_method)
    if embedding_method == "DynWalks":
        X = load_DynWalks_Embedding(embedding_method)
    else:
        X = load_OpenNE_Embedding(embedding_method)

    sil = dict()
    db = dict()
    ch = dict()
    du = dict()
    if clustering_method == "KMedoids":
        for i in range(2, 51, 1):
            D = pairwise_distances(X, metric='euclidean')
            M, C = kmedoids.kMedoids(D, i)
            node_label = dict()
            for label, nodes in C.items():
                for node in nodes:
                    node_label[node] = label
            node_label = dict(sorted(node_label.items(), key=lambda d: d[0]))
            labels = list(node_label.values())
            du[str(i)] = Dunn_Validity_Index(labels=labels, data=X, n_clusters=i)
            sil[str(i)] = silhouette_score(X, labels)
            db[str(i)] = davies_bouldin_score(X, labels)
            ch[str(i)] = calinski_harabaz_score(X, labels)
    elif clustering_method == "GaussianMixture":
        for i in range(2, 51, 1):
            clu = GaussianMixture(n_components=i)
            labels = clu.fit_predict(X)
            du[str(i)] = Dunn_Validity_Index(labels=labels, data=X, n_clusters=i)
            sil[str(i)] = silhouette_score(X, labels)
            db[str(i)] = davies_bouldin_score(X, labels)
            ch[str(i)] = calinski_harabaz_score(X, labels)
    else:
        for i in range(2, 51, 1):
            # clu = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300, random_state=0)
            # clu = AgglomerativeClustering(n_clusters=i, affinity="euclidean", linkage='average')
            clu = SpectralClustering(n_clusters=i, gamma=0.01)
            clu.fit(X)
            labels = clu.labels_
            du[str(i)] = Dunn_Validity_Index(labels=labels, data=X, n_clusters=i)
            sil[str(i)] = silhouette_score(X, labels)
            db[str(i)] = davies_bouldin_score(X, labels)
            ch[str(i)] = calinski_harabaz_score(X, labels)

    save_any_obj_pkl(sil, "metric_result\\" + clustering_method + "_" + embedding_method + "_sil.pkl")
    save_any_obj_pkl(db, "metric_result\\" + clustering_method + "_" + embedding_method + "_db.pkl")
    save_any_obj_pkl(ch, "metric_result\\" + clustering_method + "_" + embedding_method + "_ch.pkl")
    save_any_obj_pkl(du, "metric_result\\" + clustering_method + "_" + embedding_method + "_du.pkl")

    save_dict(filepath="metric_result\clustering_method\\" + clustering_method + "_" + "silhouette_score.txt", mode="a",
              dic={embedding_method: sil})
    save_dict(filepath="metric_result\clustering_method\\" + clustering_method + "_" + "davies_bouldin_score.txt", mode="a",
              dic={embedding_method: db})
    save_dict(filepath="metric_result\clustering_method\\" + clustering_method + "_" + "calinski_harabaz_score.txt", mode="a",
              dic={embedding_method: ch})
    save_dict(filepath="metric_result\clustering_method\\" + clustering_method + "_" + "dunn_score.txt", mode="a",
              dic={embedding_method: du})

    save_dict(filepath="metric_result\embedding_method\\" + embedding_method + "_" + "silhouette_score.txt", mode="a",
              dic={clustering_method: sil})
    save_dict(filepath="metric_result\embedding_method\\" + embedding_method + "_" + "davies_bouldin_score.txt", mode="a",
              dic={clustering_method: db})
    save_dict(filepath="metric_result\embedding_method\\" + embedding_method + "_" + "calinski_harabaz_score.txt", mode="a",
              dic={clustering_method: ch})
    save_dict(filepath="metric_result\embedding_method\\" + embedding_method + "_" + "dunn_score.txt", mode="a",
              dic={clustering_method: du})

    fig = plt.figure()
    fig.add_subplot(221)
    plt.plot(range(2, 51, 1), list(sil.values()), marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    fig.add_subplot(222)
    plt.plot(range(2, 51, 1), list(db.values()), marker="o")
    plt.xlabel("k")
    plt.ylabel("Davies Bouldin Index")
    fig.add_subplot(223)
    plt.plot(range(2, 51, 1), list(ch.values()), marker="o")
    plt.xlabel("k")
    plt.ylabel("Calinski Harabaz Score")
    fig.add_subplot(224)
    plt.plot(range(2, 51, 1), list(du.values()), marker="o")
    plt.xlabel("k")
    plt.ylabel("Dunn Validity Index")
    plt.show()

    fig.savefig("figures/" + clustering_method + "_" + embedding_method + "_performance.png")
