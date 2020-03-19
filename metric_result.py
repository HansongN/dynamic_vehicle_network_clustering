# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/10/20 21:39 
from utils import load_dict
import json
import matplotlib.pyplot as plt
import os


def clustering_method_performance(clu_method, emb_method, metric, color, marker, metric_name):
    filepath = "metric_result\clustering_method\\" + clu_method + "_" + metric + ".txt"
    fig = plt.figure()
    i = 0
    with open(filepath, "r") as methods:
        for method in methods:
            m_json = json.loads(method)
            m_name = list(m_json.keys())[0]
            if m_name in emb_method:
                data = list(m_json.values())[0]
                x = [int(i) for i in list(data.keys())]
                y = [float(i) for i in list(data.values())]
                plt.plot(x, y,
                         label=m_name,
                         color=color[m_name],
                         # marker=marker[m_name]
                         )
                i += 1
    methods.close()
    plt.xticks(fontsize=18)
    plt.xlabel(r"$k$", fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel(metric_name, fontdict={"size": 20})
    # plt.title(clustering_method_name[clu_method], fontdict={"size": 17})
    plt.legend(fontsize=18, loc="best")
    plt.show()

    fig.savefig("figures\clustering_method\\" + clu_method + "_" + metric_name.replace(" ", "_") + ".eps",
                dpi=400,
                format='eps')


def embedding_method_performance(emb_method, clu_method, metric, color, marker, metric_name):
    filepath = "metric_result\embedding_method\\" + emb_method + "_" + metric + ".txt"

    fig = plt.figure()
    i = 0
    with open(filepath, "r") as methods:
        for method in methods:
            m_json = json.loads(method)
            m_name = list(m_json.keys())[0]
            if m_name in clu_method:
                data = list(m_json.values())[0]
                x = [int(i) for i in list(data.keys())]
                y = [float(i) for i in list(data.values())]
                plt.plot(x, y,
                         label=clustering_method_name[m_name],
                         color=color[m_name],
                         # marker=marker[m_name]
                         )
                i += 1
    methods.close()
    plt.xticks(fontsize=18)
    plt.xlabel(r"$k$", fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel(metric_name, fontdict={"size": 20})
    # plt.title(emb_method, fontdict={"size": 17})
    plt.legend(fontsize=18, loc="best")
    plt.show()

    fig.savefig("figures\embedding_method\\" + emb_method + "_" + metric_name.replace(" ", "_") + ".eps",
                dpi=400,
                format='eps')


if __name__ == '__main__':
    if not os.path.exists("figures/embedding_method"):
        os.makedirs("figures/embedding_method")
    if not os.path.exists("figures/clustering_method"):
        os.makedirs("figures/clustering_method")

    clustering_method = ["KMeans", "HierarchicalClustering", "KMedoids", "SpectralClustering", "GaussianMixture"]
    clustering_method_name = {"KMeans": r"$K$-means",
                              "HierarchicalClustering": "Hierarchical Clustering",
                              "KMedoids": r"$K$-medoids",
                              "SpectralClustering": "Spectral Clustering",
                              "GaussianMixture": "GMM"}
    color0 = {"KMeans": "r", "KMedoids": "y", "GaussianMixture": "g", "SpectralClustering": "c", "HierarchicalClustering": "b"}
    marker0 = {"KMeans": "o", "KMedoids": "*", "GaussianMixture": "^", "SpectralClustering": "v", "HierarchicalClustering": "x"}
    embedding_method = ["DynWalks", "DeepWalk", "LINE", "GraRep", "Node2Vec"]
    color1 = {"DynWalks": "r", "DeepWalk": "y", "LINE": "g", "GraRep": "c", "Node2Vec": "b"}
    marker1 = {"DynWalks": "o", "DeepWalk": "*", "LINE": "^", "GraRep": "v", "Node2Vec": "x"}
    metrics = ["silhouette_score", "davies_bouldin_score", "calinski_harabaz_score", "dunn_score"]
    metrics_name = ["Silhouette Score", "Davies Bouldin Index", "Calinski Harabaz Score", "Dunn Validity Index"]

    used_clustering_method = ["KMeans", "KMedoids", "GaussianMixture"]
    used_embedding_method = ["DynWalks", "DeepWalk", "LINE"]
    used_metrics_name = ["Silhouette Coefficient", "Davies Bouldin Index", "Calinski Harabaz Index"]
    used_metrics = ["silhouette_score", "davies_bouldin_score", "calinski_harabaz_score"]

    # for j in range(len(used_clustering_method)):
    #     for i in range(len(used_metrics_name)):
    #         metrics_index = i
    #         clustering_method_performance(clu_method=used_clustering_method[j],
    #                                       emb_method=used_embedding_method,
    #                                       metric=used_metrics[metrics_index],
    #                                       color=color1,
    #                                       marker = marker1,
    #                                       metric_name=used_metrics_name[metrics_index])

    for j in range(len(used_embedding_method)):
        for i in range(len(used_metrics_name)):
            metrics_index = i
            embedding_method_performance(emb_method=used_embedding_method[j],
                                         clu_method=used_clustering_method,
                                         metric=used_metrics[metrics_index],
                                         color=color0,
                                         marker=marker0,
                                         metric_name=used_metrics_name[metrics_index])

