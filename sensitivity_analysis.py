# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/10/25 15:55 
import numpy as np
from utils import load_any_obj_pkl, save_dict, load_dict
import pandas as pd
from sklearn.cluster import KMeans
from clustering import Dunn_Validity_Index
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score
import os


def loadData2PD(filepath):
    data = load_any_obj_pkl(filepath)[
        -1]
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


def clusteringPerformance(X):
    clu = KMeans(n_clusters=30, init="k-means++", n_init=10, max_iter=300, random_state=0)
    clu.fit(X)
    labels = clu.labels_
    sil = silhouette_score(X, labels)
    db= davies_bouldin_score(X, labels)
    ch = calinski_harabaz_score(X, labels)
    return sil, db, ch


def sensitivity(para_name, para_values, spara_name, spara_values, filepath_part1, filepath_part2, filepath_part3):
    # sil = dict()
    # db = dict()
    # ch = dict()
    # for i in range(len(spara_values)):
    #     silt = dict()
    #     dbt = dict()
    #     cht = dict()
    #     for j in range(len(para_values)):
    #         if para_name == "alpha" or para_name == "nwalk":
    #             filepath = filepath_part1 + str(para_values[j]) + filepath_part2 + str(spara_values[i]) + filepath_part3
    #         else:
    #             filepath = filepath_part1 + str(spara_values[i]) + filepath_part2 + str(para_values[j]) + filepath_part3
    #         X = loadData2PD(filepath)
    #         silt[str(para_values[j])], dbt[str(para_values[j])], cht[str(para_values[j])] = clusteringPerformance(X)
    #     sil[str(spara_values[i])], db[str(spara_values[i])], ch[str(spara_values[i])] = silt, dbt, cht
    
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_silhouette_score.txt", mode="a", dic=sil)
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_davies_bouldin_score.txt", mode="a", dic=db)
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_calinski_harabaz_score.txt", mode="a", dic=ch)

    p_name_notation = {"alpha": r"$\alpha$", "beta": r"$\beta$", "nwalk": r"$r$", "dim": r"$d$"}

    sil = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name +"_silhouette_score.txt")
    db = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name +"_davies_bouldin_score.txt")
    ch = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name +"_calinski_harabaz_score.txt")
    # du = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name +"_dunn_score.txt")

    fig = plt.figure()
    i = 0
    for key, values in sil.items():
        plt.plot(para_values,
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Silhouette Coefficient", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Silhouette_Coefficient.eps",
                dpi=400,
                format='eps')

    fig = plt.figure()
    i = 0
    for key, values in db.items():
        plt.plot(para_values,
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Davies Bouldin Index", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Davies_Bouldin_Index.eps",
                dpi=400,
                format='eps')

    fig = plt.figure()
    i = 0
    for key, values in ch.items():
        plt.plot(para_values,
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Calinski Harabaz Index", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Calinski_Harabaz_Index.eps",
                dpi=400,
                format='eps')

    # fig = plt.figure()
    # plt.plot(para_values,
    #          [float(i) for i in list(du.values())],
    #          label="Dunn Validity Index",
    #          color="r")
    # plt.xticks(fontsize=13)
    # plt.xlabel(p_name_notation[para_name], fontdict={"size": 13})
    # plt.yticks(fontsize=13)
    # plt.ylabel("Dunn Validity Index", fontdict={"size": 13})
    # plt.title(para_name, fontdict={"size": 13})
    # plt.show()
    # fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_dunn_score.png")


def alpha_analysis():
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beta_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    sensitivity(para_name="alpha",
                para_values=alpha_list,
                spara_name="beta",
                spara_values=beta_list,
                filepath_part1="DynWalks/output/DynWalks/hangzhou_20140301_MCC_a",
                filepath_part2="_b",
                filepath_part3="_embs.pkl")


def beta_analysis():
    beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha_list = [0.2, 0.4, 0.6, 0.8]
    sensitivity(para_name="beta",
                para_values=beta_list,
                spara_name="alpha",
                spara_values=alpha_list,
                filepath_part1="DynWalks/output/DynWalks/hangzhou_20140301_MCC_a",
                filepath_part2="_b",
                filepath_part3="_embs.pkl")


def nwalk_analysis():
    nwalk_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dim_list = [16, 32, 64, 128, 256, 512]
    sensitivity_nwalk(para_name="nwalk",
                para_values=nwalk_list,
                spara_name="dim",
                spara_values=dim_list,
                filepath_part1="DynWalks/output/DynWalks/hangzhou_20140301_MCC_nwalk",
                filepath_part2="_dim",
                filepath_part3="_embs.pkl")


def dim_analysis():
    dim_list = [16, 32, 64, 128, 256, 512]
    nwalk_list = [10, 20, 40, 80]
    sensitivity_dim(para_name="dim",
                para_values=dim_list,
                spara_name="nwalk",
                spara_values=nwalk_list,
                filepath_part1="DynWalks/output/DynWalks/hangzhou_20140301_MCC_nwalk",
                filepath_part2="_dim",
                filepath_part3="_embs.pkl")


def sensitivity_nwalk(para_name, para_values, spara_name, spara_values, filepath_part1, filepath_part2, filepath_part3):
    # sil = dict()
    # db = dict()
    # ch = dict()
    # for i in range(len(spara_values)):
    #     silt = dict()
    #     dbt = dict()
    #     cht = dict()
    #     for j in range(len(para_values)):
    #         if para_name == "alpha" or para_name == "nwalk":
    #             filepath = filepath_part1 + str(para_values[j]) + filepath_part2 + str(spara_values[i]) + filepath_part3
    #         else:
    #             filepath = filepath_part1 + str(spara_values[i]) + filepath_part2 + str(para_values[j]) + filepath_part3
    #         X = loadData2PD(filepath)
    #         silt[str(para_values[j])], dbt[str(para_values[j])], cht[str(para_values[j])] = clusteringPerformance(X)
    #     sil[str(spara_values[i])], db[str(spara_values[i])], ch[str(spara_values[i])] = silt, dbt, cht

    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_silhouette_score.txt", mode="a", dic=sil)
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_davies_bouldin_score.txt", mode="a", dic=db)
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_calinski_harabaz_score.txt", mode="a", dic=ch)

    p_name_notation = {"alpha": r"$\alpha$", "beta": r"$\beta$", "nwalk": r"$r$", "dim": r"$d$"}

    sil = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name + "_silhouette_score.txt")
    db = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name + "_davies_bouldin_score.txt")
    ch = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name + "_calinski_harabaz_score.txt")
    # du = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name +"_dunn_score.txt")

    fig = plt.figure()
    i = 0
    for key, values in sil.items():
        plt.plot([i for i in range(len(para_values))],
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Silhouette Coefficient", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Silhouette_Coefficient.eps",
                dpi=400,
                format='eps')

    fig = plt.figure()
    i = 0
    for key, values in db.items():
        plt.plot([i for i in range(len(para_values))],
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Davies Bouldin Index", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Davies_Bouldin_Index.eps",
                dpi=400,
                format='eps')

    fig = plt.figure()
    i = 0
    for key, values in ch.items():
        plt.plot([i for i in range(len(para_values))],
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Calinski Harabaz Index", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Calinski_Harabaz_Index.eps",
                dpi=400,
                format='eps')


def sensitivity_dim(para_name, para_values, spara_name, spara_values, filepath_part1, filepath_part2, filepath_part3):
    # sil = dict()
    # db = dict()
    # ch = dict()
    # for i in range(len(spara_values)):
    #     silt = dict()
    #     dbt = dict()
    #     cht = dict()
    #     for j in range(len(para_values)):
    #         if para_name == "alpha" or para_name == "nwalk":
    #             filepath = filepath_part1 + str(para_values[j]) + filepath_part2 + str(spara_values[i]) + filepath_part3
    #         else:
    #             filepath = filepath_part1 + str(spara_values[i]) + filepath_part2 + str(para_values[j]) + filepath_part3
    #         X = loadData2PD(filepath)
    #         silt[str(para_values[j])], dbt[str(para_values[j])], cht[str(para_values[j])] = clusteringPerformance(X)
    #     sil[str(spara_values[i])], db[str(spara_values[i])], ch[str(spara_values[i])] = silt, dbt, cht

    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_silhouette_score.txt", mode="a", dic=sil)
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_davies_bouldin_score.txt", mode="a", dic=db)
    # save_dict(filepath="metric_result\sensitivity_analysis\DynWalks_" + para_name + "_calinski_harabaz_score.txt", mode="a", dic=ch)

    p_name_notation = {"alpha": r"$\alpha$", "beta": r"$\beta$", "nwalk": r"$r$", "dim": r"$d$"}

    sil = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name + "_silhouette_score.txt")
    db = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name + "_davies_bouldin_score.txt")
    ch = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name + "_calinski_harabaz_score.txt")
    # du = load_dict("metric_result\sensitivity_analysis\DynWalks_" + para_name +"_dunn_score.txt")

    fig = plt.figure()
    i = 0
    for key, values in sil.items():
        plt.plot([i for i in range(len(para_values))],
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks([0, 1, 2, 3, 4, 5],
               [r"$2^4$", r"$2^5$", r"$2^6$", r"$2^7$", r"$2^8$", r"$2^9$"],
               fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Silhouette Coefficient", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Silhouette_Coefficient.eps",
                dpi=400,
                format='eps')

    fig = plt.figure()
    i = 0
    for key, values in db.items():
        plt.plot([i for i in range(len(para_values))],
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks([0, 1, 2, 3, 4, 5],
               [r"$2^4$", r"$2^5$", r"$2^6$", r"$2^7$", r"$2^8$", r"$2^9$"],
               fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Davies Bouldin Index", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Davies_Bouldin_Index.eps",
                dpi=400,
                format='eps')

    fig = plt.figure()
    i = 0
    for key, values in ch.items():
        plt.plot([i for i in range(len(para_values))],
                 [float(i) for i in list(values.values())],
                 label=p_name_notation[spara_name] + "=" + key,
                 color=colors[i],
                 marker=markers[i])
        i += 1
    plt.legend(fontsize=18)
    plt.xticks([0, 1, 2, 3, 4, 5],
               [r"$2^4$", r"$2^5$", r"$2^6$", r"$2^7$", r"$2^8$", r"$2^9$"],
               fontsize=18)
    plt.xlabel(p_name_notation[para_name], fontdict={"size": 20})
    plt.yticks(fontsize=18)
    plt.ylabel("Calinski Harabaz Index", fontdict={"size": 20})
    # plt.title(para_name, fontdict={"size": 13})
    plt.show()
    fig.savefig("figures\sensitivity_analysis\DynWalks_" + para_name + "_Calinski_Harabaz_Index.eps",
                dpi=400,
                format='eps')


if __name__ == '__main__':
    if not os.path.exists("figures/sensitivity_analysis"):
        os.makedirs("figures/sensitivity_analysis")
    colors = ["r", "y", "g", "c", "b", "m"]
    markers = ["o", "v", "^", "<", ">", "x"]
    # alpha_analysis()
    # beta_analysis()
    nwalk_analysis()
    # dim_analysis()
