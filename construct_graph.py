# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/10/13 11:12

import networkx as nx
import matplotlib.pyplot as plt
from utils import save_any_obj_pkl

def grid_id(longitude, latitude, unit=0.001, n_gribx=400, base_point=None):
    if base_point is None:
        base_point = [120.0, 30.07]
    x, y = (longitude - base_point[0]) // unit, (latitude - base_point[1]) // unit
    # coordinate = [x, y]
    grid = y * n_gribx + x
    return grid


if __name__ == '__main__':
    graph_list = list()
    for hour in range(8, 20):
        g = nx.Graph()
        for minute in [0, 10, 20, 30, 40, 50]:
            cid_pos = dict()
            filepath = r"handled_data\data1\byMinute\data1_" + str(hour) + "_" + str(minute) + ".txt"
            with open(filepath, "r") as lines:
                for line in lines:
                    a = line.split()
                    car_id = a[0]
                    long = float(a[1])
                    lat = float(a[2])
                    cid_pos[car_id] = grid_id(long, lat)
            lines.close()

            car_id = list(cid_pos.keys())
            pos = list(cid_pos.values())
            cid_pos = dict()

            g.add_nodes_from(car_id)

            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    if pos[i] == pos[j]:
                        g.add_edge(car_id[i], car_id[j])
        g = max(nx.connected_component_subgraphs(g), key=len)

        # node_list = []
        # for node in g.nodes():
        #     if g.degree(node) == 0:
        #         node_list.append(node)
        # g.remove_nodes_from(node_list)

        print(str(hour) + "时：", end="")
        print("#nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        # graph_list.append(g)
        # filename = "graph_data\hangzhou_20140301_MCC_" + str(hour) + "h_edgelist.txt"
        # nx.write_edgelist(g, filename, data=False)
        # nx.write_gexf(g, 'hangzhou.gexf')
    # save_any_obj_pkl(graph_list, r"graph_data\hangzhou_20140301_MCC.pkl")
