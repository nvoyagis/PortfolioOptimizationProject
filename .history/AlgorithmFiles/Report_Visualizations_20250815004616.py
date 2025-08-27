import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cvxpy
import sklearn
import fast_tmfg
import Graph_Theory_Functions
import Dual
import os
import time
from itertools import combinations
# import AlgorithmFiles.Charts as Charts

def complete_graph(stock_dict):
    # Create complete graph w/ covariance using the selected start date
    complete_g = nx.complete_graph(n=len(stock_dict))
    complete_g = nx.relabel_nodes(complete_g, stock_dict)
    pos0 = nx.spring_layout(complete_g, scale=5, seed=1)
    nx.draw(complete_g, pos=pos0, node_color='#5192b8', node_size=600)
    nx.draw(complete_g, pos=pos0, with_labels=True, node_color='#8fd6ff', edge_color='#5192b8', node_size=650, font_size=8)
    plt.show()

def example_dual():
    G = nx.Graph()

    # Edges read from the picture
    edges = [
        (7, 2), (2, 8), (7, 8),        # top-right triangle
        (2, 0),                         # center-to-left
        (0, 5), (5, 4), (5, 1),         # left chain
        (4, 1), (4, 2),                 # links back to center
        (2, 3), (3, 6), (1, 6)          # right/bottom chain
    ]
    G.add_edges_from(edges)

    # Hand-tuned positions to match the figure
    pos = {
        7: (0.10, 1.55),
        2: (0.00, 0.35),
        8: (1.55, 0.45),

        0: (-1.10, 0.25),
        5: (-1.70, -0.75),
        4: (-0.70, -0.80),
        1: (-0.95, -1.75),

        3: (0.60, -0.40),
        6: (0.85, -1.40),
    }

    plt.figure(figsize=(6.8, 6.2))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1400,
        node_color="lightcoral",
        edgecolors="black",
        linewidths=2,
        width=2,
        font_size=12,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.8, 6.2))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1400,
        node_color='#5192b8',
        edgecolors="black",
        linewidths=2,
        width=2,
        font_size=12,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def example_dual_p2():
    G = nx.MultiGraph()
    G.add_edges_from([
        (13, 12), (13, 9), (13, 10), (13, 11),
        (9, 10), (9, 11), (10, 11)
    ])

    pos = {
        12: (0.3, 1.5),
        13: (0.0, 0.8),
        9:  (0.35, 0.45),
        10: (-0.2, 0.3),
        11: (1.0, 0.3),
    }

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # --- 1) Draw straight edges first
    edge_art = nx.draw_networkx_edges(G, pos, width=2, edge_color="black")
    # Force those edges *below* nodes by zorder
    if isinstance(edge_art, list):  # MultiGraph can return a list of LineCollections
        for coll in edge_art:
            coll.set_zorder(1)
    else:
        edge_art.set_zorder(1)

    # --- 2) Curved parallel edges (also below nodes)
    a1 = ax.annotate("", xy=pos[12], xycoords='data',
                    xytext=pos[13], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle="arc3,rad=0.30",
                                    linewidth=2))
    a1.set_zorder(0)

    a2 = ax.annotate("", xy=pos[12], xycoords='data',
                    xytext=pos[13], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle="arc3,rad=-0.30",
                                    linewidth=2))
    a2.set_zorder(0)

    a3 = ax.annotate("", xy=pos[11], xycoords='data',
                    xytext=pos[13], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle="arc3,rad=-0.5",
                                    linewidth=2))
    a3.set_zorder(0)

    a4 = ax.annotate("", xy=pos[10], xycoords='data',
                    xytext=pos[13], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle="arc3,rad=0.3",
                                    linewidth=2))
    a4.set_zorder(0)

    a5 = ax.annotate("", xy=pos[11], xycoords='data',
                    xytext=pos[13], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle="arc3,rad=-0.3",
                                    linewidth=2))
    a5.set_zorder(0)

    # Rename numeric labels to strings
    mapping = {
        12: 'A',
        13: 'B',
        9: 'C',
        10: 'D',
        11: 'E'
    }
    G = nx.relabel_nodes(G, mapping)

    # --- 3) Draw nodes on top
    node_art = nx.draw_networkx_nodes(
        G, pos, node_size=1400, node_color="lightcoral",
        edgecolors="black", linewidths=2, label=mapping
    )
    node_art.set_zorder(3)  # ensure above edges/annotations

    # --- 4) Labels highest
    label_art = nx.draw_networkx_labels(G, pos, font_size=12)
    for text in label_art.values():
        text.set_zorder(4)

    plt.axis("off")
    plt.tight_layout()
    plt.show()




    
# example_dual()
example_dual_p2()