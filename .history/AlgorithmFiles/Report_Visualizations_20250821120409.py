import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        (7, 2), (2, 8), (7, 8),         # top-right triangle
        (2, 0),                         # center-to-left
        (0, 5), (5, 4), (5, 1),         # left chain
        (4, 1), (4, 2)                 # links back to center
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
        (1, 2), (2, 3), (3, 4), (3, 1)
    ])

    pos = {
        1: (0.3, 1.5),
        2: (0.0, 0.8),
        3:  (0.35, 0.45),
        4: (-0.2, 0.3)
    }

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # --- (1) Straight edges
    edge_art = nx.draw_networkx_edges(G, pos, width=2, edge_color="black")
    if isinstance(edge_art, list):
        for coll in edge_art:
            coll.set_zorder(1)
    else:
        edge_art.set_zorder(1)

    # --- (2) Curved parallel edges (make sure zorder > faces)
    for xy, txt, rad in [
        (pos[3], pos[2],  0.30),
        (pos[3], pos[2], -0.30),
        (pos[3], pos[1], -0.50),
        (pos[3], pos[1],  0.30),
        (pos[3], pos[1], -0.30),
        (pos[3], pos[2], -0.30),
        (pos[3], pos[2], 0.30)
    ]:
        a = ax.annotate(
            "", xy=xy, xycoords='data', xytext=txt, textcoords='data',
            arrowprops=dict(arrowstyle="-", color="black",
                            connectionstyle=f"arc3,rad={rad}", linewidth=2),
            annotation_clip=False  # keep visible even near edges of axes
        )
        a.set_zorder(2)

    # --- (3) Relabel and draw nodes on top
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    G = nx.relabel_nodes(G, mapping)
    new_pos = {'A': pos[1], 'B': pos[2], 'C': pos[3], 'D': pos[4], 'E': pos[5]}

    node_art = nx.draw_networkx_nodes(
        G, new_pos, node_size=1400, node_color="lightcoral",
        edgecolors="black", linewidths=2
    )
    node_art.set_zorder(3)

    # --- (4) Labels highest
    label_art = nx.draw_networkx_labels(G, pos, font_size=12)
    for text in label_art.values():
        text.set_zorder(4)

    plt.axis("off")
    plt.tight_layout()
    plt.show()





    
example_dual()
example_dual_p2()