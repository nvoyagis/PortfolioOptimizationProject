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
    # Graph and positions
    G = nx.MultiGraph()
    G.add_edges_from([
        (13, 12), (13, 9), (13, 10), (13, 11),
        (9, 10), (9, 11), (10, 11)
    ])

    pos = {
        12: (0.30, 1.50),
        13: (0.00, 0.80),
        9:  (0.40, 0.55),
        10: (-0.25, 0.10),
        11: (1.10, 0.10)
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- draw straight edges EXCEPT the ones we’ll curve
    straight_edges = [(9,10), (9,11), (10,11)]
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, width=2, edge_color="black")

    def curved_edge(p1, p2, rad=0.25, lw=2):
        """Draw a curved edge between nodes p1->p2 with matplotlib annotate."""
        ax.annotate("",
            xy=pos[p2], xycoords='data',
            xytext=pos[p1], textcoords='data',
            arrowprops=dict(arrowstyle="-", color="black",
                            connectionstyle=f"arc3,rad={rad}",
                            linewidth=lw)
        )

    # --- curved pair already requested earlier (two parallels)
    curved_edge(13, 12,  rad= 0.30)
    curved_edge(13, 12,  rad=-0.30)

    # --- NEW curved edges
    curved_edge(13, 10,  rad= 0.18)
    curved_edge(13, 11,  rad=-0.18)
    curved_edge(13, 9,   rad= 0.12)

    # --- nodes and labels on top
    nodes = nx.draw_networkx_nodes(G, pos, node_size=1400,
                                node_color="#ee8e8e", edgecolors="black", linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()




    
# example_dual()
example_dual_p2()