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
        (1, 2), (1, 3), (1, 4), (4, 5), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4)
    ]
    G.add_edges_from(edges)

    plt.figure(figsize=(6.8, 6.2))
    nx.draw(
        G, pos=nx.planar_layout(G),
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


# Manual dual of the given planar graph
# Uses a fixed embedding that matches your picture.

import matplotlib.pyplot as plt

def example_dual_p2():
    # ---------- PRIMAL (your graph) ----------
    # Coordinates chosen to resemble your layout
    P = {
        1: (0.00, 0.00),
        2: (1.00, 0.00),
        3: (0.50, 0.37),
        4: (0.50, 0.70),
        5: (0.50, 1.05),
    }
    primal_edges = [
        (1, 2),
        (1, 3), (2, 3),
        (1, 4), (2, 4),
        (1, 5), (2, 5),
        (3, 4), (4, 5),
    ]

    # ---------- DUAL (faces become vertices) ----------
    # Face centers picked by eye so labels sit inside the regions
    D = {
        "A": (0.50, 0.10),  # triangle (1-2-3)
        "B": (0.33, 0.50),  # triangle (1-3-4)
        "C": (0.67, 0.50),  # triangle (2-3-4)
        "D": (0.33, 0.85),  # triangle (1-4-5)
        "E": (0.67, 0.85),  # triangle (2-4-5)
        "F": (-0.20, 0.35), # outer face
    }

    # One dual edge per primal edge: connect the two faces the primal edge separates
    dual_edges = [
        ("F","A"),  # from primal edge 1-2
        ("A","B"),  # 1-3
        ("A","C"),  # 2-3
        ("B","C"),  # 3-4
        ("B","D"),  # 1-4
        ("C","E"),  # 2-4
        ("D","E"),  # 4-5
        ("D","F"),  # 1-5
        ("E","F"),  # 2-5
    ]

    # ---------- PLOT 1: Overlay (primal + dual) ----------
    plt.figure(figsize=(7,7))

    # primal edges & nodes
    for u, v in primal_edges:
        x1,y1 = P[u]; x2,y2 = P[v]
        plt.plot([x1,x2],[y1,y2], linewidth=2, zorder=1)
    for v,(x,y) in P.items():
        plt.scatter([x],[y], s=800, edgecolors='black', zorder=2)
        plt.text(x, y, str(v), ha='center', va='center', fontsize=12,
                weight='bold', color='white', zorder=3)

    # dual nodes & edges (dashed)
    for f,(x,y) in D.items():
        plt.scatter([x],[y], s=300, zorder=4)
        plt.text(x+0.02, y+0.02, f, fontsize=10, weight='bold', zorder=5)
    for a,b in dual_edges:
        x1,y1 = D[a]; x2,y2 = D[b]
        plt.plot([x1,x2],[y1,y2], linestyle='--', linewidth=2, zorder=3)

    plt.axis('equal'); plt.axis('off')
    plt.title("Primal with its Dual overlaid (dashed)")
    plt.show()

    # ---------- PLOT 2: Dual only ----------
    plt.figure(figsize=(7,7))
    for a,b in dual_edges:
        x1,y1 = D[a]; x2,y2 = D[b]
        plt.plot([x1,x2],[y1,y2], linestyle='--', linewidth=2)
    for f,(x,y) in D.items():
        plt.scatter([x],[y], s=600, edgecolors='black')
        plt.text(x, y, f, ha='center', va='center', fontsize=12, weight='bold')
    plt.axis('equal'); plt.axis('off')
    plt.title("Dual graph (faces as vertices)")
    plt.show()






    
example_dual()
example_dual_p2()