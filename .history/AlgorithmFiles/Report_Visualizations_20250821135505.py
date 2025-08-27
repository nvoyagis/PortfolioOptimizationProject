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
        node_size=600,
        node_color="lightcoral",
        edgecolors="black",
        linewidths=2,
        width=2,
        font_size=12,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def example_dual_p2():
    # ---------- PRIMAL (your graph) ----------
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

    # ---------- DUAL (faces as nodes) ----------
    D = {
        "A": (0.50, 0.10),  # face (1-2-3)
        "B": (0.40, 0.43),  # face (1-3-4)
        "C": (0.60, 0.43),  # face (2-3-4)
        "D": (0.42, 0.75),  # face (1-4-5)
        "E": (0.60, 0.75),  # face (2-4-5)
        "F": (-0.20, 0.50), # outer face
    }

    dual_edges = [
        ("A","B"),  # 1-3
        ("A","C"),  # 2-3
        ("B","C"),  # 3-4
        ("B","D"),  # 1-4
        ("C","E"),  # 2-4
        ("D","E"),  # 4-5
        ("D","F"),  # 1-5
        # EF will be drawn as a spline (around node 5)
    ]

    fig, ax = plt.subplots(figsize=(7,7))

    # ----- draw primal (blue, solid) -----
    for u, v in primal_edges:
        x1,y1 = P[u]; x2,y2 = P[v]
        ax.plot([x1,x2],[y1,y2], linewidth=2, color="blue", zorder=1)
    for v,(x,y) in P.items():
        ax.scatter([x],[y], s=450, edgecolors="black", color="blue", zorder=2)
        ax.text(x, y, str(v), ha="center", va="center", fontsize=12,
                weight="bold", color="white", zorder=3)

    # ----- draw dual straight edges (red, solid) -----
    for a,b in dual_edges:
        x1,y1 = D[a]; x2,y2 = D[b]
        ax.plot([x1,x2],[y1,y2], linewidth=2, color="red", zorder=3)

    # ----- draw dual curved EF around node 5 (red, solid) -----
    xE,yE = D["E"]
    xF,yF = D["F"]
    x5,y5 = P[5]

    # Two control points placed to "wrap" above node 5.
    # Tweak these to change how tightly the spline bends around 5.
    c1 = (x5 + 0.30, y5 + 0.35)   # just above/right of 5
    c2 = (x5 - 0.30, y5 + 0.25)   # above/left toward F

    verts = [(xE,yE), c1, c2, (xF,yF)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ef_spline = PathPatch(Path(verts, codes), lw=2, edgecolor="red",
                          facecolor="none", zorder=3)
    ax.add_patch(ef_spline)

    # ----- draw dual curved AF around node 1 (red, solid) -----
    xE,yE = D["A"]
    xF,yF = D["F"]
    x1,y1 = P[1]

    # Two control points placed to "wrap" above node 5.
    # Tweak these to change how tightly the spline bends around 5.
    c1 = (x1 + 0.25, y1 - 0.35)   # just above/right of 5
    c2 = (x1 - 0.35, y1 - 0.25)   # above/left toward F

    verts = [(xE,yE), c1, c2, (xF,yF)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ef_spline = PathPatch(Path(verts, codes), lw=2, edgecolor="red",
                          facecolor="none", zorder=3)
    ax.add_patch(ef_spline)



        # ----- dual nodes (red) -----
    for f,(x,y) in D.items():
        ax.scatter([x],[y], s=450, edgecolors="black", color="red", zorder=4)
        ax.text(x, y, f, fontsize=10, weight="bold",
                color="white", ha="center", va="center", zorder=5)


    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Primal (blue) with Dual (red) — EF curved around node 5")
    plt.show()











    
example_dual()
example_dual_p2()