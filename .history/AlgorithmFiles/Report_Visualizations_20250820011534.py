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

    # --- (0) Fill faces FIRST and lowest
    faces = [
        [pos[12], pos[13], pos[9]],    # A-B-C
        [pos[13], pos[9], pos[10]],    # B-C-D
        [pos[13], pos[9], pos[11]],    # B-C-E
        [pos[9],  pos[10], pos[11]],   # C-D-E
    ]
    face_colors = ["#fce4ec", "#e1f5fe", "#e8f5e9", "#fff3e0"]
    for f, color in zip(faces, face_colors):
        ax.add_patch(
            patches.Polygon(f, closed=True, facecolor=color,
                            edgecolor="none", alpha=0.6, zorder=0)
        )

    # --- (1) Straight edges
    edge_art = nx.draw_networkx_edges(G, pos, width=2, edge_color="black")
    if isinstance(edge_art, list):
        for coll in edge_art:
            coll.set_zorder(1)
    else:
        edge_art.set_zorder(1)

    # --- (2) Curved parallel edges (make sure zorder > faces)
    for xy, txt, rad in [
        (pos[12], pos[13],  0.30),
        (pos[12], pos[13], -0.30),
        (pos[11], pos[13], -0.50),
        (pos[10], pos[13],  0.30),
        (pos[11], pos[13], -0.30),
    ]:
        a = ax.annotate(
            "", xy=xy, xycoords='data', xytext=txt, textcoords='data',
            arrowprops=dict(arrowstyle="-", color="black",
                            connectionstyle=f"arc3,rad={rad}", linewidth=2),
            annotation_clip=False  # keep visible even near edges of axes
        )
        a.set_zorder(2)

    # --- (3) Relabel and draw nodes on top
    mapping = {12: 'A', 13: 'B', 9: 'C', 10: 'D', 11: 'E'}
    G = nx.relabel_nodes(G, mapping)
    new_pos = {'A': pos[12], 'B': pos[13], 'C': pos[9], 'D': pos[10], 'E': pos[11]}

    node_art = nx.draw_networkx_nodes(
        G, new_pos, node_size=1400, node_color="lightcoral",
        edgecolors="black", linewidths=2
    )
    node_art.set_zorder(3)

    # --- (4) Labels highest
    label_art = nx.draw_networkx_labels(G, new_pos, font_size=12)
    for text in label_art.values():
        text.set_zorder(4)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import networkx as nx

def example_dual_p2_hardcoded():
    # ----- graph & layout (same as yours) -----
    G = nx.MultiGraph()
    G.add_edges_from([
        (13, 12), (13, 9), (13, 10), (13, 11),
        (9, 10), (9, 11), (10, 11)
    ])

    A = (0.3, 1.5)
    B = (0.0, 0.8)
    C = (0.35, 0.45)
    D = (-0.2, 0.3)
    E = (1.0, 0.3)

    pos = {12: A, 13: B, 9: C, 10: D, 11: E}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='datalim')

    # =========================
    #  FILLS (drawn first)
    # =========================

    # --- 1) Straight-edge triangular faces ---
    straight_faces = [
        [B, C, D],   # B-C-D
        [B, C, E],   # B-C-E
        [C, D, E],   # C-D-E
    ]
    straight_cols = ["#e1f5fe", "#e8f5e9", "#fff3e0"]
    for f, col in zip(straight_faces, straight_cols):
        ax.add_patch(patches.Polygon(
            f, closed=True, facecolor=col, edgecolor="none", alpha=0.45, zorder=0
        ))

    # --- 2) Curved triangle A–B–C (AB is curved) ---
    # Control point chosen by eye to follow your inner A–B arc
    cp_AB_inner = (0.50, 1.00)  # adjust if you want more/less bow
    verts = [A, cp_AB_inner, B, C, A]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(patches.PathPatch(
        Path(verts, codes), facecolor="#fce4ec", lw=0, alpha=0.40, zorder=0
    ))

    # --- 3) Lens between the two A–B curved edges ---
    # Use two control points—one for each arc (outer and inner)
    cp_AB_outer = (-0.20, 1.30)  # roughly your opposite-curvature arc
    verts_lens_AB = [A, cp_AB_inner, B, cp_AB_outer, A]
    codes_lens = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    ax.add_patch(patches.PathPatch(
        Path(verts_lens_AB, codes_lens), facecolor="#f3e5f5", lw=0, alpha=0.35, zorder=0
    ))

    # --- 4) Lens between the two B–E curved edges ---
    # Two control points chosen to mimic your -0.50 and -0.30 arcs
    cp_BE_outer = (0.20, 0.05)   # “deeper” curve
    cp_BE_inner = (0.60, 0.22)   # “shallower” curve
    verts_lens_BE = [B, cp_BE_outer, E, cp_BE_inner, B]
    ax.add_patch(patches.PathPatch(
        Path(verts_lens_BE, codes_lens), facecolor="#e0f7fa", lw=0, alpha=0.30, zorder=0
    ))

    # =========================
    #  EDGES
    # =========================

    # Straight edges
    edge_art = nx.draw_networkx_edges(G, pos, width=2, edge_color="black", ax=ax)
    if isinstance(edge_art, list):
        for coll in edge_art:
            coll.set_zorder(1)
    else:
        edge_art.set_zorder(1)

    # Curved edges (same as your annotate calls), kept ABOVE fills
    def arc(p_to, p_from, rad):
        a = ax.annotate(
            "", xy=p_to, xycoords='data', xytext=p_from, textcoords='data',
            arrowprops=dict(arrowstyle="-", color="black",
                            connectionstyle=f"arc3,rad={rad}", linewidth=2),
            annotation_clip=False
        )
        a.set_zorder(2)

    # A<->B (two opposite curvatures)
    arc(A, B,  0.30)
    arc(A, B, -0.30)
    # B<->E (two negative curvatures)
    arc(E, B, -0.50)
    arc(E, B, -0.30)
    # B<->D (single curved)
    arc(D, B,  0.30)

    # =========================
    #  NODES + LABELS
    # =========================
    mapping = {12: 'A', 13: 'B', 9: 'C', 10: 'D', 11: 'E'}
    G = nx.relabel_nodes(G, mapping)
    new_pos = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}

    node_art = nx.draw_networkx_nodes(
        G, new_pos, node_size=1400, node_color="lightcoral",
        edgecolors="black", linewidths=2, ax=ax
    )
    node_art.set_zorder(3)

    labels = nx.draw_networkx_labels(G, new_pos, font_size=12, ax=ax)
    for t in labels.values():
        t.set_zorder(4)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_dual_p2_hardcoded()





    
example_dual()
example_dual_p2()