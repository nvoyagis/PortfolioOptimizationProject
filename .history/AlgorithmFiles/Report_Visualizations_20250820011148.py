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

# ---------- helpers that MATCH arc3 exactly ----------
def _cp_arc3_display(ax, p1, p2, rad):
    """
    Control point for matplotlib's connectionstyle='arc3,rad=...'.
    Compute in DISPLAY coords (pixels), then convert back to DATA coords,
    so it matches the actual curved edges drawn by annotate/FancyArrowPatch.
    """
    # to display coords
    P1 = ax.transData.transform(p1)
    P2 = ax.transData.transform(p2)
    mx, my = (P1 + P2) / 2.0
    dx, dy = (P2 - P1)
    # rotate 90°: (dx,dy)->(-dy,dx); rad is in 'arc3' units (display)
    Cx, Cy = mx - rad * dy, my + rad * dx
    # back to data coords
    return tuple(ax.transData.inverted().transform((Cx, Cy)))

def add_lens_between_two_arcs(ax, p1, p2, rad1, rad2, color, alpha=0.35, zorder=0):
    """
    Fill the region bounded by two arc3 curves between p1<->p2 with curvature rad1 and rad2.
    """
    c1 = _cp_arc3_display(ax, p1, p2, rad1)
    c2 = _cp_arc3_display(ax, p2, p1, rad2)  # reverse direction for the second arc
    verts = [p1, c1, p2, c2, p1]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    patch = patches.PathPatch(Path(verts, codes), facecolor=color, lw=0, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

def add_curved_triangle(ax, p1, p2, p3, rad12, color, alpha=0.35, zorder=0):
    """
    Triangle where side p1-p2 is an arc3 curve; p2->p3 and p3->p1 are straight.
    """
    c = _cp_arc3_display(ax, p1, p2, rad12)
    verts = [p1, c, p2, p3, p1]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    patch = patches.PathPatch(Path(verts, codes), facecolor=color, lw=0, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

# ---------- main figure ----------
def example_dual_p2_colored():
    G = nx.MultiGraph()
    G.add_edges_from([
        (13, 12), (13, 9), (13, 10), (13, 11),
        (9, 10), (9, 11), (10, 11)
    ])

    pos = {
        12: (0.3, 1.5),   # A
        13: (0.0, 0.8),   # B
        9:  (0.35, 0.45), # C
        10: (-0.2, 0.3),  # D
        11: (1.0, 0.3),   # E
    }

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    # IMPORTANT: keep geometry predictable
    ax.set_aspect('equal', adjustable='datalim')

    # --- (0) Fills FIRST (lowest zorder)
    # Straight-edge triangular faces
    straight_faces = [
        [pos[13], pos[9],  pos[10]],  # B-C-D
        [pos[13], pos[9],  pos[11]],  # B-C-E
        [pos[9],  pos[10], pos[11]],  # C-D-E
    ]
    straight_colors = ["#e1f5fe", "#e8f5e9", "#fff3e0"]
    for f, col in zip(straight_faces, straight_colors):
        ax.add_patch(patches.Polygon(f, closed=True, facecolor=col, edgecolor="none",
                                     alpha=0.45, zorder=0))

    # Curved triangle (A–B arc + lines to C)
    add_curved_triangle(ax, pos[12], pos[13], pos[9], rad12=0.30, color="#fce4ec", alpha=0.40, zorder=0)

    # Lenses between parallel arcs
    add_lens_between_two_arcs(ax, pos[12], pos[13], rad1=0.30, rad2=-0.30, color="#f3e5f5", alpha=0.35, zorder=0)
    add_lens_between_two_arcs(ax, pos[13], pos[11], rad1=-0.50, rad2=-0.30, color="#e0f7fa", alpha=0.30, zorder=0)

    # --- (1) Straight edges
    edge_art = nx.draw_networkx_edges(G, pos, width=2, edge_color="black")
    if isinstance(edge_art, list):
        for coll in edge_art:
            coll.set_zorder(1)
    else:
        edge_art.set_zorder(1)

    # --- (2) Curved edges (above fills)
    def add_arc(p_to, p_from, rad):
        a = ax.annotate(
            "", xy=p_to, xycoords='data', xytext=p_from, textcoords='data',
            arrowprops=dict(arrowstyle="-", color="black",
                            connectionstyle=f"arc3,rad={rad}", linewidth=2),
            annotation_clip=False
        )
        a.set_zorder(2)

    # A<->B
    add_arc(pos[12], pos[13],  0.30)
    add_arc(pos[12], pos[13], -0.30)
    # B<->E
    add_arc(pos[11], pos[13], -0.50)
    add_arc(pos[11], pos[13], -0.30)
    # B<->D
    add_arc(pos[10], pos[13],  0.30)

    # --- (3) Relabel & draw nodes
    mapping = {12: 'A', 13: 'B', 9: 'C', 10: 'D', 11: 'E'}
    G = nx.relabel_nodes(G, mapping)
    new_pos = {'A': pos[12], 'B': pos[13], 'C': pos[9], 'D': pos[10], 'E': pos[11]}

    node_art = nx.draw_networkx_nodes(
        G, new_pos, node_size=1400, node_color="lightcoral",
        edgecolors="black", linewidths=2
    )
    node_art.set_zorder(3)

    # --- (4) Labels
    label_art = nx.draw_networkx_labels(G, new_pos, font_size=12)
    for text in label_art.values():
        text.set_zorder(4)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# Run
if __name__ == "__main__":
    example_dual_p2_colored()




    
example_dual()
example_dual_p2()