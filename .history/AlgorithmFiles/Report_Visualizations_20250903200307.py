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
import Database
import Charts
from itertools import combinations
# import AlgorithmFiles.Charts as Charts

def complete_graph(stock_dict):
    # Create complete graph w/ covariance using the selected start date
    complete_g = nx.complete_graph(n=len(stock_dict))
    complete_g = nx.relabel_nodes(complete_g, stock_dict)
    pos0 = nx.spring_layout(complete_g, scale=5, seed=1)
    nx.draw(complete_g, pos=pos0, node_color='#a0bec8', node_size=600)
    nx.draw(complete_g, pos=pos0, with_labels=True, node_color='#8fd6ff', edge_color='#a0bec8', node_size=650, font_size=8)
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
        1: (-0.20, 0.00),
        2: (1.20, 0.00),
        3: (0.50, 0.27),
        4: (0.50, 0.60),
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
        "A": (0.50, 0.12),  # face (1-2-3)
        "B": (0.35, 0.37),  # face (1-3-4)
        "C": (0.65, 0.37),  # face (2-3-4)
        "D": (0.35, 0.70),  # face (1-4-5)
        "E": (0.65, 0.70),  # face (2-4-5)
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
        ax.plot([x1,x2],[y1,y2], linewidth=2, color="#a0bec8", zorder=1)
    for v,(x,y) in P.items():
        ax.scatter([x],[y], s=450, edgecolors="black", color="#a0bec8", zorder=2)

    # ----- draw dual straight edges (red, solid) -----
    for a,b in dual_edges:
        x1,y1 = D[a]; x2,y2 = D[b]
        ax.plot([x1,x2],[y1,y2], linewidth=2, color="#de8585", zorder=3)

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
    ef_spline = PathPatch(Path(verts, codes), lw=2, edgecolor="#de8585",
                          facecolor="none", zorder=3)
    ax.add_patch(ef_spline)

    # ----- draw dual curved AF around node 1 (red, solid) -----
    xE,yE = D["A"]
    xF,yF = D["F"]
    x1,y1 = P[1]

    # Two control points placed to "wrap" above node 1.
    # Tweak these to change how tightly the spline bends around 1.
    c1 = (x1 + 0.20, y1 - 0.40)   # just above/right of 1
    c2 = (x1 - 0.40, y1 - 0.25)   # above/left toward F

    verts = [(xE,yE), c1, c2, (xF,yF)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ef_spline = PathPatch(Path(verts, codes), lw=2, edgecolor="#de8585",
                          facecolor="none", zorder=3)
    ax.add_patch(ef_spline)



        # ----- dual nodes (red) -----
    for f,(x,y) in D.items():
        ax.scatter([x],[y], s=450, edgecolors="black", color="#de8585", zorder=4)


    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig('high_res_image.png', dpi=300, pad_inches=0.2)
    plt.show()

def example_rbf_svm(df: pd.DataFrame,
                              feature_cols=None,
                              target_col=None,
                              threshold: float = 0.5,
                              scoring: str = "precision",
                              cv: int = 3,
                              n_jobs: int = -1,
                              verbose: int = 0,
                              title: str = "RBF SVM Classification"):
    """
    Train an RBF-kernel SVM on a 3-column DataFrame with GridSearchCV
    and plot the decision boundary.

    Parameters
    ----------
    df : DataFrame with exactly 3 columns (2 features + 1 numeric target score)
    feature_cols : list/tuple length 2 (optional). If None, uses the first two columns.
    target_col : str (optional). If None, uses the third column.
    threshold : float, label = (target > threshold)
    scoring : str, metric to optimize in GridSearchCV
    cv : int, number of cross-validation folds
    n_jobs : int, parallel jobs
    verbose : int, verbosity for GridSearchCV
    title : str, plot title

    Returns
    -------
    best_model : fitted sklearn Pipeline (MinMaxScaler -> SVC with best params)
    best_params : dict, best parameter set from GridSearchCV
    fig, ax : matplotlib objects
    """
    # --- Select columns
    if feature_cols is None:
        feature_cols = list(df.columns[:2])
    if target_col is None:
        target_col = df.columns[2]

    # --- Build X, y
    X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = (pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values > threshold).astype(int)

    # --- Define pipeline
    pipe = sklearn.pipeline.Pipeline([
        ("scaler", sklearn.preprocessing.MinMaxScaler()),
        ("svc", sklearn.svm.SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])

    # --- Parameter grid (as provided)
    param_grid = {
        'svc__C':     [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 10],
        'svc__gamma': ['scale', 'auto', 0.1, 0.5, 1, 5, 10],
        'svc__kernel': ['rbf'],
        'svc__degree': [1],   # ignored by RBF
        'svc__coef0':  [0],  # ignored by RBF
    }

    # --- Grid search
    gs = sklearn.model_selection.GridSearchCV(pipe,
                      param_grid=param_grid,
                      scoring=scoring,
                      cv=cv,
                      n_jobs=n_jobs,
                      verbose=verbose,
                      refit=True)
    gs.fit(X, y)
    best_model = gs.best_estimator_
    best_params = gs.best_params_
    print("Best params:", best_params)
    print(f"Best CV {scoring}: {gs.best_score_:.3f}")

    # --- Prepare grid for decision regions
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    pad_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    pad_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, 400),
        np.linspace(y_min - pad_y, y_max + pad_y, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = best_model.predict(grid).reshape(xx.shape)

    # --- Plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.contourf(xx, yy, Z, alpha=0.40,
                levels=[-0.5,0.5,1.5],
                colors=["#a0bec8","#de8585"])  # blue/red regions
    # scatter points
    ax.scatter(X[y==0,0], X[y==0,1], s=145, edgecolor="k",
               facecolor="#de8585", label="Unsuccessful Portfolio")
    ax.scatter(X[y==1,0], X[y==1,1], s=145, edgecolor="k",
               facecolor="crimson", label="Successful Portfolio")

    ax.set_xlabel('Historical Percent Change', fontsize=22)
    ax.set_ylabel('Second Order Centrality', fontsize=22)
    ax.set_title('SVC Using an Optimized Gaussian Kernel', fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    plt.savefig("svm_classification.png", dpi=300, bbox_inches='tight')


    return best_model, best_params, fig, ax











    
# example_dual()
# example_dual_p2()



# df1 = Database.table_into_df('stocks:30 1:2024-01-02 2:2024-01-31 3:2024-02-01 4:2024-02-29')


# # Get portfolios from df1
# portfolio_dict = df1.set_index("portfolio")["SPX_win_percentage"].to_dict()
# highlight_dict = df1.set_index("portfolio")["made_from_freq_edge"].to_dict()

# Charts.make_bar_graph(portfolio_dict, 'Portfolio', 'Probability of Outperforming the S&P 500', 'Probability of Outperforming the S&P 500 per Portfolio', sims=1, highlight_dict=highlight_dict)


df2 = Database.get_cols('stocks:30 1:2024-01-02 2:2024-01-31 3:2024-02-01 4:2024-02-29', ['vert_weight', 'harmonic', 'SPX_win_percentage'])
print(df2)
example_rbf_svm(df2)