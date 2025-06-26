import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
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
import Charts
import random
import ta
import Simulations

# NOTE: Not done
def optimize_hot_stocks(stocks: list[str], date_tuples: list[tuple]):
    '''
    - date_tuples contains 4-tuples of date1, date2, date3, date4. Choose one tuple.
    - Choose a value n to highlight portfolios with the top n% of stocks
    - For a tuple, take all dates within [date3, date4] and count the number of times highlighted portfolios (H) and non-highlighted (N) are above SPX
    - Divide H by the total amount of highlighted portfolios and N by the total amount of non-highlighted portoflios
    - Repeat last 2 steps for each tuple
    - Repeat all steps for each n
    '''
    n_dict = {}
    for n in range(100, 80):
        highlighted_wins = []
        nonhighlighted_wins = []
        for date_tuple in date_tuples:
            date1 = date_tuple[0]
            date2 = date_tuple[1]
            date3 = date_tuple[2]
            date4 = date_tuple[3]
            stock_dict = {}
            for i in range(len(stocks)):
                stock_dict[i] = stocks[i]

                # Create TMFG
                cov_mat = Graph_Theory_Functions.get_weight_mat(stocks, date1, date2)
                cov_mat = pd.DataFrame(cov_mat)
                model = fast_tmfg.TMFG()
                w = pd.DataFrame(np.ones((len(stocks), len(stocks))))
                cliques, seps, adj_matrix = model.fit_transform(weights=cov_mat, output='weighted_sparse_W_matrix') # This only works with Pandas dataframes for whatever reason
                g = nx.from_numpy_array(adj_matrix)

                # Add node weights to TMFG. These weights are the percent change of a stock from begin_data_date to buy_date. Color nodes by their weight.
                stock_percent_changes = {}
                for s in stocks:
                    df = pd.read_csv(
                                    f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                                    parse_dates=['Date'],
                                    date_format='%m/%d/%y')
                    df.set_index('Date', inplace=True)
                    open_value = df.loc[pd.Timestamp(date1), ' Open']
                    close_value = df.loc[pd.Timestamp(date2), ' Close']
                    stock_percent_changes[s] = (close_value - open_value) / open_value * 100
                g = nx.relabel_nodes(g, stock_dict)
                # Assign weights (percent changes) to nodes/stocks
                nx.set_node_attributes(g, stock_percent_changes, name='weight')


                '''
                1. Take node value
                2. Diffuse it across adjacent edges based off of edge weights. Keep some of it contained at the original node.
                3. Display graph using new node values
                '''
                # Calculate ADX for all stocks. ADX measures the strength of a stock's trend.
                # Convert strings to datetime objects
                d1 = pd.to_datetime(date1)
                d2 = pd.to_datetime(date2)

                # Calculate the difference in days
                day_difference = abs((d2 - d1).days)
                ADX_dict = {}
                for stock in stocks:
                    df = pd.read_csv(
                                    f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv',
                                    parse_dates=['Date'],
                                    date_format='%m/%d/%y')
                    df['ADX'] = ta.trend.adx(df[' High'], df[' Low'], df[' Close'], window=day_difference+1)
                    df['+DI'] = ta.trend.adx_pos(df[' High'], df[' Low'], df[' Close'], window=day_difference+1)
                    df['-DI'] = ta.trend.adx_neg(df[' High'], df[' Low'], df[' Close'], window=day_difference+1)
                    print(df.min())
                    print(df.max())
                    df.set_index('Date', inplace=True)
                    ADX_dict[stock] = df.loc[date2, 'ADX']


                def normalize_dict(data: dict) -> dict:
                    """
                    Normalize the values of a dictionary to the range [0, 1].
                    """
                    values = list(data.values())
                    min_val = min(values)
                    max_val = max(values)
                    range_val = max_val - min_val
                    normalized = {}

                    if range_val == 0:
                        # All values are the same --> map everything to 0.5
                        for key in data:
                            normalized[key] = 0.5

                    else:
                        for k, v in data.items():
                            normalized[k] = (v - min_val) / range_val
                    
                    return normalized

                # Normalize ADX_dict to be in [0, 1]
                ADX_dict = normalize_dict(ADX_dict)

                # Calculate "heat flow" out of each stock. Flow from one stock to another depends on the edge weight between them AND the total edge weight from the source stock.
                directed_edges = {}
                for v in g.nodes:
                    total_local_edge_weight = 0
                    for e in nx.edges(g, nbunch=v):
                        u = e[1]
                        total_local_edge_weight += g[v][u]['weight']
                    for e in nx.edges(g, nbunch=v):
                        u = e[1]
                        directed_edges[(v, u)] = g[v][u]['weight'] / total_local_edge_weight
                
                # Create a new TMFG after heat flows across edges
                predicting_TMFG = g.copy()

                # Recalculate node weights
                for v in predicting_TMFG.nodes:
                    predicting_TMFG.nodes[v]['weight'] = 0
                
                # For each node v, calculate total incoming heat from each adjacent node, u
                for v in predicting_TMFG.nodes:
                    for e in nx.edges(predicting_TMFG, nbunch=v):
                        u = e[1]
                        '''
                        Formula: 1) g.nodes[u]['weight']: Heat from an adjacent node u
                                2) (1 - ADX_dict[u]): More signficant trend => (1 - ADX_dict[u]) approaches 0 => We care more about the stock with the strong trend instead of its adjacent stocks
                                3) directed_edges[(u, v)]: Percentage of POSSIBLE heat from u that goes into the direction of v. Possible heat: (1 - ADX_dict[u]) * g.nodes[u]['weight']
                        '''
                        predicting_TMFG.nodes[v]['weight'] += directed_edges[(u, v)] * (1 - ADX_dict[u]) * g.nodes[u]['weight']
                
                # Add back each node's retained heat. This heat is was not shared with other nodes since ADX_dict[v] is within [0, 1].
                for v in predicting_TMFG.nodes:
                    predicting_TMFG.nodes[v]['weight'] += ADX_dict[v] * g.nodes[v]['weight']

                # Create new colors for Predicting TMFG
                # Extract weights
                weights = nx.get_node_attributes(predicting_TMFG, 'weight')
                weight_values = list(weights.values())

                # Normalize weights to [0, 1] for color mapping
                norm = colors.Normalize(vmin=min(weight_values), vmax=max(weight_values))
                cmap = cm.Reds


                # Make dual graph of TMFG
                triads = model.triangles
                dual = Dual.build_dual_from_faces(triads)

                face_list = {}
                for node in dual.nodes:
                    new_node = tuple(stock_dict[i] for i in node)
                    face_list[node] = new_node
                print(face_list)

                # Relabel nodes properly
                dual = nx.relabel_nodes(dual, face_list)

                # Fix node labels
                labels = {
                    node: f'({", ".join(node)})'
                    for node in dual.nodes
                }

                # Draw dual graph
                plt.figure(figsize=(12, 8))
                pos = nx.kamada_kawai_layout(dual, scale=3)
                nx.draw_networkx_edges(dual, pos, width=2)
                nx.draw(dual, pos, node_color='#a23e3e', node_size=650)
                nx.draw(dual, pos, with_labels=True, labels=labels, node_color='#de8585', edge_color='#a23e3e', node_size=600, font_size=6)
                # plt.title('Dual Graph')
                plt.axis('off')
                plt.tight_layout()
                # plt.show()
                
                # Generate random sell dates using a random stock (doesn't matter which one since all stocks use the same dates)
                df = pd.read_csv(
                        f'Data2015-2025/HistoricalPrices 2015 - 2025, {stocks[0]}.csv',
                        parse_dates=['Date'],
                        date_format='%m/%d/%y'
                    )
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')

                # Convert sell range strings to timestamps
                sell1 = pd.Timestamp(sell1)
                sell2 = pd.Timestamp(sell2)

                # Filter rows between sell1 and sell2
                filtered_sell_df = df[df['Date'] >= sell1]
                filtered_sell_df = filtered_sell_df[filtered_sell_df['Date'] <= sell2]
                filtered_sell_df.index = pd.to_datetime(filtered_sell_df.index)

                # Get all valid dates in that range
                filtered_sell_df.set_index('Date', inplace=True)
                valid_dates = filtered_sell_df.index

                # Create dictionaries to store information about portfolios
                SPX_wins = {}
                avg_returns_dict = {}
                allocations = {}

                # Run variance-minimized simulations of triads and count wins
                for node in dual.nodes:


                    # Find variance-minimizing portfolio allocations
                    percent_changes = pd.DataFrame()
                    for s in node:
                        df = pd.read_csv(
                                        f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                                        parse_dates=['Date'],
                                        date_format='%m/%d/%y')
                        df = df[df['Date'] <= date2]
                        df = df[df['Date'] >= date1]
                        # Calculate daily percent change for a stock and add it as a new column to df.
                        df['Percent Change'] = ((df[' Close'] - df[' Open']) / df[' Open']) * 100
                        percent_changes[s] = df['Percent Change']
                    small_cov_mat = percent_changes.cov()
                    small_cov_mat = pd.DataFrame(small_cov_mat)
                    print(small_cov_mat)
                    # Calculate variance-minimizing stock allocations
                    mat = np.array(2*small_cov_mat)
                    mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
                    mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
                    mat[-1][-1] = 0
                    mat_inv = np.linalg.inv(mat)
                    B = np.zeros((mat.shape[1], 1))
                    B[-1] = 1
                    stock_allocations = np.dot(mat_inv, B)
                    # Remove last row of stock_allocations and retrieve its important values
                    stock_allocations = np.delete(stock_allocations, -1, axis=0)
                    x = np.dot(np.transpose(stock_allocations), small_cov_mat)
                    variance = np.dot(x, stock_allocations)
                    print(f'Portfolio minimum variance: {variance[0][0]}')
                    print(f'Portfolio allocations:  {node[0]}: {stock_allocations[0][0]}\n                       {node[1]}: {stock_allocations[1][0]}\n                       {node[2]}: {stock_allocations[2][0]}')
                    allocations[node] = [stock_allocations[0][0], stock_allocations[1][0], stock_allocations[2][0]]
