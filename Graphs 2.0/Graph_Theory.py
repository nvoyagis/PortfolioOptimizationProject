import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cvxpy
import sklearn
# import riskfolio as rf




def get_weight_mat(stock_list: list[str]):
    # Initialize variables and sort stocks.
    mapping = {}
    complete = nx.complete_graph(len(stock_list))
    percent_changes = pd.DataFrame()
    stock_list.sort()

    # Label complete graph & create a percent change DataFrame.
    for stock in stock_list:
        # Create DataFrame for a given stock.
        df = pd.read_csv(f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv', parse_dates=['Date'])
        # Remove spaces in column names.
        df.columns = df.columns.str.strip()
        # Remove some recent data to analyze profits in the past
        cutoff_date = pd.to_datetime('2023-01-01')
        df = df[df['Date'] <= cutoff_date]
        # Calculate daily percent change for a stock and add it as a new column to df.
        df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
        # Store percent changes into a single DataFrame, with each column representing a different stock.
        percent_changes[stock] = df['Percent Change']
        # Append stock names or indices to a dictionary to make a label the nodes of a complete graph.
        mapping[stock_list.index(stock)] = stock
    g = nx.relabel_nodes(complete, mapping)

    # Compute covariance matrix and its upper triangular part. Make cov_mat positive. 
    cov_mat = percent_changes.cov()
    upper_cov_mat = np.triu(cov_mat)
    np.fill_diagonal(upper_cov_mat, 0)
    # pos_mask = upper_cov_mat > 0
    # cov_mat = cov_mat.abs()
    # TODO: MARK STOCKS THAT ARE CONNECTED BY NEGATIVE COVARIANCES 
    upper_cov_mat = 0.0001 + (1 - 2 * 0.0001) * (upper_cov_mat - np.min(upper_cov_mat))/(np.max(upper_cov_mat) - np.min(upper_cov_mat)) # Normalize covariance to be between [1/1000, 999/1000]
    # pos_upper_cov_mat = upper_cov_mat[pos_mask == True]
    upper_cov_mat_df = pd.DataFrame(upper_cov_mat)
    #print(upper_cov_mat_df)

    # Filter out the zeros of upper triangular matrix.
    nonzero_values = upper_cov_mat_df[upper_cov_mat_df != 0.0001].stack()
    print('Approximately 0 covariance: ' + str(np.min(nonzero_values)))

    # Add (nonzero) weights to complete graph.
    for i in range(len(stock_list)):
        for j in range(len(stock_list)):
            if i < j:
                g[stock_list[i]][stock_list[j]]['weight'] = upper_cov_mat_df.iloc[i, j]

    w = nx.adjacency_matrix(g, weight='weight')


    # Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(g, weight='weight')

    # pos0 = nx.spring_layout(g)
    # nx.draw_networkx_nodes(g, pos0, node_color='lightblue', node_size=1000)
    # nx.draw_networkx_labels(g, pos0, font_size=11, font_family='sans-serif')
    # nx.draw_networkx_edges(g, pos0, width=2)
    # plt.show()


    # # Determine MST layout

    # pos = nx.spiral_layout(g)
    # pos = nx.spectral_layout(g)
    # pos = nx.spring_layout(g)
    # pos = nx.kamada_kawai_layout(g)
    # pos = nx.shell_layout(g)
    # pos = nx.spectral_layout(g)
    # pos = nx.fruchterman_reingold_layout(g)
    pos = nx.forceatlas2_layout(g)

    # Draw nodes and their labels.
    nx.draw_networkx_nodes(mst, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(mst, pos, font_size=11, font_family='sans-serif')

    # Draw edges
    nx.draw_networkx_edges(mst, pos, width=2)

    # Compute quartiles for the nonzero values.
    quartiles = nonzero_values.quantile([0.25, 0.5, 0.75])
    print(quartiles)

    # Create a dictionary for edge weights of MST.
    edge_labels_mst = nx.get_edge_attributes(mst, 'weight')
    weight_list_mst = list(edge_labels_mst.values())
    edge_list_mst = list(edge_labels_mst.keys())

    # Assign quartile labels to edges based on their weights to make graph readable.
    for weight in weight_list_mst:
        if weight < quartiles[0.25]:
            edge_labels_mst[edge_list_mst[weight_list_mst.index(weight)]] = '1'
        elif weight < quartiles[0.5]:
            edge_labels_mst[edge_list_mst[weight_list_mst.index(weight)]] = '2'
        elif weight < quartiles[0.75]:
            edge_labels_mst[edge_list_mst[weight_list_mst.index(weight)]] = '3'
        else:
            edge_labels_mst[edge_list_mst[weight_list_mst.index(weight)]] = '4'

    # Draw weights.
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels_mst, font_color='red')
    plt.show()

    # Compute weight matrix for complete graph.
    edge_labels_complete = nx.get_edge_attributes(g, 'weight')
    weight_list_complete = list(edge_labels_complete.values())
    edge_list_complete = list(edge_labels_complete.keys())

    # Assign quartile labels to edges based on their weights to make graph readable.
    for weight in weight_list_complete:
        if weight < quartiles[0.25]:
            edge_labels_complete[edge_list_complete[weight_list_complete.index(weight)]] = '1'
        elif weight < quartiles[0.5]:
            edge_labels_complete[edge_list_complete[weight_list_complete.index(weight)]] = '2'
        elif weight < quartiles[0.75]:
            edge_labels_complete[edge_list_complete[weight_list_complete.index(weight)]] = '3'
        else:
            edge_labels_complete[edge_list_complete[weight_list_complete.index(weight)]] = '4'

    # Create a new graph by assigning the quartile of each edge of g to the corresponding edge of the new graph.
    g_quartiles = g.copy()
    x_counter = 0
    for x in g_quartiles.nodes():
        x_counter += 1
        y_counter = 0
        for y in g_quartiles.nodes():
            y_counter += 1
            if x_counter > y_counter:
                g_quartiles[x][y]['weight'] = int(edge_labels_complete[(y, x)])

    # Create and print the adjacency matrix of the new graph.
    w_g_quartiles = nx.adjacency_matrix(g_quartiles, weight='weight')
    print(w_g_quartiles.todense())
    return w_g_quartiles.todense()




stocks = ['DIS', 'KO', 'ADBE', 'MRK', 'KMI', 'AAPL', 'JNJ', 'CVS', 'COST', 'T', 'BA', 'EA', 'HAS', 'HD', 'HSY', 'LLY', 'NFLX', 'NKE', 'V', 'JPM']
graph = get_weight_mat(stocks)