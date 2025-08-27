import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
import scipy as sp
import cvxpy
import fast_tmfg
import Graph_Theory_Functions
import Dual
import os
import time
from itertools import combinations
import Charts
import random
import ta
# import riskfolio as rf
import ML_Analysis
import Database
import TMFG_Analysis

def normalize_dict(data: dict) -> dict:
    """
    Normalize the values of a dictionary to the range [0.001, 0.999].
    """
    values = list(data.values())
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        # All values are the same --> map everything to 0.5
        return {k: 0.5 for k, v in data.items()}
    
    return {k: 0.001 + (v - min_val) * (0.999 - 0.001) / (range_val) for k, v in data.items()}

def sort_by_growth(stock_list: list[str], day1: str, day2: str):
    growth_tracker = {}
    # Calculate the growth of each stock between day1 and day2
    for stock in stock_list:
        # Create DataFrame for a given stock
        df = pd.read_csv(f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
        df['Date'] = pd.to_datetime(df['Date'])
        # Remove spaces in column names
        df.columns = df.columns.str.strip()

        day1 = pd.to_datetime(day1)
        day2 = pd.to_datetime(day2)

        day1_price = (df.loc[df['Date'] == day1, 'Open']).iloc[0]
        day2_price = (df.loc[df['Date'] == day2, 'Close']).iloc[0]
        change = (day2_price - day1_price) / day1_price
        growth_tracker[stock] = change

    # reverse=False helps high-growth stocks achieve high centralities
    # reverse=True helps low-growth stocks achieve high centralities
    sorted_stocks = sorted(growth_tracker, key=growth_tracker.get, reverse=False)
    print(growth_tracker)
    print(sorted_stocks)
    return sorted_stocks

    

# def analyze_ewm_deviation(stock: str, end_date: str, span=30):
#     results = []

#     df = pd.read_csv(
#                     f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv',
#                     parse_dates=['Date'],
#                     date_format='%m/%d/%y')
#     df = df[df['Date'] <= end_date]
#     df['Percent Change'] = ((df[' Close'] - df[' Open']) / df[' Open']) * 100
#     data = df['Percent Change']

#     # Estimate trend
#     ewm = data.ewm(span=span, adjust=False).mean()

#     residuals = data - ewm
#     residuals = residuals.dropna()

#     mad = np.mean(np.abs(residuals))
#     rmse = np.sqrt(np.mean(residuals ** 2))
#     std_dev = np.std(residuals)
#     percent_dev = np.mean(np.abs(residuals / ewm.dropna()))

#     results.append({
#         'Stock': stock,
#         'MAD': mad,
#         'RMSE': rmse,
#         'STD_DEV': std_dev,
#         'PCT_DEV': percent_dev
#     })

#     results_df = pd.DataFrame(results).set_index('Stock')
#     return results_df

# deviation_stats = analyze_ewm_deviation('AAPL', span=20)
# print(deviation_stats)

def make_TMFG(stocks: list[str], begin_data_date: str, buy_date: str):
    date_dict = {}
    date_dict['d1'] = begin_data_date
    date_dict['d2'] = buy_date

    # Initialize data
    # NOTE: the order of stocks doesn't affect the dual graph. It only affects the position of the TMFG's nodes.
    stocks = sort_by_growth(stocks, begin_data_date, buy_date)
    stock_dict = {}
    for i in range(len(stocks)):
        stock_dict[i] = stocks[i]

    # Create TMFG
    cov_mat = Graph_Theory_Functions.get_weight_mat(stocks, begin_data_date, buy_date)
    cov_mat = pd.DataFrame(cov_mat)
    model = fast_tmfg.TMFG()
    w = pd.DataFrame(np.ones((len(stocks), len(stocks))))
    cliques, seps, adj_matrix = model.fit_transform(weights=cov_mat, output='weighted_sparse_W_matrix') # This only works with Pandas dataframes for whatever reason
    g = nx.from_numpy_array(adj_matrix)

    # Add node weights to TMFG. These weights are the percent change of a stock from begin_data_date to buy_date. Color nodes by their weight.
    stock_percent_changes = {}
    stock_dfs = {}
    for s in stocks:
        df = pd.read_csv(
                        f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                        parse_dates=['Date'],
                        date_format='%m/%d/%y')
        df = df[df['Date'] <= buy_date]
        df = df[df['Date'] >= begin_data_date]
        df.columns = df.columns.str.strip()
        stock_dfs[s] = df
        df = df[df['Date'] <= buy_date]
        df.set_index('Date', inplace=True)
        open_value = df.loc[pd.Timestamp(begin_data_date), 'Open']
        close_value = df.loc[pd.Timestamp(buy_date), 'Close']
        stock_percent_changes[s] = (close_value - open_value) / open_value * 100

    g = nx.relabel_nodes(g, stock_dict)
    # Assign weights (percent changes) to nodes/stocks
    nx.set_node_attributes(g, stock_percent_changes, name='weight')
    # Map node weight to colors
    node_values = list(stock_percent_changes.values())
    norm = plt.Normalize(vmin=min(node_values), vmax=max(node_values))
    cmap = cm.Reds
    node_colors = []
    for value in node_values:
        node_colors.append(cmap(norm(value)))
    # Draw TMFG
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('TMFG')
    pos0 = nx.planar_layout(g, scale=2)
    # nx.draw(g, pos=pos0, node_color='#5192b8', node_size=650)
    # nx.draw(g, pos=pos0, with_labels=True, node_color='#8fd6ff', edge_color='#5192b8', node_size=600, font_size=8)
    nx.draw(g, pos=pos0, with_labels=True, node_color=node_colors, edge_color='#5192b8', node_size=600, font_size=8, ax=ax)
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(g, 'weight')
    for k, v in edge_labels.items():
        edge_labels[k] = round(v, 2)
    nx.draw_networkx_edge_labels(g, pos=pos0, edge_labels=edge_labels, font_size=6, ax=ax)

    # Create colorbar based on the node color mapping
    sm = cm.ScalarMappable(cmap=cm.Reds, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("Stock Percent Change", fontsize=9)

    plt.tight_layout()
    # plt.show()
    return g



# Create TMFG for each simulation with a chosen buy date
def simulate(sims: int, seed: int, stocks: list[str], begin_data_date: str, buy_date: str, sell1: str, sell2: str, analyze_node_features = False, frequently_appearing_check = False):
    start_time = time.time()
    date_dict = {}
    date_dict['d1'] = begin_data_date
    date_dict['d2'] = buy_date
    date_dict['d3'] = sell1
    date_dict['d4'] = sell2
    # Store data as a new file
    with open(os.path.join('Simulations', f'{sims}_Sims2_{begin_data_date}_to_{buy_date}_and_{sell1}_to_{sell2}.txt'), 'w', newline='') as f:
        f.write(f'Stocks ({len(stocks)}): {stocks}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Simulations: {sims}\n')
        f.write(f'Starting date: {begin_data_date} to {buy_date}\n')
        f.write(f'Ending date range: {sell1} to {sell2}\n')
        f.write('------------------------------------------------------\n')

        # Initialize data
        # NOTE: the order of stocks doesn't affect the dual graph. It only affects the position of the TMFG's nodes.
        stocks = sort_by_growth(stocks, begin_data_date, buy_date)
        stock_dict = {}
        for i in range(len(stocks)):
            stock_dict[i] = stocks[i]

        # Create TMFG
        cov_mat = Graph_Theory_Functions.get_weight_mat(stocks, begin_data_date, buy_date)
        cov_mat = pd.DataFrame(cov_mat)
        model = fast_tmfg.TMFG()
        w = pd.DataFrame(np.ones((len(stocks), len(stocks))))
        cliques, seps, adj_matrix = model.fit_transform(weights=cov_mat, output='weighted_sparse_W_matrix') # This only works with Pandas dataframes for whatever reason
        g = nx.from_numpy_array(adj_matrix)

        # Add node weights to TMFG. These weights are the percent change of a stock from begin_data_date to buy_date. Color nodes by their weight.
        stock_percent_changes = {}
        stock_dfs = {}
        for s in stocks:
            df = pd.read_csv(
                            f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                            parse_dates=['Date'],
                            date_format='%m/%d/%y')
            df = df[df['Date'] <= sell2]
            df = df[df['Date'] >= begin_data_date]
            df.columns = df.columns.str.strip()
            stock_dfs[s] = df
            df = df[df['Date'] <= buy_date]
            df.set_index('Date', inplace=True)
            open_value = df.loc[pd.Timestamp(begin_data_date), 'Open']
            close_value = df.loc[pd.Timestamp(buy_date), 'Close']
            stock_percent_changes[s] = (close_value - open_value) / open_value * 100

        g = nx.relabel_nodes(g, stock_dict)
        # Assign weights (percent changes) to nodes/stocks
        nx.set_node_attributes(g, stock_percent_changes, name='weight')
        # Map node weight to colors
        node_values = list(stock_percent_changes.values())
        norm = plt.Normalize(vmin=min(node_values), vmax=max(node_values))
        cmap = cm.Reds
        node_colors = []
        for value in node_values:
            node_colors.append(cmap(norm(value)))
        # Draw TMFG
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.title('TMFG')
        pos0 = nx.planar_layout(g, scale=2)
        # nx.draw(g, pos=pos0, node_color='#5192b8', node_size=650)
        # nx.draw(g, pos=pos0, with_labels=True, node_color='#8fd6ff', edge_color='#5192b8', node_size=600, font_size=8)
        nx.draw(g, pos=pos0, with_labels=True, node_color='#a0bec8', edge_color='#a0bec8', node_size=600, font_size=8, ax=ax)
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(g, 'weight')
        for k, v in edge_labels.items():
            edge_labels[k] = round(v, 2)
        nx.draw_networkx_edge_labels(g, pos=pos0, edge_labels=edge_labels, font_size=6, ax=ax)

        # Create colorbar based on the node color mapping
        sm = cm.ScalarMappable(cmap=cm.Reds, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
        cbar.set_label("Stock Percent Change", fontsize=9)

        plt.tight_layout()
        plt.show()


        # Draw TMFG coloring based on expected growth by modeling expected returns as heat flow.
        '''
        1. Take node value
        2. Diffuse it across adjacent edges based off of edge weights. Keep some of it contained at the original node.
        3. Display graph using new node values
        '''

        # Calculate ADX for all stocks. ADX measures the strength of a stock's trend.
        # Convert strings to datetime objects
        d1 = pd.to_datetime(begin_data_date)
        d2 = pd.to_datetime(buy_date)

        # Calculate the difference in days
        day_difference = abs((d2 - d1).days)
        ADX_dict = {}
        for stock in stocks:
            df = pd.read_csv(
                            f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv',
                            parse_dates=['Date'],
                            date_format='%m/%d/%y')
            df.set_index('Date', inplace=True)
            df.columns = df.columns.str.strip()
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=day_difference)
            df['+DI'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=day_difference)
            df['-DI'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=day_difference)
            print(df.min())
            print(df.max())
            ADX_dict[stock] = df.loc[buy_date, 'ADX']

        # Calculate standard deviation of each stock's closing prices
        std_dev_dict = {}
        for stock in stocks:
            df = stock_dfs[stock]
            std_dev_dict[stock] = df['Close'].std()

        # Normalize ADX_dict to be in [0, 1]
        ADX_dict = normalize_dict(ADX_dict)

        std_dev_dict = normalize_dict(std_dev_dict)
        print(std_dev_dict)

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

        # Create color list
        node_colors = [cmap(norm(weights[node])) for node in predicting_TMFG.nodes()]

        # Draw Predicting TMFG
        plt.figure(figsize=(12, 8))
        plt.title('Predicting TMFG')
        pos0 = nx.planar_layout(predicting_TMFG, scale=3)
        # nx.draw(predicting_TMFG, pos=pos0, node_color='#5192b8', node_size=650)
        # nx.draw(predicting_TMFG, pos=pos0, with_labels=True, node_color='#8fd6ff', edge_color='#5192b8', node_size=600, font_size=8)
        nx.draw(predicting_TMFG, pos=pos0, with_labels=True, node_color=node_colors, edge_color='#a0bec8', node_size=600, font_size=8)
        edge_labels = nx.get_edge_attributes(predicting_TMFG, 'weight')
        for key, weight in edge_labels.items():
            edge_labels[key] = round(weight, 2)
        nx.draw_networkx_edge_labels(predicting_TMFG, pos=pos0, edge_labels=edge_labels, font_size=6)
        plt.show()
        



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
        plt.title('Dual Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Generate random sell dates using a random stock (doesn't matter which one since all stocks use the same dates)
        df = pd.read_csv(f'Data2015-2025/HistoricalPrices 2015 - 2025, {stock}.csv', parse_dates=['Date'], date_format='%m/%d/%y')
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
        print(df)

        # Convert sell range strings to timestamps
        sell1 = pd.Timestamp(sell1)
        sell2 = pd.Timestamp(sell2)

        # Filter rows between sell1 and sell2
        filtered_sell_df = df[df['Date'] >= sell1]
        filtered_sell_df = filtered_sell_df[filtered_sell_df['Date'] <= sell2]
        print(filtered_sell_df)
        filtered_sell_df.index = pd.to_datetime(filtered_sell_df.index)
        print(filtered_sell_df)

        # Get all valid dates in that range
        filtered_sell_df.set_index('Date', inplace=True)
        print(filtered_sell_df)
        valid_dates = filtered_sell_df.index

        # Create dictionaries to store information about portfolios
        SPX_wins = {}
        avg_returns_dict = {}
        allocations = {}
        portfolio_historical_percent_changes = {}

        for s in stocks:
            df = stock_dfs[s]
            df.reset_index(inplace=True)

        # Run variance-minimized simulations of triads and count wins
        for node in dual.nodes:

            # Find variance-minimizing portfolio allocations
            percent_changes = pd.DataFrame()
            for s in node:
                df = stock_dfs[s]
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
                df = df[df['Date'] <= buy_date]
                df = df[df['Date'] >= begin_data_date]
                # Calculate daily percent change for a stock and add it as a new column to df.
                df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
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
            
            # Calculate the historical percent changes of all portfolios 
            portfolio_percent_change = 0
            for i in range(len(node)):
                portfolio_percent_change += allocations[node][i] * stock_percent_changes[node[i]]
            portfolio_historical_percent_changes[node] = portfolio_percent_change

        # Update dual TMFG vertex weights. They are the percent change of the portfolio between begin_data_date and buy_date.
        nx.set_node_attributes(dual, portfolio_historical_percent_changes, name='weight')
        weights = nx.get_node_attributes(dual, 'weight')

        # Update dual TMFG edge weights. They are the covariance between adjacent portfolios, normalized between [0.001, 0.999].
        dual_edge_weight_dict = {}
        portfolio_cov_df = pd.DataFrame()
        for node in dual.nodes:
            # Calculate daily percent change for a stock and add it as a new column to df.
            temp_stock_df_dict = {}
            for i in range(len(node)):
                df = stock_dfs[node[i]]
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
                df = df[df['Date'] <= buy_date]
                df = df[df['Date'] >= begin_data_date]
                df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
                temp_stock_df_dict[i] = df
            weight1, weight2, weight3 = allocations[node]
            pc1 = temp_stock_df_dict[0]['Percent Change']
            pc2 = temp_stock_df_dict[1]['Percent Change']
            pc3 = temp_stock_df_dict[2]['Percent Change']
            combined = weight1 * pc1 + weight2 * pc2 + weight3 * pc3
            combined.index = temp_stock_df_dict[0].index # dataframe neds to set the tuples as an index
            portfolio_cov_df[node] = combined

        dual_cov_mat = portfolio_cov_df.cov()
        dual_cov_mat = pd.DataFrame(dual_cov_mat)
        for e in dual.edges:
            node1, node2 = e
            dual_edge_weight_dict[e] = dual_cov_mat[node1][node2]
            dual_edge_weight_dict = normalize_dict(dual_edge_weight_dict)
            print(dual_edge_weight_dict)

        incindent_edge_sum_dict = {}
        for p_pairs in dual_edge_weight_dict.keys():
            incindent_edge_sum_dict[p_pairs[0]] = 0
            incindent_edge_sum_dict[p_pairs[1]] = 0
        for portfolio_pairs, edge_weight in dual_edge_weight_dict.items():
            incindent_edge_sum_dict[portfolio_pairs[0]] += edge_weight
            incindent_edge_sum_dict[portfolio_pairs[1]] += edge_weight

        for node in dual.nodes:
            SPX_beat_count = 0
            triad_percent_changes = []
            portfolio_returns = []

            for i in range(sims):
                # Pick a sell random date
                random_sell_date = np.random.choice(valid_dates)
                target_date2 = pd.Timestamp(random_sell_date)

                percent_changes = pd.DataFrame()
                random_period_percent_changes = []
                for s in node:
                    df = stock_dfs[s]

                    # Set Date as index
                    df.set_index('Date', inplace=True)

                    # Use pd.Timestamp for the date lookup
                    target_date1 = pd.Timestamp(buy_date)

                    # Get the opening value for a specific date
                    open_value = df.loc[target_date1, 'Open']
                    close_value = df.loc[target_date2, 'Close']
                    random_period_percent_changes.append((close_value - open_value)/open_value * 100)

                    # Change Date back into a column so it can be accessed as normal
                    df.reset_index(inplace=True)
                    # Remove some recent data to analyze profits in the past
                    cutoff_date = pd.to_datetime(buy_date)
                    df = df[df['Date'] <= cutoff_date]
                    # Calculate daily percent change for a stock and add it as a new column to df.
                    df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
                    # Store percent changes into a single DataFrame, with each column representing a different stock.
                    percent_changes[s] = df['Percent Change']
                
                random_period_portfolio_percent_change = 0
                for i in range(len(random_period_percent_changes)):
                    random_period_portfolio_percent_change += random_period_percent_changes[i] * stock_allocations[i][0]
                
                print(f'Portfolio percent change from {buy_date} to {random_sell_date}: ' + str(random_period_portfolio_percent_change))
                print('--------------------------------------------\n')

                portfolio_returns.append(random_period_portfolio_percent_change)
                

                df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
                    parse_dates=['Date'],
                    date_format='%m/%d/%y')
                df.columns = df.columns.str.strip()

                # Set Date as index
                df.set_index('Date', inplace=True)

                # Use pd.Timestamp for the date lookup
                target_date1 = pd.Timestamp(buy_date)
                target_date2 = pd.Timestamp(random_sell_date)

                open_value = df.loc[target_date1, 'Open']
                close_value = df.loc[target_date2, 'Close']
                SPX_percent_change = (close_value - open_value)/open_value * 100
                print(f'SPX percent change from {buy_date} to {random_sell_date}: ' + str(SPX_percent_change))
                print('--------------------------------------------\n')
                if random_period_portfolio_percent_change > SPX_percent_change:
                    SPX_beat_count += 1
                triad_percent_changes.append(random_period_portfolio_percent_change)
            
            avg_return = sum(triad_percent_changes)/len(triad_percent_changes)

            f.write(f'Portfolio: {node}\n')
            f.write(f'Average portfolio return: {avg_return}\n')
            f.write(f'Number of times SPX was beat: {SPX_beat_count}\n')
            f.write('------------------------------------------------------\n')

            SPX_wins[node] = SPX_beat_count
            avg_returns_dict[node] = avg_return

        stock_frequency = {}
        for s1 in stocks:
            stock_frequency[s1] = 0
            for node in dual.nodes:
                for s2 in node:
                    if s1 == s2:
                        stock_frequency[s1] += 1

        # Round all values in a dictionary to n decimal places
        def round_dict(d, n):
            return {k: round(v, n) for k, v in d.items()}
        
        

        # Determine node colors and border thickness based on SPX_wins and avg_returns_dict
        node_colors = []
        node_linewidths = []

        for node in dual.nodes():
            # Set color based on whether the portfolio 'wins' (> sims // 2)
            if SPX_wins.get(node, 0) > sims // 2:
                node_colors.append('crimson')    # Winning portfolios
            else:
                node_colors.append('lightcoral') # Losing portfolios

            # Set border width based on whether the return is positive
            if avg_returns_dict.get(node, 0) > 0:
                node_linewidths.append(2.5)  # Thicker border
            else:
                node_linewidths.append(0.8)  # Default thin border

        # Find portfolios with frequently-appearing stocks across 2 portfolios.
        if frequently_appearing_check:
            data_collection_length = d2 - d1
            partial_data_collection_length = 3* (data_collection_length.days // 4) # 1.5, 2, 4 already used
            days_rounded_for_weeks = round(partial_data_collection_length / 7) * 7
            previous_TMFG_date1 = d1 - pd.Timedelta(days=days_rounded_for_weeks)
            previous_TMFG_date2 = d2 - pd.Timedelta(days=days_rounded_for_weeks)
            previous_TMFG = make_TMFG(stocks, previous_TMFG_date1, previous_TMFG_date2)
            frequent_edge_list = TMFG_Analysis.find_frequent_edges([previous_TMFG, g])
            transparency_dict = {}
            for portfolio in dual.nodes():
                transparency_dict[portfolio] = 1
            for edge in frequent_edge_list:
                for portfolio in dual.nodes():
                    if edge[0] in portfolio and edge[1] in portfolio:
                        transparency_dict[portfolio] = 0.5
            node_transparencies = list(transparency_dict.values())

            with open(os.path.join(f'Repeating_Edges_Data_{begin_data_date}_{buy_date}_{sell1}_{sell2}'), 'w', newline='') as f2:
                # Determine % of successful portfolios with frequent edges (Avg Rtns)
                total_count = 0
                success_count = 0
                for i in range(len(node_transparencies)):
                    if node_transparencies[i] == 0.5:
                        total_count += 1
                        if node_linewidths[i] == 2.5:
                            success_count += 1
                f2.write('Average Returns (freq): ' + str(success_count/total_count) + '\n')

                # Determine % of successful portfolios with frequent edges (Beating SPX)
                total_count = 0
                success_count = 0
                for i in range(len(node_transparencies)):
                    if node_transparencies[i] == 0.5:
                        total_count += 1
                        if node_colors[i] == 'crimson':
                            success_count += 1
                f2.write('Likely to Beat SPX (freq): ' + str(success_count/total_count) + '\n')

                # Determine % of successful portfolios without frequent edges (Avg Rtns)
                total_count = 0
                success_count = 0
                for i in range(len(node_transparencies)):
                    if node_transparencies[i] == 1:
                        total_count += 1
                        if node_linewidths[i] == 2.5:
                            success_count += 1
                f2.write('Average Returns (non-freq): ' + str(success_count/total_count) + '\n')

                # Determine % of successful portfolios without frequent edges (Beating SPX)
                total_count = 0
                success_count = 0
                for i in range(len(node_transparencies)):
                    if node_transparencies[i] == 1:
                        total_count += 1
                        if node_colors[i] == 'crimson':
                            success_count += 1
                f2.write('Likely to Beat SPX (non-freq): ' + str(success_count/total_count) + '\n')

                # Determine % of successful portfolios overall (Avg Rtns)
                total_count = len(node_transparencies)
                success_count = 0
                for i in range(len(node_transparencies)):
                    if node_linewidths[i] == 2.5:
                        success_count += 1
                f2.write('Avg Returns (all): ' + str(success_count/total_count) + '\n')

                # Determine % of successful portfolios overall (Beating SPX)
                total_count = len(node_transparencies)
                success_count = 0
                for i in range(len(node_transparencies)):
                    if node_colors[i] == 'crimson':
                        success_count += 1
                f2.write('Likely to Beat SPX (all): ' + str(success_count/total_count) + '\n')
        return

        # Draw dual graph with labels
        plt.figure(figsize=(12, 8))
        pos = nx.kamada_kawai_layout(dual, scale=3)
        nx.draw_networkx_edges(dual, pos, width=2)
        nx.draw_networkx_edge_labels(dual, pos, edge_labels=round_dict(dual_edge_weight_dict, 2), font_size=6)
        nx.draw(
            dual, pos,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            alpha=node_transparencies,
            edge_color='gray',
            node_size=600,
            font_size=6,
            linewidths=node_linewidths,
            edgecolors='black'
        )
        plt.title('Dual Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


        # NOTE: before heat spread
        # Get node weights as a dictionary
        node_weights = nx.get_node_attributes(g, 'weight')

        # Calculate the nth percentile threshold
        weights = list(node_weights.values())
        threshold = np.percentile(weights, 89)

        # Select nodes whose weight is greater than or equal to the threshold
        hot_stocks1 = []
        for stock, weight in node_weights.items():
            if weight >= threshold:
                hot_stocks1.append(stock)

        # Charts.hot_stocks_in_dual_portfolios(hot_stocks1, 'Average Returns', avg_returns=avg_returns_dict)
        # Charts.hot_stocks_in_dual_portfolios(hot_stocks1, 'SPX Wins', SPX_wins=SPX_wins)


        # NOTE: after heat spread
        # Get node weights as a dictionary
        node_weights = nx.get_node_attributes(predicting_TMFG, 'weight')

        # Calculate the nth percentile threshold
        weights = list(node_weights.values())
        threshold = np.percentile(weights, 89)

        # Select nodes whose weight is greater than or equal to the threshold
        hot_stocks2 = []
        for stock, weight in node_weights.items():
            if weight >= threshold:
                hot_stocks2.append(stock)

        
        print(hot_stocks1)
        print(hot_stocks2)
        # Charts.compare_portfolios_to_SPX(allocations, hot_stocks1, begin_data_date, buy_date, sell1, sell2)
        # Charts.compare_portfolios_to_SPX(allocations, hot_stocks2, begin_data_date, buy_date, sell1, sell2)  
    
        # Charts.hot_stocks_in_dual_portfolios(hot_stocks2, 'Average Returns', avg_returns=avg_returns_dict)
        # Charts.hot_stocks_in_dual_portfolios(hot_stocks2, 'SPX Wins', SPX_wins=SPX_wins)


        # 1) Get edge weights of triangles
        # 2) Graph edge weights w/ winning portfolios
        stock_triads = face_list.values()
        edge_weight_dict = {}
        for three_clique_stocks in stock_triads:
            s = 0
            for u, v in combinations(three_clique_stocks, 2):
                s += g[u][v]['weight']
            edge_weight_dict[three_clique_stocks] = s
        # Charts.dual_edge_weights_single_bar_graph(edge_weight_dict, 'Edge Weights', 'Sum of TMFG Edge Weights of Dual Portfolios', avg_returns_dict)
        # Charts.dual_edge_weights_single_bar_graph(edge_weight_dict, 'Edge Weights', 'Sum of TMFG Edge Weights of Dual Portfolios', SPX_wins, sims=sims)
            

        # 1) Get vertex weights of triangles
        # 2) Graph vertex weights w/ winning portfolios
        vertex_weight_dict = {}
        for three_clique_stocks in stock_triads:
            s = 0
            for u, v in combinations(three_clique_stocks, 2):
                s += g.nodes[u]['weight']
            vertex_weight_dict[three_clique_stocks] = s
        # Charts.dual_edge_weights_single_bar_graph(vertex_weight_dict, 'Vertex Weights', 'Sum of TMFG Vertex Weights of Dual Portfolios', avg_returns_dict)
        # Charts.dual_edge_weights_single_bar_graph(vertex_weight_dict, 'Vertex Weights', 'Sum of TMFG Vertex Weights of Dual Portfolios', SPX_wins, sims=sims)

        # 1) Get edge & vertex weights of triangles
        # 2) Graph edge & vertex weights w/ winning portfolios 
        edge_and_vertex_weight_dict = {}
        for key in edge_weight_dict:
            edge_and_vertex_weight_dict[key] = edge_weight_dict[key] + vertex_weight_dict[key]
        # Charts.dual_edge_weights_single_bar_graph(edge_and_vertex_weight_dict, 'Edge & Vertex Weights', 'Sum of TMFG Edge & Vertex Weights of Dual Portfolios', avg_returns_dict)
        # Charts.dual_edge_weights_single_bar_graph(edge_and_vertex_weight_dict, 'Edge & Vertex Weights', 'Sum of TMFG Edge & Vertex Weights of Dual Portfolios', SPX_wins, sims=sims)

        # Set edge weights for dual centrality calculations
        nx.set_edge_attributes(dual, dual_edge_weight_dict, name='weight') # NOTE: This may change the graph drawing because some representations utilize edge attributes to determine how to draw the graph. This does not impact the actual structure (edges and vertices) of the graph.

        # Calculate centralities of TMFG and its dual and some median centralities to use for comparisons
        TMFG_degree_centrality = round_dict(nx.degree_centrality(g), 3)
        dual_TMFG_degree_centrality = round_dict(nx.degree_centrality(dual), 3)
        # dual_TMFG_degree_centrality = normalize_dict(dual_TMFG_degree_centrality)
        TMFG_eigenvector_centrality = round_dict(nx.eigenvector_centrality(g), 3)
        dual_TMFG_eigenvector_centrality = round_dict(nx.eigenvector_centrality(dual), 3)
        # dual_TMFG_eigenvector_centrality = normalize_dict(dual_TMFG_eigenvector_centrality)
        TMFG_betweenness_centrality = round_dict(nx.betweenness_centrality(g), 3)
        dual_TMFG_betweenness_centrality = round_dict(nx.betweenness_centrality(dual), 3)
        # dual_TMFG_betweenness_centrality = normalize_dict(dual_TMFG_betweenness_centrality)
        median_dual_TMFG_betweenness_centrality = np.median(list(dual_TMFG_betweenness_centrality.values()))
        TMFG_closeness_centrality = round_dict(nx.closeness_centrality(g), 3)
        dual_TMFG_closeness_centrality = round_dict(nx.closeness_centrality(dual), 3)
        # dual_TMFG_closeness_centrality = normalize_dict(dual_TMFG_closeness_centrality)
        median_dual_TMFG_closeness_centrality = np.median(list(dual_TMFG_closeness_centrality.values()))
        TMFG_katz_centrality = round_dict(nx.katz_centrality(g, alpha=0.005), 3)
        dual_TMFG_katz_centrality = round_dict(nx.katz_centrality(dual), 3)
        # dual_TMFG_katz_centrality = normalize_dict(dual_TMFG_katz_centrality)
        median_dual_TMFG_katz_centrality = np.median(list(dual_TMFG_katz_centrality.values()))
        TMFG_cflow_centrality = round_dict(nx.current_flow_betweenness_centrality(g), 3)
        dual_TMFG_cflow_centrality = round_dict(nx.current_flow_betweenness_centrality(dual), 3)
        # dual_TMFG_cflow_centrality = normalize_dict(dual_TMFG_cflow_centrality)
        median_dual_TMFG_cflow_centrality = np.median(list(dual_TMFG_cflow_centrality.values()))
        TMFG_commbtwn_centrality = round_dict(nx.communicability_betweenness_centrality(g), 3)
        dual_TMFG_commbtwn_centrality = round_dict(nx.communicability_betweenness_centrality(dual), 3)
        # dual_TMFG_commbtwn_centrality = normalize_dict(dual_TMFG_commbtwn_centrality)
        median_dual_TMFG_commbtwn_centrality = np.median(list(dual_TMFG_commbtwn_centrality.values()))
        TMFG_load_centrality = round_dict(nx.load_centrality(g), 3)
        dual_TMFG_load_centrality = round_dict(nx.load_centrality(dual), 3)
        # dual_TMFG_load_centrality = normalize_dict(dual_TMFG_load_centrality)
        median_dual_TMFG_load_centrality = np.median(list(dual_TMFG_load_centrality.values()))
        TMFG_harmonic_centrality = round_dict(nx.harmonic_centrality(g), 3)
        dual_TMFG_harmonic_centrality = round_dict(nx.harmonic_centrality(dual), 3)
        # dual_TMFG_harmonic_centrality = normalize_dict(dual_TMFG_harmonic_centrality)
        median_dual_TMFG_harmonic_centrality = np.median(list(dual_TMFG_harmonic_centrality.values()))
        TMFG_percolation_centrality = round_dict(nx.percolation_centrality(g), 3)
        dual_TMFG_percolation_centrality = round_dict(nx.percolation_centrality(dual), 3)
        # dual_TMFG_percolation_centrality = normalize_dict(dual_TMFG_percolation_centrality)
        median_dual_TMFG_percolation_centrality = np.median(list(dual_TMFG_percolation_centrality.values()))
        TMFG_2ndorder_centrality = round_dict(nx.second_order_centrality(g), 3)
        dual_TMFG_2ndorder_centrality = round_dict(nx.second_order_centrality(dual), 3)
        # dual_TMFG_2ndorder_centrality = normalize_dict(dual_TMFG_2ndorder_centrality)
        median_dual_TMFG_2ndorder_centrality = np.median(list(dual_TMFG_2ndorder_centrality.values()))

        TMFG_voterank = nx.voterank(g)
        dual_TMFG_voterank = nx.voterank(dual)
        TMFG_pagerank = nx.pagerank(g)
        dual_TMFG_pagerank = nx.pagerank(dual)
        TMFG_estrada_index = nx.estrada_index(g)
        dual_TMFG_estrada_index = nx.estrada_index(dual)

        # Store dual TMFG centrality measures as a dataframe
        SPX_win_percentages = {portfolio: wins/sims for portfolio, wins in SPX_wins.items()}
        dual_TMFG_info = {
            'degree': dual_TMFG_degree_centrality,
            'eigenvector': dual_TMFG_eigenvector_centrality,
            'betweenness': dual_TMFG_betweenness_centrality,
            'closeness': dual_TMFG_closeness_centrality,
            'katz': dual_TMFG_katz_centrality,
            'current flow': dual_TMFG_cflow_centrality,
            'communicability betweenness': dual_TMFG_commbtwn_centrality,
            'load': dual_TMFG_load_centrality,
            'harmonic': dual_TMFG_harmonic_centrality,
            'percolation': dual_TMFG_percolation_centrality,
            'second order': dual_TMFG_2ndorder_centrality,
            'pagerank': dual_TMFG_pagerank,
            'average return': avg_returns_dict,
            'SPX win percentage': SPX_win_percentages
        }
        print(dual_TMFG_info)
        df = pd.DataFrame(dual_TMFG_info)

        for n in dual.nodes:
            dual.nodes[n]['avg_return_label'] = int(avg_returns_dict[n] > 0)
            dual.nodes[n]['SPX_win_label'] = int(SPX_win_percentages[n] > 0.5)

        dual_TMFG_info['TMFG size'] = len(stocks)
        dual_TMFG_info['TMFG edge weight sum'] = edge_weight_dict
        dual_TMFG_info['TMFG vert weight sum'] = vertex_weight_dict
        dual_TMFG_info['vert weight'] = portfolio_historical_percent_changes
        dual_TMFG_info['incindent edge weight sum'] = incindent_edge_sum_dict

        if sims != 1:
            Database.create_dual_graph_table(dual_TMFG_info, date_dict)
        
        if analyze_node_features:
            dual_TMFG_info.pop('average return')
            dual_TMFG_info.pop('SPX win percentage')
            GAT, feature_names = ML_Analysis.networkx_to_GAT(dual, dual_TMFG_info, 'avg_return_label')
            model = ML_Analysis.PortfolioGAT(num_features=GAT.x.shape[1])
            success_metric = 'Average Return'
            ML_Analysis.trainGAT(model, GAT, feature_names, success_metric, epochs=500)
            count = 0
            for k, v in avg_returns_dict.items():
                if v > 0:
                    count += 1
            print('Percentage of successful portfolios:' + str(count/len(avg_returns_dict.keys())))

            GAT, feature_names = ML_Analysis.networkx_to_GAT(dual, dual_TMFG_info, 'SPX_win_label')
            model = ML_Analysis.PortfolioGAT(num_features=GAT.x.shape[1])
            success_metric = 'SPX Outperformance Rate'
            ML_Analysis.trainGAT(model, GAT, feature_names, success_metric, epochs=500)
            count = 0
            for k, v in SPX_win_percentages.items():
                if v > 0.5:
                    count += 1
            print('Percentage of successful portfolios:' + str(count/len(SPX_win_percentages.keys())))

        f.write(f'Stock frequencies: {stock_frequency}\n')
        f.write('------------------------------------------------------\n')
        f.write(f'TMFG Degree Centrality: {TMFG_degree_centrality}\n')
        f.write(f'Dual TMFG Degree Centrality: {dual_TMFG_degree_centrality}\n')
        f.write(f'TMFG Eigenvector Centrality: {TMFG_eigenvector_centrality}\n')
        f.write(f'Dual TMFG Eigenvector Centrality: {dual_TMFG_eigenvector_centrality}\n')
        f.write(f'TMFG Betweenness Centrality: {TMFG_betweenness_centrality}\n')
        f.write(f'Dual TMFG Betweenness Centrality: {dual_TMFG_betweenness_centrality}\n')
        f.write(f'Median Dual TMFG Betweenness Centrality: {median_dual_TMFG_betweenness_centrality}\n')
        f.write(f'TMFG Closeness Centrality: {TMFG_closeness_centrality}\n')
        f.write(f'Dual TMFG Closeness Centrality: {dual_TMFG_closeness_centrality}\n')
        f.write(f'Median Dual TMFG Closeness Centrality: {median_dual_TMFG_closeness_centrality}\n')
        f.write(f'TMFG Katz Centrality: {TMFG_katz_centrality}\n')
        f.write(f'Dual TMFG Katz Centrality: {dual_TMFG_katz_centrality}\n')
        f.write(f'Median Dual TMFG Katz Centrality: {median_dual_TMFG_katz_centrality}\n')
        f.write(f'TMFG Current Flow Centrality: {TMFG_cflow_centrality}\n')
        f.write(f'Dual TMFG Current Flow Centrality: {dual_TMFG_cflow_centrality}\n')
        f.write(f'Median Dual TMFG Current Flow Centrality: {median_dual_TMFG_cflow_centrality}\n')
        f.write(f'TMFG Communicability Betweenness Centrality: {TMFG_commbtwn_centrality}\n')
        f.write(f'Dual TMFG Communicability Betweenness Centrality: {dual_TMFG_commbtwn_centrality}\n')
        f.write(f'Median Dual TMFG Communicability Betweenness Centrality: {median_dual_TMFG_commbtwn_centrality}\n')
        f.write(f'TMFG Load Centrality: {TMFG_load_centrality}\n')
        f.write(f'Dual TMFG Load Centrality: {dual_TMFG_load_centrality}\n')
        f.write(f'Median Dual TMFG Load Centrality: {median_dual_TMFG_load_centrality}\n')
        f.write(f'TMFG Harmonic Centrality: {TMFG_harmonic_centrality}\n')
        f.write(f'Dual TMFG Harmonic Centrality: {dual_TMFG_harmonic_centrality}\n')
        f.write(f'Median Dual TMFG Harmonic Centrality: {median_dual_TMFG_harmonic_centrality}\n')
        f.write(f'TMFG Percolation Centrality: {TMFG_percolation_centrality}\n')
        f.write(f'Dual TMFG Percolation Centrality: {dual_TMFG_percolation_centrality}\n')
        f.write(f'Median Dual TMFG Percolation Centrality: {median_dual_TMFG_percolation_centrality}\n')
        f.write(f'TMFG Second Order Centrality: {TMFG_2ndorder_centrality}\n')
        f.write(f'Dual TMFG Second Order Centrality: {dual_TMFG_2ndorder_centrality}\n')
        f.write(f'Median Dual TMFG Second Order Centrality: {median_dual_TMFG_2ndorder_centrality}\n')
        f.write(f'TMFG Voterank: {TMFG_voterank}\n')
        f.write(f'Dual TMFG Voterank: {dual_TMFG_voterank}\n')
        f.write(f'TMFG Pagerank: {TMFG_pagerank}\n')
        f.write(f'Dual TMFG Pagerank: {dual_TMFG_pagerank}\n')
        f.write(f'TMFG Estrada Index: {TMFG_estrada_index}\n')
        f.write(f'Dual TMFG Estrada Index: {dual_TMFG_estrada_index}\n')
        f.write(f'S&P 500 Wins Dictionary: {SPX_wins}\n')
        f.write('------------------------------------------------------\n')

        # Graph Dual TMFG data
        # Graphing.dual_single_bar_graph_wins(dual_TMFG_eigenvector_centrality, 'Eigenvector Centrality', 'Eigenvector Centrality Per Portfolio', SPX_wins, sims)
        # Graphing.dual_single_bar_graph_wins(dual_TMFG_degree_centrality, 'Degree Centrality', 'Degree Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_betweenness_centrality, 'Betweenness Centrality', 'Betweenness Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_closeness_centrality, 'Closeness Centrality', 'Closeness Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_commbtwn_centrality, 'Communicability Betweenness Centrality', 'Communicability Betweenness Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_load_centrality, 'Load Centrality', 'Load Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_percolation_centrality, 'Percolation Centrality', 'Percolation Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_cflow_centrality, 'Current Flow Centrality', 'Current Flow Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_2ndorder_centrality, 'Second Order Centrality', 'Second Order Centrality Per Portfolio', SPX_wins, sims)
        Charts.dual_single_bar_graph_wins(dual_TMFG_harmonic_centrality, 'Harmonic Centrality', 'Harmonic Centrality Per Portfolio', SPX_wins, sims)
        # Graphing.dual_single_bar_graph_wins(dual_TMFG_katz_centrality, 'Katz Centrality', 'Katz Centrality Per Portfolio', SPX_wins, sims)

        # Graphing.dual_single_bar_graph_returns(dual_TMFG_eigenvector_centrality, 'Eigenvector Centrality', 'Eigenvector Centrality Per Portfolio', avg_returns_dict)
        # Graphing.dual_single_bar_graph_returns(dual_TMFG_degree_centrality, 'Degree Centrality', 'Degree Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_betweenness_centrality, 'Betweenness Centrality', 'Betweenness Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_closeness_centrality, 'Closeness Centrality', 'Closeness Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_commbtwn_centrality, 'Communicability Betweenness Centrality', 'Communicability Betweenness Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_load_centrality, 'Load Centrality', 'Load Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_percolation_centrality, 'Percolation Centrality', 'Percolation Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_cflow_centrality, 'Current Flow Centrality', 'Current Flow Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_2ndorder_centrality, 'Second Order Centrality', 'Second Order Centrality Per Portfolio', avg_returns_dict)
        Charts.dual_single_bar_graph_returns(dual_TMFG_harmonic_centrality, 'Harmonic Centrality', 'Harmonic Centrality Per Portfolio', avg_returns_dict)
        # Graphing.dual_single_bar_graph_returns(dual_TMFG_katz_centrality, 'Katz Centrality', 'Katz Centrality Per Portfolio', avg_returns_dict)
        
        
        dual_TMFG_super_centrality = {}
        # NOTE: Doesn't include second order centrality bc a smaller values corresponds with a higher centrality
        for key in dual_TMFG_betweenness_centrality:
            dual_TMFG_super_centrality[key] = dual_TMFG_betweenness_centrality[key] + 0*dual_TMFG_closeness_centrality[key] + dual_TMFG_commbtwn_centrality[key] + dual_TMFG_load_centrality[key] + dual_TMFG_percolation_centrality[key] + dual_TMFG_cflow_centrality[key] + 0*dual_TMFG_harmonic_centrality[key]
        # Graphing.dual_single_bar_graph_wins(dual_TMFG_super_centrality, 'Portfolios', 'Super Centrality', 'Super Centrality Per Portfolio', SPX_wins, sims)

        # Graphing.compare_bar_graphs(dual_TMFG_betweenness_centrality, dual_TMFG_closeness_centrality, 'Betweenness Centrality', 'Closeness Centrality', 'Centrality Score', 'Betweenness vs Closeness Centrality of Dual TMFG Portfolios', SPX_wins, sims)
        # Graphing.compare_bar_graphs(dual_TMFG_commbtwn_centrality, dual_TMFG_load_centrality, 'Communicability Betweenness Centrality', 'Load Centrality', 'Centrality Score', 'Communicability Betweenness vs Load Centrality of Dual TMFG Portfolios', SPX_wins, sims)
        # Graphing.compare_bar_graphs(dual_TMFG_percolation_centrality, dual_TMFG_cflow_centrality, 'Percolation Centrality', 'Current Flow Centrality', 'Centrality Score', 'Percolation vs Current Flow Centrality of Dual TMFG Portfolios', SPX_wins, sims)

        # Graphing.tmfg_single_bar_graph(TMFG_eigenvector_centrality, 'Eigenvector Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_degree_centrality, 'Degree Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_betweenness_centrality, 'Betweenness Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_closeness_centrality, 'Closeness Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_commbtwn_centrality, 'Communicability Betweenness Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_load_centrality, 'Load Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_percolation_centrality, 'Percolation Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_cflow_centrality, 'Current Flow Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_2ndorder_centrality, 'Second Order Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_harmonic_centrality, 'Harmonic Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_katz_centrality, 'Katz Centrality')
        # Graphing.tmfg_single_bar_graph(TMFG_pagerank, 'Pagerank')

        f.write(f'Total SPX wins: {sum(SPX_wins.values())/len(dual.nodes)}\n')
        f.write(f'Average return: {sum(avg_returns_dict.values())/len(dual.nodes)}\n')
        end_time = time.time()
        f.write(f'Execution time: {end_time - start_time:.4f} seconds')






# Identify dates that are below the n-day EMA of a portfolio
def find_dips(stocks: list, n: int, first_date: str, last_date: str):
    overlap_dates = None
    first_date = pd.to_datetime(first_date)
    last_date = pd.to_datetime(last_date)
    for s in stocks:
        df = pd.read_csv(
            f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
            parse_dates=['Date'],
            date_format='%m/%d/%y')
        df = df[(df['Date'] >= first_date) & (df['Date'] <= last_date)]
        ema_open = df['Open'].ewm(span=n, adjust=False).mean()
        # Find dates where the current opening price is less than the n-day EMA
        mask = df['Open'] < ema_open
        dates_below_ema = set(df.loc[mask, 'Date'])
        if overlap_dates is None:
            overlap_dates = dates_below_ema  # Initialize on first iteration
        else:
            overlap_dates = overlap_dates & dates_below_ema  # Intersect sets
    return list(overlap_dates)

# Identify dates that are above the n-day EMA of a portfolio
def find_peaks(stocks: list, n: int, first_date: str, last_date: str):
    overlap_dates = None
    first_date = pd.to_datetime(first_date)
    last_date = pd.to_datetime(last_date)
    for s in stocks:
        df = pd.read_csv(
            f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
            parse_dates=['Date'],
            date_format='%m/%d/%y')
        df = df[(df['Date'] >= first_date) & (df['Date'] <= last_date)]
        df['Date'] = df['Date'].dt.date
        ema_open = df['Open'].ewm(span=n, adjust=False).mean()
        # Find dates where the current opening price is less than the n-day EMA
        mask = df['Open'] > ema_open
        dates_below_ema = set(df.loc[mask, 'Date'])
        if overlap_dates is None:
            overlap_dates = dates_below_ema  # Initialize on first iteration
        else:
            overlap_dates = overlap_dates & dates_below_ema  # Intersect sets
    return list(overlap_dates)



def simulate_all_3combos(sims: int, seed: int, stocks: list[str], buy_date: str, sell1: str, sell2: str):
    start_time = time.time()
    avg_return_dict = {}
    SPX_wins_dict = {}
    with open(os.path.join('Simulations', f'{sims}_3ComboSims_{buy_date}_and_{sell1}_to_{sell2}.txt'), 'w', newline='') as f:
        f.write(f'Stocks ({len(stocks)}): {stocks}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Simulations: {sims}\n')
        f.write(f'Starting date: {buy_date}\n')
        f.write(f'Ending date range: {sell1} to {sell2}\n')
        f.write('------------------------------------------------------\n')

        # Generate random sell dates using a random stock (doesn't matter which stock is used since all stocks use the same dates)
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

        # Run variance-minimized simulations of all possible triads and count wins
        combos = list(combinations(stocks, 3))
        for c in combos:
            
            SPX_beat_count = 0
            triad_percent_changes = []
            portfolio_returns = []

            for i in range(sims):
                # Pick a sell random date
                random_sell_date = np.random.choice(valid_dates)

                percent_changes = pd.DataFrame()
                random_period_percent_changes = []
                for s in c:
                    df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, {s}.csv',
                    parse_dates=['Date'],
                    date_format='%m/%d/%y')

                    # Set Date as index
                    df.set_index('Date', inplace=True)
                    # Remove spaces in column names
                    df.columns = df.columns.str.strip()

                    # Use pd.Timestamp for the date lookup
                    target_date1 = pd.Timestamp(buy_date)
                    target_date2 = pd.Timestamp(random_sell_date)

                    # Get the opening value for a specific date
                    open_value = df.loc[target_date1, 'Open']
                    close_value = df.loc[target_date2, 'Close']
                    random_period_percent_changes.append((close_value - open_value)/open_value * 100)

                    # Change Date back into a column so it can be accessed as normal
                    df.reset_index(inplace=True)
                    # Remove some recent data to analyze profits in the past
                    cutoff_date = pd.to_datetime(buy_date)
                    df = df[df['Date'] <= cutoff_date]
                    # Calculate daily percent change for a stock and add it as a new column to df.
                    df['Percent Change'] = ((df['Close'] - df['Open']) / df['Open']) * 100
                    # Store percent changes into a single DataFrame, with each column representing a different stock.
                    percent_changes[s] = df['Percent Change']
                
                # Make covariance matrix of a 3-stock portoflio
                cov_mat = percent_changes.cov()
                # Calculate variance-minimizing stock allocations
                mat = np.array(2*cov_mat)
                mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
                mat = np.hstack([mat, np.ones((mat.shape[0], 1))])
                mat[-1][-1] = 0
                mat_inv = np.linalg.inv(mat)
                B = np.zeros((mat.shape[1], 1))
                B[-1] = 1
                stock_allocations = np.dot(mat_inv, B)
                # Remove last row of stock_allocations and retrieve its important values
                stock_allocations = np.delete(stock_allocations, -1, axis=0)
                x = np.dot(np.transpose(stock_allocations), cov_mat)
                variance = np.dot(x, stock_allocations)

                print('Portfolio minimum variance: ' + str(variance[0][0]))
                print('Portfolio allocations: ' + str(c[0]) + ': ' + str(stock_allocations[0][0]) + '\n                       ' + str(c[1]) + ': ' + str(stock_allocations[1][0]) + '\n                       ' + str(c[2]) + ': ' + str(stock_allocations[2][0]))
                

                random_period_portfolio_percent_change = 0
                for i in range(len(random_period_percent_changes)):
                    random_period_portfolio_percent_change += random_period_percent_changes[i] * stock_allocations[i][0]
                
                print(f'Portfolio percent change from {buy_date} to {random_sell_date}: ' + str(random_period_portfolio_percent_change))
                print('--------------------------------------------\n')

                portfolio_returns.append(random_period_portfolio_percent_change)
                

                df = pd.read_csv(
                    f'Data2015-2025/HistoricalPrices 2015 - 2025, SPX.csv',
                    parse_dates=['Date'],
                    date_format='%m/%d/%y')

                # Set Date as index
                df.set_index('Date', inplace=True)
                # Remove spaces in column names
                df.columns = df.columns.str.strip()

                # Use pd.Timestamp for the date lookup
                target_date1 = pd.Timestamp(buy_date)
                target_date2 = pd.Timestamp(random_sell_date)

                open_value = df.loc[target_date1, 'Open']
                close_value = df.loc[target_date2, 'Close']
                SPX_percent_change = (close_value - open_value)/open_value * 100
                print(f'SPX percent change from {buy_date} to {random_sell_date}: ' + str(SPX_percent_change))
                print('--------------------------------------------\n')
                if random_period_portfolio_percent_change > SPX_percent_change:
                    SPX_beat_count += 1
                triad_percent_changes.append(random_period_portfolio_percent_change)
            
            avg_return = sum(triad_percent_changes)/len(triad_percent_changes)
            SPX_wins_dict[c] = SPX_beat_count
            avg_return_dict[c] = avg_return

            f.write(f'Portfolio: {c}\n')
            f.write(f'Average portfolio return: {avg_return}\n')
            f.write(f'Number of times SPX was beat: {SPX_beat_count}\n')
            f.write('------------------------------------------------------\n')
        
        total_SPX_wins = 0
        for k, v in SPX_wins_dict.items():
            total_SPX_wins += v
        total_return = 0
        for k, v in avg_return_dict.items():
            total_return += v

        f.write(f'Average SPX wins: {total_SPX_wins/len(combos)}\n')
        f.write(f'Average return: {total_return/len(combos)}\n')
        end_time = time.time()
        f.write(f'Execution time: {end_time - start_time:.4f} seconds\n')
        # TODO: Calculate average return and average SP500 wins across all portfolios

def analyze_portfolio_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    total_portfolios = 0
    spx_beats_over_5 = 0
    return_above_0 = 0

    for i in range(len(lines)):
        if lines[i].startswith("Portfolio"):
            total_portfolios += 1

            # Look ahead to find return and SPX beat lines
            ret_line = lines[i + 1].strip()
            spx_line = lines[i + 2].strip()

            # Extract values
            if "Average portfolio return" in ret_line:
                try:
                    return_val = float(ret_line.split(":")[1])
                    if return_val > 0:
                        return_above_0 += 1
                except ValueError:
                    pass

            if "Number of times SPX was beat" in spx_line:
                try:
                    spx_val = int(spx_line.split(":")[1])
                    if spx_val > 5:
                        spx_beats_over_5 += 1
                except ValueError:
                    pass

    # Compute proportions
    prop_spx_beats = spx_beats_over_5 / total_portfolios if total_portfolios > 0 else 0
    prop_return_pos = return_above_0 / total_portfolios if total_portfolios > 0 else 0

    print(f"Proportion of portfolios that beat SPX more than 5 times: {prop_spx_beats:.3f}")
    print(f"Proportion of portfolios with return > 0: {prop_return_pos:.3f}")



random.seed(1)
stocks = ['DIS', 'KO', 'ADBE', 'MRK', 'KMI', 'AAPL', 'JNJ', 'CVS', 'COST', 'T', 'BA', 'EA', 'HAS', 'HD', 'HSY', 'LLY', 'NFLX', 'NKE', 'V', 'JPM', 'FDX', 'KR', 'KHC', 'LULU', 'MA', 'BBY', 'GOOG', 'ALL', 'KMX', 'MNST', 
          'AZO', 'COF', 'WFC', 'XOM', 'CVX', 'MAR', 'MCD', 'ORCL', 'UNH', 'EBAY', 'CPB', 'DPZ', 'JBHT', 'TSN', 'WYNN', 'DLTR', 'EXPE', 'JNPR', 'SJM', 'NTAP', 'ATO', 'AFL', 'DXCM', 'MCHP', 'DAL', 'TGT', 'PCG',
          'CSGP', 'WTW', 'AXON']
stocks2 = random.sample(stocks, 50)
stocks3 = random.sample(stocks, 40)
stocks4 = random.sample(stocks, 30)
stocks5 = random.sample(stocks, 20)
stocks6 = random.sample(stocks, 10)

# simulate_all_3combos(10, 1, stocks, '2023-01-31', '2023-02-01', '2023-02-28') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-02-28', '2023-03-01', '2023-03-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-03-31', '2023-04-03', '2023-04-28') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-04-28', '2023-05-01', '2023-05-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-05-31', '2023-06-01', '2023-06-30') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-06-30', '2023-07-03', '2023-07-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-07-31', '2023-08-01', '2023-08-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-08-31', '2023-09-01', '2023-09-29') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-09-29', '2023-10-02', '2023-10-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-10-31', '2023-11-01', '2023-11-30') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-11-30', '2023-12-01', '2023-12-29') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2023-12-29', '2024-01-02', '2024-01-31')


# simulate_all_3combos(10, 1, stocks, '2024-01-31', '2024-02-01', '2024-02-29') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-02-29', '2024-03-01', '2024-03-28') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-03-28', '2024-04-01', '2024-04-30') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-04-30', '2024-05-01', '2024-05-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-05-31', '2024-06-03', '2024-06-28') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-06-28', '2024-07-01', '2024-07-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-07-31', '2024-08-01', '2024-08-30') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-08-30', '2024-09-03', '2024-09-30') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-09-30', '2024-10-01', '2024-10-31') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-10-31', '2024-11-01', '2024-11-29') NOTE: Done
# simulate_all_3combos(10, 1, stocks, '2024-11-29', '2024-12-02', '2024-12-31') NOTE: Done
 



# analyze_portfolio_file("Simulations/10_3ComboSims_2023-12-29_and_2024-01-02_to_2024-01-31.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-11-30_and_2023-12-01_to_2023-12-29.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-10-31_and_2023-11-01_to_2023-11-30.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-09-29_and_2023-10-02_to_2023-10-31.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-08-31_and_2023-09-01_to_2023-09-29.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-07-31_and_2023-08-01_to_2023-08-31.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-06-30_and_2023-07-03_to_2023-07-31.txt")
# analyze_portfolio_file("Simulations/10_3ComboSims_2023-05-31_and_2023-06-01_to_2023-06-30.txt")



# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-01-02', '2024-01-31'), make_TMFG(stocks, '2024-02-01', '2024-02-29'))
# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-06-03', '2024-06-28'), make_TMFG(stocks, '2024-07-01', '2024-07-31'))




# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-01-02', '2024-01-31'), make_TMFG(stocks, '2024-01-17', '2024-02-15'))
# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-01-17', '2024-02-15'), make_TMFG(stocks, '2024-02-01', '2024-02-29'))
# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-02-01', '2024-02-29'), make_TMFG(stocks, '2024-02-15', '2024-03-15'))
# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-01-02', '2024-01-03'), make_TMFG(stocks, '2024-01-03', '2024-01-04'))
# ML_Analysis.visualize_tmfg_prediction(make_TMFG(stocks, '2024-01-03', '2024-01-04'), make_TMFG(stocks, '2024-01-04', '2024-01-05'))

# timed_dual_dictionary = {
#     '2024-01-02': make_TMFG(stocks, '2024-01-02', '2024-01-31'), 
#     '2024-02-01': make_TMFG(stocks, '2024-02-01', '2024-02-29'),
#     '2024-03-01': make_TMFG(stocks, '2024-03-01', '2024-03-28'),
#     '2024-04-01': make_TMFG(stocks, '2024-04-01', '2024-04-30'),
#     '2024-05-01': make_TMFG(stocks, '2024-05-01', '2024-05-31'), 
#     '2024-06-03': make_TMFG(stocks, '2024-06-03', '2024-06-28'),
#     '2024-07-01': make_TMFG(stocks, '2024-07-01', '2024-07-31'),
#     '2024-08-01': make_TMFG(stocks, '2024-08-01', '2024-08-30'),
#     '2024-09-03': make_TMFG(stocks, '2024-09-03', '2024-09-30'),
#     '2024-10-01': make_TMFG(stocks, '2024-10-01', '2024-10-31'),
#     '2024-11-01': make_TMFG(stocks, '2024-11-01', '2024-11-29')
# }
# timed_dual_dictionary = {
#     '2024-01-02': make_TMFG(stocks, '2024-01-02', '2024-01-16'),
#     '2024-01-17': make_TMFG(stocks, '2024-01-17', '2024-01-31'),
#     '2024-02-01': make_TMFG(stocks, '2024-02-01', '2024-02-15'),
#     '2024-02-15': make_TMFG(stocks, '2024-02-15', '2024-02-29'),
#     '2024-03-01': make_TMFG(stocks, '2024-03-01', '2024-03-15'),
#     '2024-03-15': make_TMFG(stocks, '2024-03-15', '2024-03-28'),
#     '2024-04-01': make_TMFG(stocks, '2024-04-01', '2024-04-15'),
#     '2024-04-15': make_TMFG(stocks, '2024-04-15', '2024-04-30')
# }
# timed_dual_dictionary = {
#     '2024-01-02': make_TMFG(stocks, '2024-01-02', '2024-01-31'), 
#     '2024-02-01': make_TMFG(stocks, '2024-01-17', '2024-02-15'),
#     '2024-03-01': make_TMFG(stocks, '2024-02-01', '2024-02-29'),
#     '2024-02-15': make_TMFG(stocks, '2024-02-15', '2024-03-15'),
#     '2024-03-01': make_TMFG(stocks, '2024-03-01', '2024-03-28'),
#     '2024-03-15': make_TMFG(stocks, '2024-03-15', '2024-04-15')
# }
# predicted_graph = ML_Analysis.predict_tmfg_from_temporal_graphs(timed_dual_dictionary, len(stocks))
# # real_graph = make_TMFG(stocks, '2024-05-01', '2024-05-15')
# real_graph = make_TMFG(stocks, '2024-04-01', '2024-04-30')
# ML_Analysis.visualize_tmfg_prediction(predicted_graph, real_graph)
# ML_Analysis.visualize_tmfg_prediction(real_graph, predicted_graph)
# ML_Analysis.evaluate_graph_prediction(predicted_graph, real_graph)


# simulation2(1, 1, stocks, '2021-01-01', '2022-01-01', '2024-01-01', '2025-01-01')

# buy_dates = find_dips(['ADBE', 'HD', 'HSY'], 50, '2022-01-04', '2023-01-01')
# sell_dates = find_peaks(['ADBE', 'EA', 'LLY'], 50, '2023-01-04', '2024-01-01')
# print(buy_dates)
# print(sell_dates)


# Monthly simulations of 2024
# simulate(1, 1, stocks, '2024-01-02', '2024-01-16', '2024-01-17', '2024-01-31', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2024-01-17', '2024-01-31', '2024-02-01', '2024-02-15')
# simulate(100, 1, stocks, '2024-02-01', '2024-02-15', '2024-02-15', '2024-02-29')


simulate(100, 1, stocks, '2023-01-03', '2023-01-31', '2023-02-01', '2023-02-28') 
# simulate(100, 1, stocks, '2023-02-01', '2023-02-28', '2023-03-01', '2023-03-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-03-01', '2023-03-31', '2023-04-03', '2023-04-28', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-04-03', '2023-04-28', '2023-05-01', '2023-05-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-05-01', '2023-05-31', '2023-06-01', '2023-06-30', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-06-01', '2023-06-30', '2023-07-03', '2023-07-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-07-03', '2023-07-31', '2023-08-01', '2023-08-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-08-01', '2023-08-31', '2023-09-01', '2023-09-29', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-09-01', '2023-09-29', '2023-10-02', '2023-10-31', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2023-10-02', '2023-10-31', '2023-11-01', '2023-11-30', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-11-01', '2023-11-29', '2023-12-01', '2023-12-29', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2023-12-01', '2023-12-29', '2024-01-02', '2024-01-31', frequently_appearing_check=True) 


# simulate(100, 1, stocks, '2024-01-02', '2024-01-31', '2024-02-01', '2024-02-29', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-02-01', '2024-02-29', '2024-03-01', '2024-03-28', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-03-01', '2024-03-28', '2024-04-01', '2024-04-30', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2024-04-01', '2024-04-30', '2024-05-01', '2024-05-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-05-01', '2024-05-31', '2024-06-03', '2024-06-28', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-06-03', '2024-06-28', '2024-07-01', '2024-07-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-07-01', '2024-07-31', '2024-08-01', '2024-08-30', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-08-01', '2024-08-30', '2024-09-03', '2024-09-30', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2024-09-03', '2024-09-30', '2024-10-01', '2024-10-31', frequently_appearing_check=True) 
# simulate(100, 1, stocks, '2024-10-01', '2024-10-31', '2024-11-01', '2024-11-29', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2024-11-01', '2024-11-29', '2024-12-02', '2024-12-31', frequently_appearing_check=True)

# Charts.visualize_stock_data('SPX', '2023-01-03', '2023-01-31', '2023-02-01', '2023-02-28')
# Charts.visualize_stock_data('SPX', '2023-02-01', '2023-02-28', '2023-03-01', '2023-03-31')
# Charts.visualize_stock_data('SPX', '2023-03-01', '2023-03-31', '2023-04-03', '2023-04-28')
# Charts.visualize_stock_data('SPX', '2023-04-03', '2023-04-28', '2023-05-01', '2023-05-31')
# Charts.visualize_stock_data('SPX', '2023-05-01', '2023-05-31', '2023-06-01', '2023-06-30')
# Charts.visualize_stock_data('SPX', '2023-06-01', '2023-06-30', '2023-07-03', '2023-07-31')
# Charts.visualize_stock_data('SPX', '2023-07-03', '2023-07-31', '2023-08-01', '2023-08-31')
# Charts.visualize_stock_data('SPX', '2023-08-01', '2023-08-31', '2023-09-01', '2023-09-29')
# Charts.visualize_stock_data('SPX', '2023-09-01', '2023-09-29', '2023-10-02', '2023-10-30')
# Charts.visualize_stock_data('SPX', '2023-10-02', '2023-10-30', '2023-11-01', '2023-11-30')
# Charts.visualize_stock_data('SPX', '2023-11-01', '2023-11-30', '2023-12-02', '2023-12-29')
# Charts.visualize_stock_data('SPX', '2023-12-02', '2023-12-29', '2024-01-02', '2024-01-31')
# Charts.visualize_stock_data('SPX', '2024-01-02', '2024-01-31', '2024-02-01', '2024-02-29')
# Charts.visualize_stock_data('SPX', '2024-02-01', '2024-02-29', '2024-03-01', '2024-03-28')
# Charts.visualize_stock_data('SPX', '2024-03-01', '2024-03-28', '2024-04-01', '2024-04-30')
# Charts.visualize_stock_data('SPX', '2024-04-01', '2024-04-30', '2024-05-01', '2024-05-31')
# Charts.visualize_stock_data('SPX', '2024-05-01', '2024-05-31', '2024-06-03', '2024-06-28')
# Charts.visualize_stock_data('SPX', '2024-06-03', '2024-06-28', '2024-07-01', '2024-07-31')
# Charts.visualize_stock_data('SPX', '2024-07-01', '2024-07-31', '2024-08-01', '2024-08-30')
# Charts.visualize_stock_data('SPX', '2024-08-01', '2024-08-30', '2024-09-03', '2024-09-30')
# Charts.visualize_stock_data('SPX', '2024-09-03', '2024-09-30', '2024-10-01', '2024-10-31')
# Charts.visualize_stock_data('SPX', '2024-10-01', '2024-10-31', '2024-11-01', '2024-11-29')
# Charts.visualize_stock_data('SPX', '2024-11-01', '2024-11-29', '2024-12-02', '2024-12-31')




# simulate(20, 1, stocks, '2021-01-04', '2021-12-31', '2022-01-03', '2022-12-30')
# simulate(100, 1, stocks, '2021-07-01', '2021-12-31', '2022-01-03', '2022-06-30')
# simulate(100, 1, stocks, '2021-10-03', '2021-12-31', '2022-01-03', '2022-03-31')

# simulate(1, 1, stocks, '2022-01-03', '2022-12-30', '2023-01-03', '2023-12-29')
# simulate(100, 1, stocks, '2022-07-01', '2022-12-30', '2023-01-03', '2023-06-30')
# simulate(100, 1, stocks, '2022-10-03', '2022-12-30', '2023-01-03', '2023-03-31')

# Upward SPX 
# simulate(100, 1, stocks, '2023-11-14', '2024-01-24', '2024-01-25', '2024-04-05', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2021-03-04', '2021-10-28', '2021-10-29', '2021-12-17', frequently_appearing_check=True)
# simulate(1, 1, stocks, '2021-07-29', '2021-10-28', '2021-10-29', '2024-12-31') # '2021-12-17'
# simulate(100, 1, stocks, '2020-12-23', '2021-02-05', '2021-02-08', '2021-05-14', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2023-03-13', '2023-05-26', '2023-05-30', '2023-07-28', frequently_appearing_check=True)



# simulate(1, 1, stocks, '2020-10-16', '2021-05-17', '2021-05-18', '2021-12-17')   # NOTE: Long-term --> Insignificant findings

# simulate_all_3combos(1, 1, stocks, '2024-01-24', '2024-01-25', '2024-04-05')

# Downward SPX
# simulate(100, 1, stocks, '2022-03-29', '2022-04-22', '2022-04-25', '2022-05-20', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2022-08-12', '2022-09-21', '2022-09-22', '2022-10-14', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2023-07-28', '2023-09-12', '2023-09-13', '2023-11-02', frequently_appearing_check=True)

# # Stagnating SPX
# simulate(100, 1, stocks, '2024-10-11', '2024-11-19', '2024-11-20', '2024-12-31', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2021-12-16', '2022-03-15', '2022-03-16', '2022-07-14', frequently_appearing_check=True)
# simulate(100, 1, stocks, '2020-08-26', '2020-10-02', '2020-10-06', '2020-11-05', frequently_appearing_check=True)




# simulate_all_3combos(1, 1, stocks, '2024-01-24', '2024-01-25', '2024-04-05')
